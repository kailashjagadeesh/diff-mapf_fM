import time
import heapq
import numpy as np

from core.planner.cbs import ConflictBasedSearch


class CardinalCBSNode:
    """CBS node with conflict_density as secondary sort key (Idea 3)."""

    def __init__(self, plan_indices, constraints, cost, node_id, parent_id, conflict_density=0.0):
        self.cost = cost
        self.constraints = constraints
        self.plan_indices = plan_indices
        self.id = node_id
        self.parent_id = parent_id
        self.conflict_density = conflict_density

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        # Prefer lower density (easier conflicts) when costs are equal
        return self.conflict_density < other.conflict_density


class CardinalCBS(ConflictBasedSearch):
    """
    CBS with cardinality-aware conflict selection and conditional repair, inspired by
    ICBS (Boyarski et al., IJCAI 2015) and CBSH/WDG (Felner et al., ICAPS 2018).

    Changes vs standard CBS:
    - Idea 1: Earliest-timestep conflict selection (not first-found by agent-pair order)
    - Idea 2: Classify conflicts as cardinal / semi-cardinal / non-cardinal by scanning B candidates
    - Idea 3: Conflict density as secondary priority queue key
    - Idea 4: Skip repair for non-cardinal conflicts; skip rebranch for cardinal arms
    """

    def __init__(self, plans, parameters, planners, sim_steps=1.0, pos_tol=0.02, ori_tol=0.1):
        super().__init__(plans, parameters, planners, sim_steps, pos_tol, ori_tol)
        # Per-(plan_indices_tuple, agent_A, agent_B) collision cache for multi-conflict detection
        self._pair_cache = {}

    def _pair_check(self, key, agent_A, plan_A, agent_B, plan_B):
        """Cached collision check keyed by (plan_indices_tuple, A, B)."""
        if key not in self._pair_cache:
            self._pair_cache[key] = self.check_collisions((agent_A, plan_A), (agent_B, plan_B))
        return self._pair_cache[key]

    def _classify_conflict(self, agent_A, agent_B, plan, plan_indices, constraints):
        """
        Scan unconstrained candidate plans to classify conflict cardinality.

        cardinal_A: True if all unconstrained candidates for agent_A still collide with
                    agent_B's current plan (rebranching A alone cannot resolve this conflict).
        cardinal_B: symmetric for agent_B.
        density: fraction of unconstrained candidates (across both arms) that have no
                 collision-free alternative — used as secondary sort key (Idea 3).
        """
        constrained_A = constraints.get(agent_A, set())
        constrained_B = constraints.get(agent_B, set())

        a_free = 0
        a_total = 0
        for j in range(len(self.plans[agent_A])):
            if j in constrained_A:
                continue
            a_total += 1
            collision, _, _ = self.check_collisions(
                (agent_A, self.plans[agent_A][j]), (agent_B, plan[agent_B])
            )
            if not collision:
                a_free += 1

        b_free = 0
        b_total = 0
        for j in range(len(self.plans[agent_B])):
            if j in constrained_B:
                continue
            b_total += 1
            collision, _, _ = self.check_collisions(
                (agent_A, plan[agent_A]), (agent_B, self.plans[agent_B][j])
            )
            if not collision:
                b_free += 1

        total = a_total + b_total
        density = ((a_total - a_free) + (b_total - b_free)) / total if total > 0 else 1.0
        cardinal_A = a_free == 0  # True when no free alternative exists (or no candidates)
        cardinal_B = b_free == 0
        return cardinal_A, cardinal_B, density

    def find_plans(self, agents_deque):
        start_time = time.time()
        earliest_collision_time = self.parameters["prediction_horizon"]
        root_plan_indices = [0] * self.num_agents
        root_plan = [self.plans[i][0] for i in range(self.num_agents)]
        root_cost = self.compute_cost(root_plan, root_plan_indices)

        root_id = self.node_id_counter
        self.node_id_counter += 1
        initial_node = CardinalCBSNode(
            root_plan_indices, {}, root_cost, node_id=root_id, parent_id=None
        )
        self._add_graph_node(initial_node, style="filled", fillcolor="lightblue", shape="box")
        base_plan = root_plan.copy()

        self.metrics["num_generated"] += 1
        heapq.heappush(self.open_set, initial_node)

        while self.open_set and time.time() - start_time < self.parameters["timeout"]:
            self.metrics["num_expanded"] += 1
            current_node = heapq.heappop(self.open_set)

            if self.graph.body[-(len(str(current_node.id)) + 13):] != "palegreen":
                self._add_graph_node(
                    current_node, style="filled", fillcolor="lightgrey", shape="box"
                )

            plan = [
                self.plans[i][current_node.plan_indices[i]] for i in range(self.num_agents)
            ]

            # Idea 1: Collect all pairwise conflicts; select earliest-timestep conflict
            all_conflicts = []
            indices_key = tuple(current_node.plan_indices)
            for a in range(self.num_agents):
                for b in range(a + 1, self.num_agents):
                    pair_key = (indices_key, a, b)
                    result = self._pair_check(pair_key, a, plan[a], b, plan[b])
                    if result[0]:
                        t = result[1]
                        if t != -1 and t < earliest_collision_time:
                            earliest_collision_time = t
                            base_plan = plan.copy()
                        # Use inf for t==-1 (collision detected but timestep unknown) so
                        # known-timestep conflicts are always preferred
                        sort_t = t if t != -1 else float("inf")
                        all_conflicts.append((sort_t, a, b, result[2]))

            if not all_conflicts:
                if self.num_agents == 1:
                    dummy_plan = np.zeros_like(plan[0])
                    collision, first_collision_step, _ = self.check_collisions(
                        (0, plan[0]), (0, dummy_plan)
                    )
                    earliest_collision_time = (
                        first_collision_step
                        if collision
                        else self.parameters["prediction_horizon"]
                    )
                self._add_graph_node(
                    current_node,
                    style="filled",
                    fillcolor="palegreen",
                    shape="doublecircle",
                )
                if self.node_id_counter > 1:
                    self.graph.render(
                        "cbs_search_tree", view=False, cleanup=True, format="svg"
                    )
                self.metrics["planning_time"] = time.time() - start_time
                return plan, earliest_collision_time

            # Pick conflict at earliest timestep
            chosen = min(all_conflicts, key=lambda c: c[0])
            collision_bodies = chosen[3]

            # Map pybullet body IDs back to agent indices
            agent_A = None
            agent_B = None
            for i in range(self.num_agents):
                if (
                    collision_bodies[0]
                    and self.single_agent_planners[i].pybullet_id == collision_bodies[0]
                ):
                    agent_A = i
                if (
                    collision_bodies[1]
                    and self.single_agent_planners[i].pybullet_id == collision_bodies[1]
                ):
                    agent_B = i
            if agent_A is None or agent_B is None:
                return base_plan, earliest_collision_time

            conflict = (agent_A, agent_B)

            updated_constraints = current_node.constraints.copy()
            updated_constraints.setdefault(agent_A, set()).add(
                current_node.plan_indices[agent_A]
            )
            updated_constraints.setdefault(agent_B, set()).add(
                current_node.plan_indices[agent_B]
            )

            # Idea 2: Classify cardinality by scanning existing B candidates
            cardinal_A, cardinal_B, density = self._classify_conflict(
                agent_A, agent_B, plan, list(current_node.plan_indices), updated_constraints
            )

            # Idea 4: Conditional dispatch
            # Non-cardinal arm → rebranch (its existing candidates have free alternatives)
            # Cardinal arm     → repair  (no existing candidate avoids the conflict)
            do_rebranch_A = not cardinal_A
            do_rebranch_B = not cardinal_B
            do_repair = cardinal_A or cardinal_B

            # Rebranch agent_A
            if do_rebranch_A:
                for A in range(self.parameters["num_samples"]):
                    if A not in updated_constraints[agent_A]:
                        new_indices = list(current_node.plan_indices)
                        new_indices[agent_A] = A
                        new_plan = [
                            self.plans[k][new_indices[k]] for k in range(self.num_agents)
                        ]
                        new_cost = self.compute_cost(new_plan, new_indices)
                        new_id = self.node_id_counter
                        self.node_id_counter += 1
                        new_node = CardinalCBSNode(
                            new_indices,
                            updated_constraints,
                            new_cost,
                            node_id=new_id,
                            parent_id=current_node.id,
                            conflict_density=density,
                        )
                        constraint_label = (
                            f"Agent {agent_A} != Plan {current_node.plan_indices[agent_A]}"
                        )
                        self._add_graph_node(new_node, shape="box")
                        self._add_graph_edge(current_node, new_node, constraint_label)
                        self.metrics["num_generated"] += 1
                        self.metrics["num_rebranch"] += 1
                        heapq.heappush(self.open_set, new_node)

            # Rebranch agent_B
            if do_rebranch_B:
                for B in range(self.parameters["num_samples"]):
                    if B not in updated_constraints[agent_B]:
                        new_indices = list(current_node.plan_indices)
                        new_indices[agent_B] = B
                        new_plan = [
                            self.plans[k][new_indices[k]] for k in range(self.num_agents)
                        ]
                        new_cost = self.compute_cost(new_plan, new_indices)
                        new_id = self.node_id_counter
                        self.node_id_counter += 1
                        new_node = CardinalCBSNode(
                            new_indices,
                            updated_constraints,
                            new_cost,
                            node_id=new_id,
                            parent_id=current_node.id,
                            conflict_density=density,
                        )
                        constraint_label = (
                            f"Agent {agent_B} != Plan {current_node.plan_indices[agent_B]}"
                        )
                        self._add_graph_node(new_node, shape="box")
                        self._add_graph_edge(current_node, new_node, constraint_label)
                        self.metrics["num_generated"] += 1
                        self.metrics["num_rebranch"] += 1
                        heapq.heappush(self.open_set, new_node)

            # Repair via dual model (only when at least one arm is cardinal)
            if do_repair:
                dual_plan_A = self.dual_agent_planner.predict_plan(conflict, agents_deque)
                if dual_plan_A is not None:
                    self.metrics["num_repair"] += 1
                    for A in range(self.parameters["num_samples"]):
                        new_plan_A = dual_plan_A[A]
                        collision, _, _ = self.check_collisions(
                            (agent_A, new_plan_A), (agent_B, plan[agent_B])
                        )
                        if not collision:
                            self.plans[agent_A] = np.concatenate(
                                [self.plans[agent_A], new_plan_A[None, ...]], axis=0
                            )
                            new_indices = list(current_node.plan_indices)
                            new_indices[agent_A] = len(self.plans[agent_A]) - 1
                            new_plan = [
                                self.plans[k][new_indices[k]] for k in range(self.num_agents)
                            ]
                            new_plan[agent_A] = new_plan_A
                            new_cost = self.compute_cost(new_plan, new_indices)
                            new_id = self.node_id_counter
                            self.node_id_counter += 1
                            new_node = CardinalCBSNode(
                                new_indices,
                                updated_constraints.copy(),
                                new_cost,
                                node_id=new_id,
                                parent_id=current_node.id,
                                conflict_density=density,
                            )
                            constraint_label = (
                                f"Agent {agent_A} = Dual Plan {new_indices[agent_A]}"
                            )
                            self._add_graph_node(new_node, shape="box")
                            self._add_graph_edge(current_node, new_node, constraint_label)
                            self.metrics["num_generated"] += 1
                            heapq.heappush(self.open_set, new_node)

                conflict_reversed = (conflict[1], conflict[0])
                dual_plan_B = self.dual_agent_planner.predict_plan(
                    conflict_reversed, agents_deque
                )
                if dual_plan_B is not None:
                    self.metrics["num_repair"] += 1
                    for B in range(self.parameters["num_samples"]):
                        new_plan_B = dual_plan_B[B]
                        collision, _, _ = self.check_collisions(
                            (agent_B, new_plan_B), (agent_A, plan[agent_A])
                        )
                        if not collision:
                            self.plans[agent_B] = np.concatenate(
                                [self.plans[agent_B], new_plan_B[None, ...]], axis=0
                            )
                            new_indices = list(current_node.plan_indices)
                            new_indices[agent_B] = len(self.plans[agent_B]) - 1
                            new_plan = [
                                self.plans[k][new_indices[k]] for k in range(self.num_agents)
                            ]
                            new_plan[agent_B] = new_plan_B
                            new_cost = self.compute_cost(new_plan, new_indices)
                            new_id = self.node_id_counter
                            self.node_id_counter += 1
                            new_node = CardinalCBSNode(
                                new_indices,
                                updated_constraints.copy(),
                                new_cost,
                                node_id=new_id,
                                parent_id=current_node.id,
                                conflict_density=density,
                            )
                            constraint_label = (
                                f"Agent {agent_B} = Dual Plan {new_indices[agent_B]}"
                            )
                            self._add_graph_node(new_node, shape="box")
                            self._add_graph_edge(current_node, new_node, constraint_label)
                            self.metrics["num_generated"] += 1
                            heapq.heappush(self.open_set, new_node)

        self.metrics["planning_time"] = time.time() - start_time
        return base_plan, earliest_collision_time
