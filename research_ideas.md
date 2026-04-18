# Plan: Cardinality-Aware Conflict Selection for FM-MAP / DG-MAP

## Context

The CBS planner in `core/planner/cbs.py` currently uses a **first-found, always-rebranch-then-repair** strategy:
- Conflict selection (lines 91–107): picks the first conflict found by iterating agent pairs — no prioritization
- Resolution order (lines 153–300): always does rebranch *and* repair for every conflict node
- Queue ordering (lines 18–19): min-heap on cost only — no heuristic estimates

In the MAPF literature, **cardinality** classifies conflicts by whether resolving them forces a cost increase. Applied here, a "cardinal" conflict is one where no existing candidate plan for an arm avoids the collision — so rebranching is futile and repair (expensive dual-model inference) is mandatory. Identifying this *before* committing to a resolution path saves wasted rebranch nodes and steers repair calls where they matter.

---

## Literature Background (what to cite / build on)

| Paper | Core idea | Relevance |
|-------|-----------|-----------|
| **ICBS** (Boyarski et al., IJCAI 2015) | Cardinal > semi-cardinal > non-cardinal classification via MDDs; 20× speedup over CBS | Direct foundation |
| **CBSH** (Felner et al., ICAPS 2018) | Admissible heuristics aggregating cardinal conflict costs; WDG dependency graph | High-level queue guidance |
| **f-Aware** (Boyarski et al., AAAI 2021) | f-cardinal = both child f-values exceed parent's; combines cost + heuristic | Stronger prioritization |
| **Symmetry Reasoning** (Li et al., AIJ 2022) | Breaking equivalent conflict resolutions; up to 10,000× node reduction | Relevant for structured workspaces |
| **EECBS** (Li & Ruml, AAAI 2021) | Bounded-suboptimal CBS; conflict prioritization under time budget | Relevant given 60s timeout |
| **Learning to Resolve Conflicts** (AAAI 2021) | RL policy for conflict prioritization | Motivation for learned cardinality score |

---

## Key Files

| File | Relevant lines |
|------|----------------|
| `core/planner/cbs.py` | 18–19 (queue order), 30–35 (metrics), 73–107 (conflict detection), 135–151 (agent ID mapping), 153–300 (rebranch + repair) |
| `core/planner/base_search.py` | 29–52 (cost), 54–103 (collision check + conflict tuple) |
| `core/planner/agent_planners.py` | `predict_plan()` — diffusion dual model |
| `core/planner/flow_agent_planners.py` | `predict_plan()` — flow dual model |

---

## Ideas (from cheapest to most involved)

### Idea 1 — Earliest-Timestep Conflict Priority (trivial)

**What**: Change conflict selection from first-found (by agent-pair iteration order) to earliest-collision-timestep. Among all detected conflicts, pick the one with the smallest `action_idx`.

**Why**: Resolving temporally-earlier conflicts prunes more of the downstream search tree. This is orthogonal to cardinality and costs nothing extra.

**Change**: Lines 91–107 — instead of `break` on first conflict, collect all conflicts, then select `min(conflicts, key=lambda c: c[1])` (index 1 = action_idx).

**Cost**: O(n²) collision checks per expand (same as now but no early break) — marginal overhead.

---

### Idea 2 — Approximate Cardinality via Candidate Scan (main idea)

**What**: Before choosing rebranch vs repair, scan agent_A's and agent_B's existing B=10 candidates against the conflict partner's current plan. Classify the conflict:

| Type | Condition | Action |
|------|-----------|--------|
| **Cardinal** | All plans of A conflict with B's current plan *and* all plans of B conflict with A's current plan | Skip rebranch entirely → go straight to repair |
| **Semi-cardinal (A)** | All plans of A conflict with B; B has free alternatives | Rebranch B only; repair A |
| **Semi-cardinal (B)** | All plans of B conflict with A; A has free alternatives | Rebranch A only; repair B |
| **Non-cardinal** | Both A and B have alternative plans | Rebranch both; skip repair |

**Why**: Currently repair is called unconditionally (lines 220, 262) even when one or both arms have valid alternatives among their existing B candidates. Skipping repair for non-cardinal conflicts saves dual-model inference calls — the most expensive operation. Skipping rebranch for cardinal conflicts avoids generating nodes that will immediately re-conflict.

**How to compute** (in cbs.py before line 153):
```python
# Check how many of agent_A's plans avoid agent_B's current plan
a_free = sum(
    1 for idx in range(len(self.plans[agent_A]))
    if idx not in updated_constraints[agent_A]
    and not self.check_collisions((agent_A, self.plans[agent_A][idx]),
                                   (agent_B, self.plans[agent_B][current_b_idx]))[0]
)
b_free = sum(...)  # symmetric for agent_B

is_cardinal_A = (a_free == 0)
is_cardinal_B = (b_free == 0)
```

**Cost**: O(B) extra collision checks per conflict resolution — cheap (B=10, collision checks are fast pybullet queries already cached).

---

### Idea 3 — Conflict Density Score for Queue Ordering

**What**: Augment each CBS node with a **conflict density** score = fraction of all (plan_A_idx, plan_B_idx) pairs that collide between the two conflicting arms. Use this as a secondary sort key in the open list (primary = cost, secondary = conflict density).

- High density ≈ cardinal → harder to resolve cheaply → expand later (or expand with repair strategy)
- Low density ≈ non-cardinal → easy rebranch → expand sooner

**Why**: Inspired by WDG (Weighted Dependency Graph) from CBSH. Gives the planner awareness of how "stuck" a conflict is without computing full MDDs.

**Change**:
- `CBSNode` gains a `conflict_density` field (float, default 0.0)
- `__lt__` tie-breaks on `conflict_density` (ascending — prefer easier conflicts first)
- Compute density when creating child nodes: count colliding pairs / B²

---

### Idea 4 — Conditional Repair: Only Repair Cardinal Conflicts

**What**: Combine ideas 1–3 into a single policy:
- Non-cardinal conflicts → rebranch only (skip lines 220–300)
- Semi-cardinal conflicts → rebranch the free agent + repair the constrained agent
- Cardinal conflicts → repair only (skip rebranch lines 153–218), or repair first then rebranch if repair fails

This is the direct analog of ICBS prioritization in the discrete setting, adapted to the B-candidate + dual-model setting.

**Expected impact**:
- Fewer repair calls on easy instances → lower `total_cbs_repair` and `avg_planning_time`
- Deeper search on hard instances (cardinal conflicts get direct repair) → higher `total_cbs_expanded`
- Net effect: same or better success rate, lower latency on easy tasks

---

### Idea 5 — Learned Cardinality Score (longer-term)

**What**: Train a small MLP classifier on top of the observation vectors:
- Input: obs_A ⊕ obs_B (dim = 114) at the collision timestep
- Output: p(cardinal) ∈ [0,1]
- Training signal: ground truth from offline CBS rollouts (was repair required to resolve?)

**Why**: Avoids O(B) collision checks at inference; generalizes to novel arm configurations. Inspired by "Learning to Resolve Conflicts for MAPF" (AAAI 2021).

**Cost**: One forward pass per conflict detection; training requires labeled CBS rollout data.

**Recommendation**: Implement ideas 1–4 first; use the new `total_cbs_repair` / `total_cbs_rebranch` metrics (already implemented) to measure quality. If data shows a clear pattern in when repair is needed, train the classifier.

---

---

## Separate Pipeline Design (no overwriting of existing code)

### Philosophy
The existing pipeline is **untouched**. All new code lives in one new file. Two small additions to existing files dispatch to the new class via a `--cbs_strategy` flag.

### Files changed / created

| File | Change | Type |
|------|--------|------|
| `core/planner/cardinal_cbs.py` | **New file** — `CardinalCBS` subclass of `ConflictBasedSearch` | New |
| `application/demo.py` | Add `cbs_strategy: str = "standard"` to `Parameters` dataclass (line 61) + `--cbs_strategy` CLI arg | +2 lines |
| `application/executer.py` | Change line 122 dispatch: if `parameters["cbs_strategy"] == "cardinal"` import and use `CardinalCBS`, else use existing `ConflictBasedSearch` | +5 lines |

### `core/planner/cardinal_cbs.py` structure

```python
from core.planner.cbs import ConflictBasedSearch, CBSNode
import heapq

class CardinalCBS(ConflictBasedSearch):
    """CBS with cardinal conflict prioritization (ideas 1–4)."""

    def find_plans(self, agents_deque):
        # Same outer loop structure as parent; overrides _select_conflict
        # and _resolve_conflict internally
        ...

    def _collect_all_conflicts(self, plan):
        """Idea 1: scan all agent pairs, return list of (action_idx, agent_A, agent_B)."""
        ...

    def _select_conflict(self, plan):
        """Idea 1: pick conflict with smallest action_idx (earliest in time)."""
        conflicts = self._collect_all_conflicts(plan)
        if not conflicts:
            return None
        return min(conflicts, key=lambda c: c[0])  # c[0] = action_idx

    def _classify_conflict(self, agent_A, agent_B, plan, constraints):
        """Idea 2: scan existing B candidates to compute cardinality.
        Returns (cardinal_A: bool, cardinal_B: bool, density: float).
        cardinal_A = True means all of A's candidates conflict with B's current plan.
        density = fraction of (A_plan, B_plan) pairs that mutually collide.
        """
        b_current_idx = plan indices for agent_B ...
        a_free = count of agent_A candidates not in constraints[A] that don't collide with B's plan
        b_free = count of agent_B candidates not in constraints[B] that don't collide with A's plan
        density = colliding_pairs / total_pairs
        return (a_free == 0), (b_free == 0), density

    def _resolve_conflict(self, node, conflict, agents_deque):
        """Idea 4: dispatch rebranch/repair based on cardinality classification."""
        agent_A, agent_B = conflict
        cardinal_A, cardinal_B, density = self._classify_conflict(...)

        child_nodes = []
        if not cardinal_A:
            child_nodes += self._rebranch(node, agent_A, ...)  # Idea 3: attach density to node
        if not cardinal_B:
            child_nodes += self._rebranch(node, agent_B, ...)
        if cardinal_A or cardinal_B:
            child_nodes += self._repair(node, conflict, agents_deque, ...)
        # If both non-cardinal: skip repair entirely
        return child_nodes
```

### `CBSNode` augmentation (inside `cardinal_cbs.py`)

Override or extend `CBSNode.__lt__` to add secondary sort on `conflict_density` (Idea 3):
```python
@dataclass
class CardinalCBSNode(CBSNode):
    conflict_density: float = 0.0
    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.conflict_density < other.conflict_density  # prefer easier conflicts
```

### New CLI usage (no change to existing commands)

```bash
# Standard pipeline (unchanged)
python application/demo.py \
  --single_agent_model ... --dual_agent_model ... \
  --backbone diffusion --num_samples 10 --n_timesteps 100 --num_experiments 100

# Cardinal CBS pipeline (new)
python application/demo.py \
  --single_agent_model ... --dual_agent_model ... \
  --backbone diffusion --num_samples 10 --n_timesteps 100 --num_experiments 100 \
  --cbs_strategy cardinal
```

Works with both `--backbone diffusion` and `--backbone flow` — the planner class is orthogonal to the backbone.

---

## Implementation Order

1. `core/planner/cardinal_cbs.py` — full `CardinalCBS` class (Ideas 1–4)
2. `application/demo.py` — add `cbs_strategy` field + CLI arg (~2 lines)
3. `application/executer.py` — add dispatch at line 122 (~5 lines)

---

## Verification

```bash
cd /usr1/kj_codebase/ipl_project/diff-mapf_fM && export PYTHONPATH=$(pwd):$PYTHONPATH

# Baseline (standard CBS)
python application/demo.py \
  --single_agent_model application/runs/plain_diffusion/mini_custom_diffusion_1.pth \
  --dual_agent_model   application/runs/plain_diffusion/mini_custom_diffusion_2.pth \
  --backbone diffusion --num_samples 10 --n_timesteps 100 --num_experiments 100 \
  --cbs_strategy standard 2>&1 | tee runs/logs/benchmark_standard_cbs.log

# Cardinal CBS
py