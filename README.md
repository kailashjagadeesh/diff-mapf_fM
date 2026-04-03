# Diffusion-Guided Multi-Arm Motion Planning



**Project:** [diff-mapf-mers.csail.mit.edu](https://diff-mapf-mers.csail.mit.edu/) • **Paper:** [arXiv:2509.08160](https://arxiv.org/abs/2509.08160)

![Overview](docs/static/images/overview.png)

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Pre-Trained Models & Data](#pre-trained-models-and-data)
- [Evaluate](#evaluate)
- [Summarize Results](#summarize-results)
- [Render Videos](#render-videos)
- [Train From Scratch](#train-from-scratch)
- [Flow Matching Pipeline](#flow-matching-pipeline)
- [Repository Layout](#repository-layout)
- [Parameter Reference](#parameter-reference)
- [Citation](#citation)
- [Credits](#credits)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository implements **DG-MAP**, a closed-loop multi-arm motion planner combining specialized generative policies with a MAPF-inspired CBS search framework.

Two policy backbones are supported and can be benchmarked side-by-side:

| Backbone | Model | Inference steps | Entry point |
|----------|-------|-----------------|-------------|
| **Diffusion** (DDPM) | `ConditionalUnet1D` + `DDPMScheduler` | 100 denoising steps | `core/agent_manager.py` |
| **Flow Matching** | `ConditionalUnet1D` + Euler ODE | 10 ODE steps (~10× faster) | `core/flow_agent_manager.py` |

Both backbones train two specialized models:
- **Single-arm model** — generates individual arm trajectory proposals
- **Dual-arm model** — resolves pairwise collisions between conflicting arms

The CBS planner is backbone-agnostic; switching is controlled by a single `--backbone` flag at eval time.

---

## Requirements
- **Python:** 3.9+ (managed via Conda)
- **Dependencies:** Specified in `environment.yml`
- **GPU:** Recommended (RTX 3090 / 4090 or better for training; inference runs on CPU too)
- **OS:** Linux (tested on Ubuntu 22.04.5 LTS and 24.04.3 LTS)

---

## Setup
```bash
conda env create -f environment.yml
conda activate multiarm
# Set repo root so imports work from any working directory:
export PYTHONPATH=<path-to-diff-mapf>:$PYTHONPATH
```

---

## Pre-Trained Models and Data

### Option 1: Fetch Everything Automatically
```bash
./fetch_assets.sh all --outdir .
```
- Add `--list` to preview the download plan without saving.
- Swap `all` for `models`, `datasets`, or `benchmarks` to grab a single category.

### Option 2: Download Assets Manually

**Pre-trained diffusion planners**
- [Plain Diffusion models](https://www.dropbox.com/scl/fo/daah2lixb3digjcp5i7ti/AIJ3po1nKmUSPKTBJvJn5qs?rlkey=wsq4b705kx8qi0sxq5j9ipzmz&st=pay2cq62&dl=0) → extract into `application/runs/plain_diffusion/`
- [Diffusion-QL models](https://www.dropbox.com/scl/fo/tmrnoz8sticj660oj5v2i/APZdOkl6FX-gXjLSeSxVG9s?rlkey=488l482qmr5kq4y5776j1k2qz&st=xkvfv1nq&dl=0) → extract into `application/runs/diffusion_ql/`

**Expert datasets** (extract into `datasets/`)
- [Single-Agent](https://www.dropbox.com/scl/fi/e2mnzqsrh9wf96bhbthb7/single_agent.zip?rlkey=a8uf0gukb04te46zu4164o196&st=hpa9rwh0&dl=0)
- [Dual-Agent](https://www.dropbox.com/scl/fi/w9o3c05ndyeavbiu3r1vu/dual_agent.zip?rlkey=u0bqnedvzlvwta8xfpzih382k&st=lzg49q8u&dl=0)
- [Single-QL-Agent](https://www.dropbox.com/scl/fi/i71hnenw21oxc9uqpbcy7/ql_single_agent.zip?rlkey=uxnewd77fl9tsfpj1o9sbh1ed&st=wonrl3nj&dl=0)
- [Dual-QL-Agent](https://www.dropbox.com/scl/fi/l3601jzsouu1pl3c6lv5w/ql_dual_agent.zip?rlkey=tgra17py4fh2qthj5m7tpcedh&st=qqq1jwex&dl=0)

**Benchmark tasks (Ha et al.)**
```bash
wget -qO- https://multiarm.cs.columbia.edu/downloads/data/benchmark.tar.xz | tar xvfJ -
mv benchmark application/tasks/
```

---

## Evaluate

Both backbones use the same `application/demo.py` entry point. Select the backbone with `--backbone`.

### Diffusion backbone (default)
```bash
python application/demo.py \
    --single_agent_model "runs/plain_diffusion/mini_custom_diffusion_1.pth" \
    --dual_agent_model   "runs/plain_diffusion/mini_custom_diffusion_2.pth" \
    --backbone diffusion \
    --num_samples 10 \
    --n_timesteps 100
```

### Flow matching backbone
```bash
python application/demo.py \
    --single_agent_model "runs/<flow-single-run>/ckpt_single_agent_model_00100.pth" \
    --dual_agent_model   "runs/<flow-dual-run>/ckpt_dual_agent_model_00100.pth" \
    --backbone flow \
    --num_samples 10 \
    --n_steps 10
```

> **Note:** Each `.pth` checkpoint must have a matching `.npz` normalization stats file at the same path (e.g. `ckpt_single_agent_model_00100.npz`). These are saved automatically alongside every checkpoint during flow matching training.

### Key eval flags

| Flag | Default | Recommended range | Notes |
|------|---------|-------------------|-------|
| `--backbone` | `diffusion` | `diffusion` \| `flow` | Selects planner classes |
| `--num_samples` | `10` | `10 – 50` | More samples = better CBS search quality, slower per step |
| `--n_timesteps` | `100` | `50 – 100` | Diffusion only. Lower = faster but noisier trajectories |
| `--n_steps` | `10` | `5 – 20` | Flow matching only. 10 is the recommended balance |
| `--timeout` | `60` | `60 – 600` | Per-experiment wall-clock budget (seconds) |
| `--observation_dim` | `57` | fixed at `57` | Must match training config |
| `--prediction_horizon` | `16` | fixed at `16` | Must match training config |
| `--observation_horizon` | `2` | fixed at `2` | Must match training config |

---

## Summarize Results
```bash
python application/evaluate_results.py --result_dir <result-dir>
```
Works identically for diffusion and flow matching runs — the results CSV format is the same for both.

---

## Render Videos

Simulations are recorded as `.pkl` files by `PybulletRecorder` during every eval run. Use `application/render_video.py` to convert them to `.mp4` — no GUI needed, works fully headless.

### Convert a full run
```bash
python application/render_video.py \
    --pkl_dir runs/<backbone>/<timestamp>/simulation_<timestamp>
```
This finds all `.pkl` files recursively under the directory and writes a matching `.mp4` next to each one.

### Convert a single experiment
```bash
python application/render_video.py \
    --pkl runs/<backbone>/<timestamp>/simulation_<timestamp>/simulation_42.pkl
```

### Custom output path
```bash
python application/render_video.py \
    --pkl runs/.../simulation_42.pkl \
    --output videos/experiment_42.mp4
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--pkl` | — | Path to a single `.pkl` file (mutually exclusive with `--pkl_dir`) |
| `--pkl_dir` | — | Directory to search recursively for `.pkl` files |
| `--output` | same path as `.pkl` with `.mp4` extension | Output path (single-file mode only) |
| `--width` | `1280` | Video width in pixels |
| `--height` | `720` | Video height in pixels |
| `--fps` | `30` | Frames per second |
| `--egl` | off | Use GPU EGL renderer instead of CPU software renderer |
| `--yaw` | `46.39` | Camera yaw in degrees |
| `--pitch` | `-55.0` | Camera pitch in degrees |
| `--dist` | `1.9` | Camera distance from scene centre |

### CPU vs GPU rendering

By default the script uses `ER_TINY_RENDERER` (PyBullet's built-in CPU software renderer) — no extra dependencies needed. On Linux with EGL drivers (e.g. an RTX 4090), pass `--egl` to use GPU-accelerated rendering which is significantly faster for high-resolution or long rollouts:

```bash
python application/render_video.py --pkl_dir runs/.../simulation_<timestamp> \
    --egl --width 1920 --height 1080 --fps 60
```

### Typical workflow
```bash
# 1. Run evaluation headlessly (saves .pkl per experiment automatically)
python application/demo.py --backbone flow \
    --single_agent_model runs/flow_single/.../ckpt_single_agent_model_00100 \
    --dual_agent_model   runs/flow_dual/.../ckpt_dual_agent_model_00100 \
    --num_experiments 100

# 2. Convert all saved simulations to video
python application/render_video.py \
    --pkl_dir runs/flow_dual/<timestamp>/simulation_<timestamp>
```

---

## Train From Scratch

### Download Expert Datasets
Use the dataset links in [Pre-Trained Models and Data](#pre-trained-models-and-data). Extract each archive into `datasets/` so the training scripts can find the zarr files.

```
datasets/
  single_agent/    ← used with --num_agents 1
  dual_agent/      ← used with --num_agents 2
```

### Diffusion backbone

```bash
mkdir -p runs

# Single-arm model
python -u core/agent_manager.py \
    --config configs/diffusion.json \
    --offline_dataset datasets/single_agent \
    --num_agents 1 \
    --num_epochs 300 \
    --name diffusion_single_agent

# Dual-arm model
python -u core/agent_manager.py \
    --config configs/diffusion.json \
    --offline_dataset datasets/dual_agent \
    --num_agents 2 \
    --num_epochs 300 \
    --name diffusion_dual_agent
```

To resume from a checkpoint:
```bash
--load <path-to-checkpoint>
```

### Diffusion-QL backbone
```bash
python -u core/agent_manager.py \
    --config configs/diffusionQL.json \
    --offline_dataset datasets/ql_single_agent \
    --num_agents 1 \
    --num_epochs 300 \
    --name diffusionQL_single_agent
```

### Key training flags (diffusion)

| Flag | Default | Recommended range | Notes |
|------|---------|-------------------|-------|
| `--num_epochs` | `100` | `100 – 500` | 300 is a good default; use `--early_stop` to auto-terminate |
| `--grad_norm` | `0.0` (off) | `0.0` or `1.0` | Enable gradient clipping (`1.0`) for unstable runs |
| `--early_stop` | off | — | Stops training when BC loss diverges |
| `batch_size` *(in config)* | `4096` | `2048 – 4096` | Reduce to `2048` if running two jobs in parallel on one GPU |
| `pi_lr` *(in config)* | `0.0001` | `0.00005 – 0.0003` | Standard AdamW LR; cosine decay is applied automatically |
| `n_timesteps` *(in config)* | `100` | `50 – 100` | DDPM steps; 100 gives best quality |
| `unet_layers` *(in config)* | `[256,512,1024]` | `[256,512,1024]` | Larger = more capacity but slower training |

---

## Flow Matching Pipeline

The flow matching pipeline is a drop-in replacement for the diffusion backbone using **Rectified Flow** (linear Conditional Flow Matching). Key advantages:

- **10× faster inference** — 10 Euler ODE steps vs. 100 DDPM denoising steps
- **Same datasets** — trains on the identical `single_agent` / `dual_agent` zarr datasets
- **Same architecture** — reuses `ConditionalUnet1D`; the network now predicts a velocity field instead of noise

### Training (flow matching)

```bash
mkdir -p runs

# Single-arm flow model
python -u core/flow_agent_manager.py \
    --config configs/flow.json \
    --offline_dataset datasets/single_agent \
    --num_agents 1 \
    --num_epochs 300 \
    --name flow_single_agent

# Dual-arm flow model
python -u core/flow_agent_manager.py \
    --config configs/flow.json \
    --offline_dataset datasets/dual_agent \
    --num_agents 2 \
    --num_epochs 300 \
    --name flow_dual_agent
```

### Key training flags (flow matching)

| Flag | Default | Recommended range | Notes |
|------|---------|-------------------|-------|
| `--num_epochs` | `100` | `100 – 500` | Flow matching typically converges faster than diffusion; 200 is often sufficient |
| `--grad_norm` | `0.0` (off) | `0.0` or `1.0` | Same as diffusion |
| `--early_stop` | off | — | Same as diffusion |
| `batch_size` *(in config)* | `4096` | `2048 – 4096` | Same guidance as diffusion |
| `pi_lr` *(in config)* | `0.0001` | `0.00005 – 0.0003` | Same as diffusion |
| `n_steps` *(in config)* | `10` | `5 – 20` | ODE steps at inference. 10 is the recommended default |
| `unet_layers` *(in config)* | `[256,512,1024]` | `[256,512,1024]` | Identical to diffusion config |

### Checkpoint format
Flow matching checkpoints save a model weights file **and** a matching `.npz` normalization stats file alongside it:
```
runs/flow_single_agent/<timestamp>/
  ckpt_single_agent_model_00001       ← model weights (no extension)
  ckpt_single_agent_model_00001.npz   ← normalization stats (auto-saved)
  ...
```
Pass the weights path directly — no renaming needed. The planner derives the stats path automatically using `os.path.splitext`, which handles paths with or without a `.pth` extension.

### Eval with flow matching backbone
```bash
python application/demo.py \
    --single_agent_model "runs/flow_single_agent/<timestamp>/ckpt_single_agent_model_00100" \
    --dual_agent_model   "runs/flow_dual_agent/<timestamp>/ckpt_dual_agent_model_00100" \
    --backbone flow \
    --n_steps 10 \
    --num_samples 10
```

---

## Repository Layout
```text
application/
  demo.py                        # main evaluation entrypoint (--backbone flag)
  evaluate_results.py            # result aggregation / reporting
  executer.py                    # runtime planner dispatch (diffusion or flow)
configs/
  diffusion.json                 # diffusion training config
  diffusionQL.json               # diffusion-QL training config
  flow.json                      # flow matching training config
core/
  agent_manager.py               # diffusion training driver
  flow_agent_manager.py          # flow matching training driver
  utils.py                       # common CLI/config helpers (diffusion)
  flow_utils.py                  # common CLI/config helpers (flow)
  models/
    diffusionNet.py              # ConditionalUnet1D, DiffusionActor, DiffusionCritic
    diffusion.py                 # DiffusionLearner (BC training)
    diffusionQL.py               # DiffusionQLLearner (BC + Q-learning)
    flowNet.py                   # FlowActor (velocity field, Euler ODE)
    flow.py                      # FlowLearner (BC training, flow matching)
  planner/
    cbs.py                       # Conflict-Based Search
    agent_planners.py            # Agent, ResolveDualConflict (diffusion)
    flow_agent_planners.py       # FlowAgent, FlowResolveDualConflict (flow)
datasets/
  single_agent/                  # zarr — single-arm expert demos
  dual_agent/                    # zarr — dual-arm expert demos
  ql_single_agent/               # zarr — single-arm demos with rewards
  ql_dual_agent/                 # zarr — dual-arm demos with rewards
```

---

## Parameter Reference

### Training parameters (both backbones)

| Parameter | Config key | Default | Optimal range | Description |
|-----------|-----------|---------|---------------|-------------|
| Learning rate | `pi_lr` | `0.0001` | `5e-5 – 3e-4` | AdamW LR with cosine decay |
| Batch size | `batch_size` | `4096` | `2048 – 4096` | Larger batches more stable; reduce if OOM |
| Epochs | `--num_epochs` | `100` | `100 – 500` | 300 recommended; `--early_stop` for auto |
| Prediction horizon | `prediction_horizon` | `16` | fixed `16` | Action sequence length predicted per step |
| Observation horizon | `observation_horizon` | `2` | fixed `2` | Number of past observations used as context |
| Action horizon | `action_horizon` | `1` | `1 – 4` | Steps executed before replanning |
| UNet layers | `unet_layers` | `[256,512,1024]` | fixed | Channel widths for each UNet level |
| Time embedding dim | `time_dim` | `256` | `128 – 256` | Sinusoidal embedding size for timestep/flow-t |

### Diffusion-specific parameters

| Parameter | Config key | Default | Optimal range | Description |
|-----------|-----------|---------|---------------|-------------|
| DDPM steps | `n_timesteps` | `100` | `50 – 100` | Denoising chain length; fewer = faster but noisier |
| Beta schedule | `beta_schedule` | `squaredcos_cap_v2` | — | Noise schedule type; `squaredcos_cap_v2` recommended |

### Flow matching-specific parameters

| Parameter | Config key | Default | Optimal range | Description |
|-----------|-----------|---------|---------------|-------------|
| ODE steps | `n_steps` | `10` | `5 – 20` | Euler integration steps; 10 is the recommended default |

> **Tuning `n_steps`:** Values below 5 produce visibly coarser trajectories. Values above 15 yield diminishing returns. For benchmarking, use the same `n_steps` across all experiments.

### Eval parameters

| Flag | Default | Optimal range | Description |
|------|---------|---------------|-------------|
| `--num_samples` | `10` | `10 – 50` | Candidate action sequences per arm per planning step |
| `--timeout` | `60` | `60 – 600` | Per-experiment wall clock budget in seconds |
| `--n_timesteps` | `100` | `50 – 100` | Diffusion denoising steps at inference |
| `--n_steps` | `10` | `5 – 20` | Flow ODE steps at inference |
| `--backbone` | `diffusion` | `diffusion` \| `flow` | Policy backbone to use |

> **Tuning `--num_samples`:** The CBS planner treats each generated trajectory as a candidate. More samples increases the probability of finding a collision-free plan at the cost of compute. `10` works well for 3–4 arms; increase to `20–50` for 6–8 arms or hard tasks.

---

## Citation
If you use our work or codebase in your research, please cite our paper.
```bibtex
@InProceedings{pmlr-v305-parimi25a,
  title = {Diffusion-Guided Multi-Arm Motion Planning},
  author = {Parimi, Viraj and Williams, Brian C.},
  booktitle = {Proceedings of The 9th Conference on Robot Learning},
  pages = {4684--4696},
  year = {2025},
  editor = {Lim, Joseph and Song, Shuran and Park, Hae-Won},
  volume = {305},
  series = {Proceedings of Machine Learning Research},
  month = {27--30 Sep},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v305/main/assets/parimi25a/parimi25a.pdf},
  url = {https://proceedings.mlr.press/v305/parimi25a.html},
}
```

---

## Credits
Portions of code and datasets are adapted from:
- **Decentralized MultiArm**: <https://github.com/real-stanford/decentralized-multiarm>
- **PyBullet–Blender Recorder** (visualization): <https://github.com/huy-ha/pybullet-blender-recorder>

---

## License
This project is licensed under the [Apache License 2.0](LICENSE); portions of the code are adapted from credited works released under the same license.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` after install | Check `PYTHONPATH` is set to repo root |
| OOM during training | Reduce `batch_size` to `2048` in the config JSON |
| Running two training jobs in parallel OOM | Use `batch_size: 2048` for each job on a 24 GB GPU |
| `FileNotFoundError` on `.npz` at eval | Checkpoint `.pth` must have a matching `.npz` at the same path — flow training saves both automatically; for diffusion you may need to generate stats manually |
| No runs appearing in `runs/` | Confirm write permissions; `mkdir -p runs` if missing |
| Benchmark tasks missing | Re-run the `wget \| tar` commands; verify `application/tasks/benchmark/` exists |
| Flow loss not decreasing | Try lowering `pi_lr` to `5e-5`; ensure `batch_size ≥ 1024` |
| Diffusion trajectories jerky at eval | Increase `--n_timesteps` to `100`; decrease `--num_samples` if timeout is hit |
