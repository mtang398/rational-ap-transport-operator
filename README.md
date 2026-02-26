# Discretization-Agnostic Neural Operators for Radiative/Particle Transport

A PyTorch research repository implementing baseline neural operator models for
radiative and particle transport. The core goal is to learn the **solution operator**

> **T : (σ_a, σ_s, q, BC, ε, g) → I(x, ω)**

where the model is queried at arbitrary spatial points **x** and angular directions
**ω** at evaluation time — no re-training needed when the discretization changes.

---

## Research Goal

The central goal is to develop and rigorously evaluate an **asymptotic-preserving,
discretization-agnostic neural operator for radiative/particle transport** that maps

> **(σ_a, σ_s, q, BCs, ε, anisotropy) → I(x, ω)** (and derived moments φ, J, QoIs)

and to test whether **trainable rational activation functions** materially improve
generalization across three axes relative to standard activations (ReLU/GELU/SiLU):

1. **SN order transfer** — trained on S8, evaluated on S4 / S16 / S32 / S64
2. **Spatial resolution transfer** — trained on 17×17, evaluated on 34×34 / 68×68
3. **Cross-regime (ε sweep)** — trained across ε ∈ [0.01, 1], evaluated at ε → 0
   (diffusion limit) and ε = 1 (transport limit)

All three axes are tested on **C5G7 MOX, C5G7-TD, Kobayashi 3D void**, and
**Pinte et al. 2009** benchmarks.

The AP Micro-Macro model explicitly decomposes the intensity as
`I = I_P1(φ, J) + R`, which by construction recovers the correct diffusion
limit as ε → 0. Rational activations — learnable piecewise-rational functions
that adapt their shape during training — are the candidate nonlinearity being
tested across all three model families (FNO, DeepONet, AP Micro-Macro).

---

## What is Real, What is Approximate, and What is Synthetic

### Summary

| Component | Status | Details |
|---|---|---|
| Cross sections | ✅ **Real** | Digit-for-digit from published benchmark reports |
| Geometry / material layout | ✅ **Real** | Exact pin lattices, duct shapes, disk profiles |
| Kinetics parameters (C5G7-TD) | ✅ **Real** | Published β_i, λ_i from NEA/NSC/DOC(2016)7 |
| Flux / intensity targets | ⚠️ **Approximate** | Physics-based but not reference SN/MC solutions |
| Training datasets | ⚠️ **Synthetically generated** | Produced by `generate_dataset.py` on your machine |
| Reference benchmark data | ❌ **Not included** | Must be obtained from OECD/NEA or run a solver |

### The datasets are synthetically generated — not from official codes

When you run `generate_dataset.py` or `run_all.py`, the training data is
**computed on your machine** using internal approximations — no external
solver is invoked and no official benchmark data is downloaded. The datasets
in `runs/datasets/` are synthetic in the sense that:

- They are not the official OECD/NEA reference solutions
- They are not output from a validated SN or Monte Carlo code
- Each call to `generate_dataset.py` produces a fresh set of samples with
  ±3% random XS perturbations for diversity

This is intentional for the current development stage: the pipeline runs
end-to-end without any external dependencies, and the physically-consistent
approximations are sufficient to test model architecture and training
stability. For publication-quality results, replace with solver-generated
targets (see [Getting Real Flux Targets](#getting-real-flux-targets)).

### What is real in the inputs

| Benchmark | What is real |
|---|---|
| **C5G7** | Published 7-group XS for all 6 materials (UO2, MOX4.3/7.0/8.7, guide tube, fission chamber, moderator); full 51×51 quarter-core geometry (3×3 assemblies × 17×17 pin lattice). Source: NEA/NSC/DOC(2003)16, Table 2. |
| **C5G7-TD** | Same XS as C5G7 + published 6-group delayed-neutron parameters (β_total=0.006502, λ_i) from NEA/NSC/DOC(2016)7; rod-ejection transient solved with RK4 point kinetics |
| **Kobayashi** | Exact published geometry (L-shaped duct, dogleg, dog-ear) and σ_t values (source=0.1, void=1e-8, absorber=10.0 cm⁻¹). Source: NSC-DOC(2000)4. |
| **Pinte 2009** | Published disk density law (Σ ∝ r⁻¹, flared H ∝ r^1.25), stellar parameters (T_eff=9500 K, L=47 L_sun), dust opacity from Mie theory (κ_abs=2.3, κ_sca=10.4 cm²/g, g=0.60). Source: Pinte et al. 2009, Table 1. |

### What is approximate in the targets

| Benchmark | Current flux/intensity target | Accuracy | Real target requires |
|---|---|---|---|
| **C5G7** | Multigroup diffusion: −D∇²φ + σ_a φ = S | Good in optically thick regions; wrong in voids | OpenMC MC or OpenSn SN |
| **C5G7-TD** | Point kinetics α(t) × steady-state shape φ_ss(x) | Good for slow transients; wrong when spatial shape changes | Time-dependent SN solver |
| **Kobayashi** | First-flight transport kernel | Exact in void; misses scatter in source region | Full SN or MC |
| **Pinte 2009** | 5-step Λ-iteration radiative transfer | Accurate for τ < few; underestimates thick-disk scattering | MCFOST or RADMC-3D |

**The OpenMC model for C5G7 is fully implemented** (`src/solvers/openmc_c5g7_model.py`)
and produces real Monte Carlo flux once the `openmc` binary is installed via conda.
See [Pending Work](#pending-work) below.

---

## Pending Work

> **These are the remaining steps before this repository can produce
> publication-quality results.** Tasks 1–5 replace the current synthetic
> approximations with data from official OECD/NEA benchmark documents and
> validated solvers. Task 6 implements the rational activation experiments
> that are the primary research contribution.

### Task 1 — Download the official OECD/NEA benchmark documents

These PDFs contain the authoritative cross sections, geometry tables, and
**reference flux solutions** (the numbers your model should eventually match).
All are free to download from the OECD Nuclear Energy Agency:

| Benchmark | Document ID | Direct download URL |
|---|---|---|
| **C5G7** | NEA/NSC/DOC(2003)16 | https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nsc-doc2003-16.pdf |
| **C5G7-TD** | NEA/NSC/DOC(2016)7 | https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/nsc-doc2016-7.pdf |
| **Kobayashi** | NSC/DOC(2000)4 | https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/nsc-doc2000-4.pdf |

Download all three PDFs and keep them for reference. The cross sections already
hardcoded in this repository were taken from these documents. What is still
missing is the **reference flux distribution** printed in the appendices —
that is the ground truth your trained model should converge to.

For **Pinte 2009**, the reference intensity maps are attached as FITS files to
the original paper:
- **Paper**: Pinte et al. 2009, A&A 498, 967–980 — https://doi.org/10.1051/0004-6361/200811474
- **Benchmark data** (FITS intensity maps): contact the authors or download
  from the journal supplementary material page.

### Task 2 — Replace C5G7 flux targets with OpenMC Monte Carlo output

The code for this is **already written** — you just need the `openmc` binary.

```bash
# Step 1 — Install OpenMC with its compiled binary via conda
#           (pip alone installs only the Python API, not the transport solver)
conda create -n openmc_env python=3.11
conda activate openmc_env
conda install -c conda-forge openmc

# Step 2 — Also install Python deps in this env
pip install torch numpy h5py tqdm scipy einops tensorboard

# Step 3 — Smoke test: runs a small C5G7 eigenvalue calculation
cd "C:\Users\Maosen\2026 Neurips"
python scripts/run_openmc_c5g7.py \
    --n_particles 5000 --n_batches 20 --n_inactive 5 --n_samples 5

# Verify: k-effective printed to console should be ≈ 1.185
# Published OECD reference: k_eff = 1.18655 ± 0.00033

# Step 4 — Full production run (generate real MC training data)
python scripts/run_openmc_c5g7.py \
    --splits train val test \
    --n_samples 200 50 50 \
    --n_particles 500000 --n_batches 500 --n_inactive 100
```

Output: `runs/datasets/c5g7_train.zarr.h5` etc. with real Monte Carlo scalar
flux. The rest of the pipeline (`train.py`, `eval.py`, `run_all.py`) is
unchanged — just re-run from Step 2 of Quick Start, skipping `generate_dataset.py`
for `c5g7`.

### Task 3 — Replace Kobayashi flux targets with a reference SN solver

**OpenSn** (formerly OpenSN / Delphi) is a free, open-source SN solver:
- Homepage: https://github.com/Open-Sn/opensn
- Runs natively on **Linux/macOS**; on Windows use **WSL2** (Windows Subsystem
  for Linux).

```bash
# WSL2 install (Windows) — one-time setup
wsl --install          # then restart, set up Ubuntu in WSL2

# Inside WSL2
sudo apt update && sudo apt install -y cmake python3-dev python3-pip git
git clone https://github.com/Open-Sn/opensn.git
cd opensn && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
# Follow opensn docs for Python bindings
```

The interface stub is in `src/solvers/opensn_interface.py` — it already knows
how to call OpenSn and parse its flux output. Once OpenSn is installed, set the
`solver` field in `configs/benchmarks/kobayashi.yaml` to `opensn` and
re-run `generate_dataset.py`.

### Task 4 — Replace Pinte 2009 intensity with MCFOST or RADMC-3D output

- **MCFOST**: https://mcfost.readthedocs.io — Monte Carlo radiative transfer for
  protoplanetary disks. The exact disk model from Pinte et al. 2009 was run with
  MCFOST; config files are in the paper's supplementary material.
- **RADMC-3D**: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/ —
  alternative RT solver, widely used.

After running either solver, convert its output using
`scripts/convert_pinte2009.py` (adapt the flux-loading section) and save with
`scripts/generate_dataset.py --benchmark pinte2009`.

### Task 5 — Replace C5G7-TD flux targets with a time-dependent SN solution

Currently the time-dependent target is `α(t) × φ_ss(x)` from point kinetics,
which assumes the spatial flux shape does not change during the transient.
A proper target requires a full space-time SN solve.

- **OpenSn** supports time-dependent transport (see its documentation).
- The converter in `src/data/converters/c5g7_td.py` is ready to accept
  externally provided flux snapshots; replace the `point_kinetics` section
  with loader code once solver output is available.

### Task 6 — Integrate trainable rational activations across all models

**Library**: [`rational_activations`](https://github.com/ml-research/rational_activations),
PyTorch API: `from rational.torch import Rational`.

The scientific question is whether replacing fixed nonlinearities (ReLU/GELU/SiLU)
with trainable rational functions improves **cross-regime, cross-resolution, and
SN-transfer generalization**. This requires changes across config, models, trainer,
evaluation scripts, and tests.

#### A — Feature flags and config

Add a unified activation configuration block used by all three models:

- `model.activation.name`: one of `["relu", "gelu", "silu", "rational"]`
- `model.activation.rational`: dict with keys `approx_func` (default `"leaky_relu"`),
  `degrees` (default `[5,4]`), `version` (default `"A"`), `trainable` (default `true`),
  `train_numerator` (default `true`), `train_denominator` (default `true`),
  `share_policy` (one of `"none"` / `"global"` / `"per_block"`),
  `capture_every_epochs` (default `0`), `capture_x_range` (default `[-5,5]`),
  `capture_num_points` (default `512`), `export_graphs` (default `false`),
  `export_dir` (default `"runs/<run_id>/rational_graphs"`)

Add optimizer config for rational parameters:

- `optim.rational_lr_mult` (default `1.0`) — multiplier applied to the base LR for rational params
- `optim.rational_weight_decay` (default same as base) — separate weight decay for rational params

Add three new Hydra config files that inherit from the baseline model configs and
only override the activation block:

- `configs/model/fno_rational.yaml`
- `configs/model/deeponet_rational.yaml`
- `configs/model/ap_micromacro_rational.yaml`

#### B — Activation factory in `src/models/common.py`

Add an `ActivationFactory` class that:

- Returns `nn.ReLU` / `nn.GELU` / `nn.SiLU` for standard activations.
- Imports `Rational` from `rational.torch` (with a clear `ImportError` if absent)
  and instantiates it with the config args for rational activations.
- Implements the three sharing policies via an internal registry:
  - `"global"` — one shared `Rational` instance reused everywhere
  - `"per_block"` — one instance per `block_id` key
  - `"none"` — a fresh instance at every call
- Provides helper functions `is_rational_module(m)` and `iter_rational_modules(model)`.
- Provides a curve-sampling helper that evaluates `y = rational(x)` over
  `capture_x_range` and returns `(x, y)` on CPU for TensorBoard logging or
  optional SVG/PNG export (guarded with `try/except ImportError` for matplotlib).

#### C — Update all three models to use the factory

Every hard-coded nonlinearity inside `src/models/fno.py`, `src/models/deeponet.py`,
and `src/models/ap_micromacro.py` must be constructed through `ActivationFactory`.
Use consistent `block_id` strings (e.g., `"fno_block_0"`, `"deeponet_branch"`,
`"ap_macro"`, `"ap_micro"`) so the sharing policy is deterministic. Forward
signatures and output semantics are unchanged.

#### D — Trainer updates in `src/trainers/trainer.py`

Split optimizer param groups:

- `base_params` — all parameters not belonging to `Rational` modules
- `rational_params` — parameters of `Rational` modules (numerator and denominator)
  with `lr = base_lr * rational_lr_mult` and `weight_decay = rational_weight_decay`

Ensure there is zero parameter overlap between groups.

Add rational logging:

- At training start: log number of `Rational` modules, their degrees/version, and
  numerator/denominator parameter counts.
- Every `capture_every_epochs` epochs (if > 0): sample each rational module's curve
  and log to TensorBoard; optionally export graphs to `export_dir`.
- Extend the existing NaN/Inf check to report whether NaNs originate from rational
  module outputs.

#### E — Evaluation: no logic changes

Checkpoint loading must work with `Rational` modules (they are standard `nn.Module`
subclasses, so `torch.load` / `model.load_state_dict` requires no changes). Metrics
and protocol logic are unchanged.

#### F — Run scripts: rational ablation table

Extend `run_all.py` (or add `scripts/run_with_rational.sh`) to run, for each
benchmark and each model family, a paired comparison:

- Standard activation run (e.g., SiLU baseline)
- Rational activation run (identical except activation config)

Each run executes smoke train → eval → SN-transfer → resolution-transfer →
ε-sweep. Aggregate all results into `runs/aggregate.csv` with columns:
`benchmark, model, activation, seed, train_res, test_res, train_Nw, test_Nw,
epsilon, sn_transfer_rel_l2, resolution_transfer_rel_l2, regime_sweep_rel_l2`.

Add optional config presets for degree/version ablations:

- Degrees `(3,2)` and `(7,6)` as alternatives to the default `(5,4)`
- Version `"B"` as an alternative to `"A"`

#### G — Tests

Add tests to `tests/` that verify:

- `ActivationFactory` returns a `Rational` module when configured with
  `name="rational"`, and a standard `nn.Module` otherwise.
- Forward pass succeeds for all three models with rational activation and
  variable-size ω batches (the discretization-agnostic case).
- Optimizer param groups contain no parameter overlap and rational params are
  routed to the correct group.

#### Commands (to add to README after implementation)

```bash
# Train AP Micro-Macro with rational activations on C5G7
python train.py --benchmark c5g7 --model ap_micromacro \
    model.activation.name=rational

# Train FNO with rational activations
python train.py --benchmark c5g7 --model fno \
    model.activation.name=rational \
    optim.rational_lr_mult=2.0

# Full rational-vs-standard ablation (all benchmarks, all models)
python run_all.py --activation silu rational

# Degree sweep
python sweep.py --benchmark c5g7 --model ap_micromacro \
    model.activation.name=rational \
    "model.activation.rational.degrees=[3,2],[5,4],[7,6]"
```

---

## Quick Start

> **Note on data:** `generate_dataset.py` creates **synthetic training data on your
> machine** using built-in physics approximations. It does NOT download official
> benchmark data or call an external solver. The resulting `runs/datasets/` files are
> sufficient to develop and test model architectures but are **not** official reference
> solutions. See [Pending Work](#pending-work)
> for instructions on obtaining the real OECD/NEA data and running validated solvers.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate synthetic datasets (diffusion/first-flight approx — no external solver needed)
python scripts/generate_dataset.py --benchmark c5g7      --n_samples 200 --split train
python scripts/generate_dataset.py --benchmark c5g7      --n_samples 50  --split val
python scripts/generate_dataset.py --benchmark kobayashi --n_samples 100 --split train
python scripts/generate_dataset.py --benchmark pinte2009 --n_samples 150 --split train

# 3. Train one model
python train.py --benchmark c5g7 --model ap_micromacro --n_epochs 100

# 4. Evaluate
python eval.py --checkpoint runs/c5g7_ap_micromacro_seed42/checkpoints/best.pt \
               --protocol all

# 5. Run everything in one shot (generate → train all models → eval → aggregate CSV)
python run_all.py --quick          # smoke test, ~3 min
python run_all.py                  # full run, ~1-4h depending on GPU

# 6. Hyperparameter sweep
python sweep.py --benchmark c5g7 --model ap_micromacro --seeds 1 2 3
```

---

## One-Shot Pipeline: `run_all.py`

`run_all.py` automates the entire research pipeline end-to-end:

```
generate data → train (FNO, DeepONet, AP Micro-Macro) → evaluate (3 protocols) → aggregate CSV
```

```bash
# Smoke test — ~3 minutes, 20 training samples, 3 epochs
python run_all.py --quick

# Single benchmark + model
python run_all.py --benchmark c5g7 --model ap_micromacro

# Multiple benchmarks
python run_all.py --benchmark c5g7 kobayashi --model fno ap_micromacro

# Custom scale
python run_all.py --epochs 200 --n_train 500 --n_val 100

# Skip re-generating data (use existing runs/datasets/)
python run_all.py --skip_generate

# Skip training (use existing checkpoints)
python run_all.py --skip_train
```

**Outputs** after a full run:

```
runs/aggregate/all_results.csv              ← THE paper table
runs/eval/<benchmark>_<model>/
    sn_transfer.csv
    resolution_transfer.csv
    regime_sweep.csv
runs/<benchmark>_<model>/checkpoints/best.pt
```

The terminal prints a summary table:

```
Benchmark              Model              sn_transfer    resolution_transfer  regime_sweep
c5g7                   ap_micromacro      0.0412         0.0631               0.0887
c5g7                   fno                0.0589         0.0714               0.1203
...
```

---

---

## Models

### FNO — Fourier Neural Operator
FFT-based spectral convolution on the uniform spatial grid, lifted with Fourier
positional features. An angular query head evaluates at arbitrary ω directions
(discretization-agnostic). Moments φ, J are computed by quadrature over the
predicted intensity.

### DeepONet
- **Branch net**: CNN that encodes the input fields (σ_a, σ_s, q) into a latent vector
- **Trunk net**: MLP that encodes query points (x, ω, t) with Fourier features
- Combines via dot product; supports variable-size angular sets at evaluation time

### AP Micro-Macro *(recommended baseline)*
Explicitly decomposes intensity into macro and micro parts:

```
Macro net  →  φ(x), J(x)            (FNO-based, grid quantities)
I_P1(x,ω)  =  φ/(4π) + (3/4π) J·ω  (deterministic P1 closure)
Micro net  →  R(x, ω)               (residual, angular query head)
I(x, ω)    =  I_P1(x, ω) + R(x, ω)
```

Loss = intensity L2 + moment consistency + diffusion-limit regularization (ε-weighted).
This model is consistent with the diffusion limit by construction.

---

## Evaluation Protocols

| Protocol | What it tests | Key metric |
|---|---|---|
| **SN Transfer** | Train S8, test S4/S16/S32/S64 | I_rel_l2 vs N_ω |
| **Resolution Transfer** | Train 17×17, test 34×34, 68×68 | I_rel_l2 vs multiplier |
| **Regime Sweep** | ε from 0.001 to 5.0 | I_rel_l2 vs ε |

---

## Benchmark Tasks

| Benchmark | Type | Groups | Dim | XS source | Flux source |
|---|---|---|---|---|---|
| `c5g7` | Steady eigenvalue | 7 | 2D | NEA/NSC/DOC(2003)16, Table 2 | Diffusion approx / OpenMC |
| `c5g7_td` | Time-dependent | 7 | 2D | Same + delayed-n kinetics | Point kinetics × SS shape |
| `kobayashi` | Fixed source, void | 1 | 3D | NSC-DOC(2000)4 | First-flight kernel |
| `pinte2009` | Radiative transfer | 1 | 2D | Pinte et al. 2009, Table 1 | 5-step Λ-iteration |

---

## File Tree

```
src/
  data/
    schema.py              Canonical dataclasses (InputFields, QueryPoints, TargetFields)
    dataset.py             PyTorch Dataset + collate_fn (variable-Nω padding)
    io.py                  HDF5 writer/reader (h5py on Windows, zarr on Linux/macOS)
    converters/
      c5g7.py              Quarter-core geometry, 6 materials, diffusion flux
      c5g7_td.py           Rod-ejection transient, 6-group point kinetics
      kobayashi.py         L-shaped/dogleg void duct, first-flight flux
      pinte2009.py         Protoplanetary disk, Λ-iteration intensity
  models/
    common.py              FourierFeatures, SphericalFourierFeatures, MLP, SpectralConv
    fno.py                 FNO with angular query head
    deeponet.py            DeepONet (branch + trunk)
    ap_micromacro.py       AP Micro-Macro (P1 + residual)
  trainers/
    trainer.py             Training loop: AMP, EMA, grad clipping, TensorBoard, checkpointing
  eval/
    metrics.py             l2_error, relative_l2, moment errors, energy balance
    protocols.py           SNTransferProtocol, ResolutionTransferProtocol, RegimeSweepProtocol
  solvers/
    openmc_c5g7_model.py   ★ Full C5G7 OpenMC model (real MC, needs conda binary)
    openmc_interface.py    OpenMC interface + diffusion fallback
    opensn_interface.py    OpenSn stub (real API calls ready, needs OpenSn installed)
    mock_backend.py        Diffusion/P1 approximation backend
  utils/
    seed.py                Deterministic seeding (CUDA-safe)
    io_utils.py            save_json / save_csv (robust to variable column sets)
    logging_utils.py       Logging setup
    config.py              Hydra config dataclasses
configs/
  benchmarks/
    c5g7.yaml              spatial_shape=[51,51], n_groups=7
    c5g7_td.yaml           time_dependent=true, n_time=20
    kobayashi.yaml         spatial_shape=[20,20,20], problem=1
    pinte2009.yaml         wavelength_um=1.0
  models/
    fno.yaml
    deeponet.yaml
    ap_micromacro.yaml
  train.yaml
  eval.yaml
scripts/
  generate_dataset.py      Generate any benchmark dataset
  run_openmc_c5g7.py       ★ Run OpenMC → generate real MC flux dataset
  convert_c5g7.py          Standalone C5G7 converter
  convert_c5g7_td.py
  convert_kobayashi_void.py
  convert_pinte2009.py
  inspect_dataset.py       Print dataset statistics, optionally plot
  run_baselines.sh         Legacy bash smoke test
tests/
  test_schema.py
  test_models.py
  test_dataset.py
  test_metrics.py
  test_trainer.py
train.py                   Main training CLI
eval.py                    Main evaluation CLI
sweep.py                   Hyperparameter grid sweep
run_all.py                 ★ One-shot pipeline: generate → train → eval → aggregate
requirements.txt
```

---

## Data Schema

Every sample is a `TransportSample` with three parts:

```python
sample.inputs   # InputFields:  sigma_a [nx,ny,G], sigma_s, q, BCSpec, params, metadata
sample.query    # QueryPoints:  x [Nx,d], omega [Nω,d], w_omega [Nω], t (optional)
sample.targets  # TargetFields: I [Nx,Nω,G], phi [Nx,G], J [Nx,d,G], qois dict
```

`collate_fn` in `dataset.py` handles variable `Nω` across samples by padding and
returning an `omega_mask`.

Storage: h5py HDF5 on Windows (no atomic-rename issues), zarr on Linux/macOS.

---

## Configuration

Uses [Hydra](https://hydra.cc/) for config composition. Quick override examples:

```bash
python train.py benchmark=c5g7 model=fno trainer.lr=5e-4 trainer.n_epochs=50
python eval.py  benchmark=c5g7 model=ap_micromacro eval.protocol=regime_sweep
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this codebase, please cite the relevant benchmarks:

- **C5G7**: OECD/NEA, NEA/NSC/DOC(2003)16 — *Benchmark on Deterministic Transport Calculations Without Spatial Homogenisation*
- **C5G7-TD**: OECD/NEA, NEA/NSC/DOC(2016)7 — *Deterministic Time-Dependent Neutron Transport Benchmark*
- **Kobayashi**: OECD/NEA, NSC-DOC(2000)4 — *3-D Radiation Transport Benchmark Problems with Void Region*
- **Pinte 2009**: Pinte et al., A&A 498, 967–980 (2009) — *Benchmark problems for continuum radiative transfer*
- **OpenMC**: Romano et al., Ann. Nucl. Energy 82 (2015) 90–97
