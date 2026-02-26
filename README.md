# Discretization-Agnostic Neural Operators for Radiative/Particle Transport

A PyTorch research repository implementing baseline neural operator models for
radiative and particle transport. The core goal is to learn the **solution operator**

> **T : (σ_a, σ_s, q, BC, ε, g) → I(x, ω)**

where the model is queried at arbitrary spatial points **x** and angular directions
**ω** at evaluation time — no re-training needed when the discretization changes.

---

## Research Goal

Standard neural PDE solvers are tied to a fixed grid and a fixed set of angular
directions (SN order). This repository tests whether neural operators can transfer
across:

1. **SN order** — trained on S8, evaluated on S4 / S16 / S32
2. **Spatial resolution** — trained on 17×17, evaluated on 34×34 / 68×68
3. **Physical regime** — trained across ε ∈ [0.01, 1], evaluated at ε → 0 (diffusion
   limit) and ε = 1 (transport limit)

The AP Micro-Macro model explicitly decomposes the intensity as
`I = I_P1(φ, J) + R`, which by construction recovers the correct diffusion
limit as ε → 0.

---

## What is Real vs Approximate

### Inputs (cross sections, geometry) — fully real

| Benchmark | What is real |
|---|---|
| **C5G7** | Published 7-group XS for all 6 materials (UO2, MOX4.3/7.0/8.7, guide tube, fission chamber, moderator); full 51×51 quarter-core geometry (3×3 assemblies × 17×17 pin lattice) |
| **C5G7-TD** | Same XS as C5G7 + published 6-group delayed-neutron kinetics parameters (β_i, λ_i) from NEA/NSC/DOC(2016)7; rod-ejection transient modelled with RK4 point kinetics |
| **Kobayashi** | Exact geometry (L-shaped duct, dogleg, dog-ear) + published σ_t values (source=0.1, void=1e-8, absorber=10.0 cm⁻¹) |
| **Pinte 2009** | Published disk density law (Σ ∝ r⁻¹, flared scale height), stellar parameters (T_eff=9500 K, L=47 L_sun), dust opacity from Mie theory (κ_abs=2.3, κ_sca=10.4 cm²/g, g=0.60) |

### Flux/intensity targets — approximate (pending solver)

| Benchmark | Current target | Real target requires |
|---|---|---|
| **C5G7** | Diffusion approximation (–D∇²φ + σ_a φ = S) | OpenMC multi-group MC or OpenSn SN |
| **C5G7-TD** | Point kinetics × steady-state shape | Time-dependent SN solver |
| **Kobayashi** | First-flight transport kernel (exact in void) | Full SN or MC in absorber region |
| **Pinte 2009** | 5-step Λ-iteration (accurate for τ < few) | MCFOST or RADMC-3D RT solver |

**The OpenMC model for C5G7 is fully implemented** (see `src/solvers/openmc_c5g7_model.py`)
and will produce real Monte Carlo flux once the `openmc` binary is installed via conda.
See [Getting Real Flux Targets](#getting-real-flux-targets) below.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate datasets (uses diffusion/first-flight approximation — no solver needed)
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

## Getting Real Flux Targets (OpenMC)

The complete C5G7 OpenMC model is implemented and ready. You need the `openmc`
binary, which requires conda:

```bash
# Step 1 — Install OpenMC with binary (Windows / Linux / macOS)
conda create -n openmc_env python=3.11
conda activate openmc_env
conda install -c conda-forge openmc
pip install torch numpy h5py tqdm scipy einops tensorboard

# Step 2 — Quick test (~2 min, 5000 particles)
cd "C:\Users\Maosen\2026 Neurips"
python scripts/run_openmc_c5g7.py \
    --n_particles 5000 --n_batches 20 --n_inactive 5 --n_samples 5
# Expected: k-effective ≈ 1.185x  (published reference: 1.18655 ± 0.00033)

# Step 3 — Full production run (~30 min CPU, ~5 min with MPI)
python scripts/run_openmc_c5g7.py \
    --splits train val test \
    --n_samples 200 50 50 \
    --n_particles 500000 --n_batches 500 --n_inactive 100
```

This writes `runs/datasets/c5g7_train.zarr.h5` etc. with real Monte Carlo flux
targets. The rest of the pipeline (`train.py`, `eval.py`, `run_all.py`) works
unchanged.

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
