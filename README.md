# Rational AP Transport Operator

A PyTorch research codebase for learning the **solution operator of radiative/particle transport**:

> **T : (σ_a, σ_s, q, BC, ε, g) → I(x, ω)**

The model is queried at arbitrary spatial points **x** and angular directions **ω** at evaluation time — no retraining when the discretization changes.

---

## Research Goal

The goal is to develop and evaluate an **asymptotic-preserving, discretization-agnostic neural operator** that maps cross-section fields, sources, boundary conditions, and transport regime parameter ε to the full angular intensity I(x, ω) and its derived moments (φ, J). Each training sample is a genuinely different physics problem — independently perturbed cross-sections, source strengths, boundary conditions, and transport regime — not just the same physics at different resolutions.

The central scientific question is whether **trainable rational activation functions** materially improve generalization across three axes relative to standard activations (ReLU/GELU/SiLU):

1. **SN order transfer** — trained on S8, evaluated on S4 / S16 / S32 / S64
2. **Spatial resolution transfer** — trained at base resolution, evaluated at 2× / 4× finer grids
3. **Cross-regime (ε sweep)** — trained across ε ∈ [0.01, 1], evaluated at ε → 0 (diffusion limit) and ε = 1 (transport limit)

---

## The AP Micro-Macro Model

### The core difficulty: stiffness across regimes

The scaled linear transport equation reads:

> **Ω·∇I = (1/ε)(⟨I⟩ − I) + source**

where ε is the Knudsen number (mean free path / domain size). When **ε = O(1)** (transport regime) the angular variable Ω matters and the solution is strongly anisotropic. When **ε → 0** (diffusion regime) the collision term dominates, the solution becomes nearly isotropic, and I(x,ω) ≈ φ(x)/(4π) everywhere — the full angular dependence collapses to a scalar.

A plain neural network (FNO, DeepONet) trained across all ε must learn both extremes from the same weights, and tends to average them and do neither well. This is the failure mode that asymptotic-preserving methods are designed to avoid.

### The decomposition

The AP Micro-Macro model avoids this by **hard-coding the diffusion limit into the architecture**:

```
I(x, ω)  =  I_P1(x, ω)           +   R(x, ω)
             ─────────────────         ──────────────
             MacroNet output           MicroNet output
             deterministic P1          learned residual
             φ(x)/(2π) + J(x)·ω·1/π   captures anisotropy
                                        beyond P1
```

**I_P1** is not learned — it is the explicit P1 angular reconstruction from MacroNet's predicted moments:
- 2D: `I_P1(x,ω) = φ(x)/(2π) + (1/π) J(x)·ω`
- 3D: `I_P1(x,ω) = φ(x)/(4π) + (3/4π) J(x)·ω`

This is the leading-order Chapman-Enskog term; it integrates to give φ exactly and its first angular moment gives J exactly. It is isotropic when J = 0, which is precisely the diffusion limit.

**MacroNet** (FNO-based) predicts the spatial moments φ(x) and J(x) from the full input fields (σ_a, σ_s, q, BC). The FNO backbone captures global spatial correlations across the domain.

**MicroNet** (MLP-based) predicts the angular residual R(x, ω) — the deviation from P1. It takes MacroNet's spatial latent, Fourier features of ω, and log(ε) as input, allowing it to learn to suppress corrections in the diffusion regime (where P1 is already accurate).

**Loss** = intensity L2 + moment consistency (quadrature moments of predicted I must agree with MacroNet's direct φ, J predictions) + ε-weighted diffusion regularization (extra gradient signal when ε is small).

The diffusion limit is **exact by construction**: as ε → 0, R → 0 and the model reduces to a diffusion solver. MacroNet only needs to learn the diffusion equation rather than discovering the limit from data. This design follows the micro-macro decomposition of Lemou & Mieussens (2008) and the APNN framework of Jin & Ma (2022), extended from single-instance PINNs to full operator learning over coefficient fields.

---

## Baseline Models

### FNO — Fourier Neural Operator

FFT-based spectral convolution on the spatial grid, lifted with Fourier positional features. An angular query head evaluates at arbitrary ω directions. Moments φ, J are computed by quadrature over predicted intensity. Implemented in [`src/models/fno.py`](src/models/fno.py).

### DeepONet

- **Branch net**: CNN encoding input fields (σ_a, σ_s, q) into a latent vector
- **Trunk net**: MLP encoding query points (x, ω) with Fourier features
- Combined via dot product; supports variable-size angular sets at evaluation time

Implemented in [`src/models/deeponet.py`](src/models/deeponet.py).

---

## Benchmarks

### C5G7 — 2D Nuclear Criticality Eigenvalue

The OECD/NEA C5G7 benchmark: a 2D quarter-core PWR with 6 materials (UO2, MOX 4.3/7.0/8.7%, guide tube, fission chamber, moderator) on a 51×51 spatial grid, 7 neutron energy groups.

**Ground truth**: real OpenMC Monte Carlo eigenvalue flux (5 M active particle histories per sample, ~0.7% flux uncertainty). Each sample runs a fully independent OpenMC simulation with its own perturbed cross-section library.

**What varies per sample** (out of T : (σ_a, σ_s, q, BC, ε, g) → I):

| Input | Status | Detail |
|---|---|---|
| σ_a (absorption) | ✅ **Varies** | ×U[0.9, 1.1] independently per material per energy group (7 materials × 7 groups = 49 independent scalars) |
| σ_s (scattering) | ✅ **Varies** | ×U[0.9, 1.1] independently per material per group, drawn independently of σ_a |
| q / ν·σ_f (fission source) | ✅ **Varies** | ×U[0.9, 1.1] per material per group (eigenvalue problem — fission source is derived from ν·σ_f, not an independent q) |
| BC | ❌ **Fixed** | Vacuum (zero incoming flux) on all four sides — fixed by the C5G7 benchmark definition |
| ε | ❌ **Not physical** | Stored as a model-input label but **does not affect the OpenMC simulation**. Physics is identical across all ε labels. Regime sweep is therefore skipped for C5G7. |
| g (anisotropy) | ❌ **Fixed** | 0 (isotropic scattering) — fixed by the C5G7 benchmark definition |
| Geometry | ❌ **Fixed** | 51×51 quarter-core pin layout — fixed by the C5G7 benchmark definition |

The ±10% XS perturbation (`--xs_perturb 0.10`) produces meaningfully distinct flux shapes: the C5G7 flux is sensitive to material composition, so k-eff and the spatial flux distribution both change substantially across samples.

### Pinte 2009 — 2D Protoplanetary Disk Radiative Transfer

The Pinte et al. (2009) benchmark: a 2D axisymmetric protoplanetary disk with dust scattering, on a 32×32 spatial grid, 1 wavelength group.

**Ground truth**: currently a 5-step Λ-iteration approximation with P1 angular reconstruction. Real targets require MCFOST or RADMC-3D (see [Pending Work](#pending-work)).

**What varies per sample** (out of T : (σ_a, σ_s, q, BC, ε, g) → I):

| Input | Status | Detail |
|---|---|---|
| σ_a (absorption) | ✅ **Varies** | κ_abs ×U[0.9, 1.1] → σ_a = κ_abs · ρ(x) |
| σ_s (scattering) | ✅ **Varies** | κ_sca ×U[0.9, 1.1] → σ_s = κ_sca · ρ(x) |
| q (stellar heating) | ✅ **Varies** | proportional to B_lam (the stellar boundary condition) |
| BC | ✅ **Varies** | B_lam ×U[0.8, 1.2] — stellar irradiation inflow driving the entire RT problem |
| ε | ✅ **Varies** | U[ε_min, ε_max] per sample — physically meaningful, governs optical depth / scattering regime |
| g (anisotropy) | ✅ **Varies** | Henyey-Greenstein g ×U[0.9, 1.1], clipped to [0, 0.99] |
| Geometry | ✅ **Varies** | R_in ×U[0.7, 1.3], flaring index ξ ×U[0.8, 1.2], scale height H₀ ×U[0.7, 1.3] |

Pinte 2009 has the richest per-sample diversity: all six inputs of T vary, including geometry and BC.

---

## Quick Start

### Requirements

- Python 3.9+, PyTorch 2.0+ (CUDA recommended)
- `pip install -r requirements.txt`
- **OpenMC** for real C5G7 targets: `conda install -c conda-forge openmc`
  (without it, a mock analytic fallback is used and a WARNING is logged)

### One-shot run

```bash
git clone https://github.com/your-org/rational-ap-transport-operator
cd rational-ap-transport-operator
pip install -r requirements.txt

python run_all.py --quick          # smoke test, ~5 min
python run_all.py                  # full run, both benchmarks × 3 models
python run_all.py --benchmark c5g7 --model ap_micromacro  # single combo
```

`run_all.py` runs the complete pipeline: generate → train → evaluate → aggregate. All outputs land in `runs/`.

### Step-by-step

**1. Generate datasets**

```bash
python scripts/generate_dataset.py --benchmark c5g7 --split train --n_samples 200
python scripts/generate_dataset.py --benchmark c5g7 --split val   --n_samples 50
# test + resolution_x2/x4 in one call:
python scripts/generate_dataset.py --benchmark c5g7 --split all_eval --n_samples 50
```

Add `--n_batches 300` for full publication-quality OpenMC accuracy (~85 s/sample vs ~17 s default).

**2. Train**

```bash
python train.py --benchmark c5g7 --model ap_micromacro \
    --n_epochs 100 --data_dir runs/datasets
```

**3. Evaluate**

```bash
python eval.py \
    --checkpoint runs/c5g7_ap_micromacro/checkpoints/best.pt \
    --benchmark c5g7 --model ap_micromacro \
    --protocol all --data_dir runs/datasets
```

---

## Evaluation Protocols

| Protocol | What it tests | Benchmarks |
|---|---|---|
| `test_set` | Held-out test split at training resolution | Both |
| `sn_transfer` | Generalization to unseen SN orders (S4 → S64) | Both |
| `resolution_transfer` | Generalization to 2×/4× finer spatial grids (bilinear-interpolated targets) | Both |
| `regime_sweep` | Generalization across ε ∈ [0.001, 5.0] | Pinte 2009 only |

All protocols read entirely from pre-generated disk splits — no solver is invoked during evaluation. Pass `--split all_eval` to `generate_dataset.py` to produce all evaluation splits upfront.

---

## Pending Work

### Task 1 — Real flux targets for Pinte 2009

Integrate MCFOST or RADMC-3D:
- **MCFOST**: https://mcfost.readthedocs.io
- **RADMC-3D**: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/

After running, convert output with [`scripts/convert_pinte2009.py`](scripts/convert_pinte2009.py) and register the solver in `_BENCHMARK_SOLVER_PREFERENCE["pinte2009"]` in [`src/solvers/__init__.py`](src/solvers/__init__.py).

### Task 2 — Trainable rational activations

Integrate [`rational_activations`](https://github.com/ml-research/rational_activations) (`from rational.torch import Rational`) across all three models:

- **Config**: add `model.activation.name` (`relu`/`gelu`/`silu`/`rational`) and rational hyperparams (degree, sharing policy) to model configs
- **`src/models/common.py`**: add `ActivationFactory` returning standard or `Rational` modules
- **All three models**: replace hard-coded nonlinearities with `ActivationFactory` calls
- **`src/trainers/trainer.py`**: split optimizer into `base_params` and `rational_params` groups with separate LR/weight-decay
- **`run_all.py`**: add paired ablation (standard vs rational) for each benchmark × model
- **Tests**: verify forward pass, optimizer param groups, and `ActivationFactory` dispatch

---

## Repository Structure

```
rational-ap-transport-operator/
├── train.py              Training CLI
├── eval.py               Evaluation CLI (4 protocols)
├── sweep.py              Hyperparameter grid sweep
├── run_all.py            One-shot pipeline: generate → train → eval → aggregate
├── requirements.txt
├── configs/
│   ├── benchmarks/       c5g7.yaml, pinte2009.yaml
│   └── models/           fno.yaml, deeponet.yaml, ap_micromacro.yaml
├── scripts/
│   ├── generate_dataset.py
│   ├── run_openmc_c5g7.py
│   ├── convert_c5g7.py
│   ├── convert_pinte2009.py
│   └── inspect_dataset.py
├── src/
│   ├── data/             schema.py, dataset.py, io.py, converters/
│   ├── models/           fno.py, deeponet.py, ap_micromacro.py, common.py
│   ├── trainers/         trainer.py
│   ├── eval/             metrics.py, protocols.py
│   ├── solvers/          openmc_interface.py, mock_backend.py
│   └── utils/            config.py, logging_utils.py, seed.py, io_utils.py
└── tests/
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Citations

**Benchmarks**
- OECD/NEA. *Benchmark on Deterministic Transport Calculations Without Spatial Homogenisation.* NEA/NSC/DOC(2003)16. (C5G7)
- Pinte et al. *Benchmark problems for continuum radiative transfer.* A&A 498, 967–980 (2009). https://doi.org/10.1051/0004-6361/200811474

**Neural operators**
- Li et al. *Fourier Neural Operator for Parametric PDEs.* ICLR 2021. https://arxiv.org/abs/2010.08895
- Lu et al. *Learning Nonlinear Operators via DeepONet.* Nature MI 3, 218–229 (2021). https://doi.org/10.1038/s42256-021-00302-5
- Kovachki et al. *Neural Operator: Learning Maps Between Function Spaces.* JMLR 24(89) (2023). https://arxiv.org/abs/2108.08481

**Asymptotic-preserving methods**
- Jin & Ma. *Asymptotic-Preserving Neural Networks for Multiscale Kinetic Equations.* CiCP 31(5) (2022). https://doi.org/10.4208/cicp.OA-2021-0166
- Lemou & Mieussens. *A New AP Scheme Based on Micro-Macro Formulation.* SIAM J. Sci. Comput. 31(1) (2008). https://doi.org/10.1137/07069479X
- Jin. *Efficient AP Schemes for Multiscale Kinetic Equations.* SIAM J. Sci. Comput. 21(2) (1999). https://doi.org/10.1137/S1064827598334599

**Reference solver**
- Romano et al. *OpenMC: A State-of-the-Art Monte Carlo Code.* Ann. Nucl. Energy 82, 90–97 (2015). https://doi.org/10.1016/j.anucene.2014.07.048
