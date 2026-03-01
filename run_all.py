"""
One-shot script: generate data → train all models → evaluate all protocols → aggregate.

Usage:
    python run_all.py                  # full legitimate run (~hours)
    python run_all.py --quick          # smoke test, ~3 min
    python run_all.py --benchmark c5g7 --model ap_micromacro   # single combo
    python run_all.py --epochs 200 --n_train 500                     # custom scale

Output:
    runs/aggregate/all_results.csv       ← THE paper table (all models x benchmarks x protocols)
    runs/eval/<run>/test_set.csv
    runs/eval/<run>/sn_transfer.csv
    runs/eval/<run>/resolution_transfer.csv
    runs/eval/<run>/regime_sweep.csv     ← pinte2009 only
    runs/<run>/checkpoints/best.pt

Protocol support by benchmark:
    c5g7      : test_set, sn_transfer, resolution_transfer
    pinte2009 : test_set, sn_transfer, resolution_transfer, regime_sweep
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

BENCHMARKS = ["c5g7", "pinte2009"]
MODELS = ["fno", "deeponet", "ap_micromacro"]

# Per-benchmark defaults for eval model args
# (must match train.py defaults so checkpoint loads correctly)
BENCH_EVAL_ARGS = {
    "c5g7":      ["--macro_channels", "32", "--n_fno_blocks", "4", "--n_modes", "12", "--micro_latent_dim", "64", "--n_basis", "128"],
    "pinte2009": ["--macro_channels", "32", "--n_fno_blocks", "4", "--n_modes", "12", "--micro_latent_dim", "64", "--n_basis", "128"],
}

# Benchmarks that do not support regime_sweep:
#   c5g7 – k-eigenvalue problem solved by OpenMC. Epsilon is stored only
#           as a model-input label and does not affect the OpenMC physics,
#           so a regime sweep would not test genuine transport-regime variation.
NO_REGIME_SWEEP = {"c5g7"}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(cmd: list[str], tag: str) -> bool:
    """Run a subprocess, stream output live, return True on success."""
    log.info(f"  RUN [{tag}]: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, executable=sys.executable if cmd[0] == sys.executable else None)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else "FAILED"
    log.info(f"  [{tag}] {status} in {elapsed:.1f}s")
    return ok


def py(*args) -> list[str]:
    """Build a python subprocess command."""
    return [sys.executable] + list(args)


# ──────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────

def step_generate(
    benchmark: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_particles: int = 100_000,
    n_batches: int = 100,
    n_inactive: int = 50,
) -> bool:
    """Generate ALL splits before training begins."""
    ok = True

    def _gen(split, n):
        return run(py("scripts/generate_dataset.py",
                      "--benchmark",   benchmark,
                      "--solver",      "auto",
                      "--n_samples",   str(n),
                      "--split",       split,
                      "--output_dir",  "runs/datasets",
                      "--seed",        "42",
                      "--n_particles", str(n_particles),
                      "--n_batches",   str(n_batches),
                      "--n_inactive",  str(n_inactive)),
                   tag=f"gen:{benchmark}:{split}")

    # ALL three generation steps complete before training starts.
    ok &= _gen("train",    n_train)
    ok &= _gen("val",      n_val)
    # all_eval: test + resolution_x2/x4 + every regime_eps* in one call
    ok &= _gen("all_eval", n_test)

    return ok


def step_train(benchmark: str, model: str, epochs: int, batch: int,
               n_train: int, n_val: int, seed: int) -> Path | None:
    run_name = f"{benchmark}_{model}"
    ok = run(py("train.py",
                "--benchmark",        benchmark,
                "--model",            model,
                "--n_epochs",         str(epochs),
                "--batch_size",       str(batch),
                "--n_samples_train",  str(n_train),
                "--n_samples_val",    str(n_val),
                "--run_name",         run_name,
                "--seed",             str(seed),
                "--log_dir",          "runs",
                "--data_dir",         "runs/datasets"),
             tag=f"train:{benchmark}:{model}")
    if not ok:
        return None
    ckpt_dir = Path("runs") / run_name / "checkpoints"
    best = ckpt_dir / "best.pt"
    latest = ckpt_dir / "latest.pt"
    ckpt = best if best.exists() else (latest if latest.exists() else None)
    if ckpt is None:
        log.error(f"  No checkpoint found in {ckpt_dir}")
    return ckpt


def step_eval(benchmark: str, model: str, ckpt: Path,
              protocol: str, n_test: int, batch: int,
              extra_args: list[str]) -> bool:
    run_name = f"{benchmark}_{model}"
    return run(py("eval.py",
                  "--checkpoint",     str(ckpt),
                  "--benchmark",      benchmark,
                  "--model",          model,
                  "--protocol",       protocol,
                  "--output_dir",     f"runs/eval/{run_name}",
                  "--n_test_samples", str(n_test),
                  "--batch_size",     str(batch),
                  "--seed",           "42",
                  "--data_dir",       "runs/datasets",
                  *extra_args),
               tag=f"eval:{benchmark}:{model}:{protocol}")


def step_aggregate():
    """Collect every per-run CSV into runs/aggregate/all_results.csv."""
    # Files to skip — these are derived/summary outputs, not primary protocol CSVs
    SKIP_STEMS = {"summary", "all_results"}
    PROTOCOL_STEMS = {"test_set", "sn_transfer", "resolution_transfer", "regime_sweep"}

    agg_rows = []
    for csv_path in glob.glob("runs/eval/**/*.csv", recursive=True):
        p = Path(csv_path)
        if p.stem in SKIP_STEMS or p.stem not in PROTOCOL_STEMS:
            continue
        run_id  = p.parent.name          # e.g. c5g7_ap_micromacro
        protocol = p.stem                 # e.g. sn_transfer
        # Parse run_id → benchmark and model
        # Convention: run_id = {benchmark}_{model}
        parts = run_id.split("_")
        # model is last 1 or 2 tokens; benchmark is everything before
        if "micromacro" in run_id:
            model = "ap_micromacro"
            bm = run_id.replace("_ap_micromacro", "")
        elif run_id.endswith("_fno"):
            model = "fno"
            bm = run_id[:-4]
        elif run_id.endswith("_deeponet"):
            model = "deeponet"
            bm = run_id[:-9]
        else:
            model = "unknown"
            bm = run_id

        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    row["run_id"]    = run_id
                    row["benchmark"] = bm
                    row["model"]     = model
                    row["protocol"]  = protocol
                    agg_rows.append(row)
        except Exception as e:
            log.warning(f"  Could not read {csv_path}: {e}")

    if not agg_rows:
        log.warning("  No CSV results found to aggregate.")
        return

    out = Path("runs/aggregate/all_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in agg_rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in agg_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    log.info(f"  Aggregate CSV: {out}  ({len(agg_rows)} rows)")

    # Also print a quick summary table to stdout
    _print_summary(agg_rows)


def _print_summary(rows: list[dict]):
    """Print a compact table: benchmark × model × protocol → key metrics."""
    from collections import defaultdict

    # ── I_rel_l2 table ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY  (I_rel_l2, lower is better)")
    print("=" * 100)

    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        val = r.get("I_rel_l2", "")
        try:
            table[r["benchmark"]][r["model"]][r["protocol"]].append(float(val))
        except (ValueError, TypeError):
            pass

    protocols = ["test_set", "sn_transfer", "resolution_transfer", "regime_sweep"]
    col_w = 22
    header = f"{'Benchmark':<22} {'Model':<18} " + "  ".join(f"{p:<{col_w}}" for p in protocols)
    print(header)
    print("-" * len(header))

    for bm in sorted(table):
        for mdl in sorted(table[bm]):
            row_str = f"{bm:<22} {mdl:<18} "
            for proto in protocols:
                vals = table[bm][mdl][proto]
                if vals:
                    row_str += f"{sum(vals)/len(vals):<{col_w}.4f}  "
                else:
                    row_str += f"{'N/A':<{col_w}}  "
            print(row_str)
        print()

    print("=" * 100)
    print("Full results: runs/aggregate/all_results.csv")
    print("Per-run CSVs: runs/eval/<benchmark>_<model>/*.csv")
    print("Checkpoints:  runs/<benchmark>_<model>/checkpoints/best.pt")
    print("=" * 100 + "\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot: generate → train → evaluate → aggregate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test mode: tiny data, 3 epochs, ~3 min total")
    p.add_argument("--benchmark", nargs="+", default=None,
                   help="Subset of benchmarks, e.g. --benchmark c5g7 pinte2009")
    p.add_argument("--model", nargs="+", default=None,
                   help="Subset of models, e.g. --model ap_micromacro fno")
    p.add_argument("--epochs",  type=int, default=None, help="Override training epochs")
    p.add_argument("--n_train", type=int, default=None, help="Override n training samples")
    p.add_argument("--n_val",   type=int, default=None, help="Override n val samples")
    p.add_argument("--n_test",  type=int, default=None, help="Override n test samples for eval")
    p.add_argument("--batch",   type=int, default=None, help="Override batch size")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip_generate", action="store_true",
                   help="Skip dataset generation (use existing runs/datasets/)")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training (use existing checkpoints)")
    # OpenMC settings — particle count matches publication; batches reduced for speed
    p.add_argument("--n_particles", type=int, default=100_000,
                   help="MC particles per batch (default 100 000, same as publication).")
    p.add_argument("--n_batches", type=int, default=100,
                   help="Total batches: 50 inactive + 50 active (default 100). "
                        "Use 300 for full publication quality (50 inactive + 250 active).")
    p.add_argument("--n_inactive", type=int, default=50,
                   help="Inactive batches for fission source convergence (default 50, "
                        "same as OECD/NEA C5G7 reference — do not reduce).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── environment summary ──
    from src.utils.logging_utils import log_environment_info as _lei
    benchmarks_to_run = args.benchmark or BENCHMARKS
    models_to_run     = args.model     or MODELS
    # Log once for the first benchmark/model to show GPU; per-benchmark logs
    # below will also show data source.
    _lei(
        benchmark_name=benchmarks_to_run[0],
        model_name=", ".join(models_to_run),
        data_dir="runs/datasets",
    )

    # ── resolve config ──
    if args.quick:
        epochs  = args.epochs  or 3
        n_train = args.n_train or 20
        n_val   = args.n_val   or 6
        n_test  = args.n_test  or 10
        batch   = args.batch   or 4
        log.info("QUICK MODE: epochs=3, n_train=20, ~3 minutes")
    else:
        epochs  = args.epochs  or 100
        n_train = args.n_train or 200
        n_val   = args.n_val   or 50
        n_test  = args.n_test  or 50
        batch   = args.batch   or 8
        log.info("FULL MODE: epochs=100, n_train=200, expect ~1-4h depending on GPU")

    benchmarks = args.benchmark or BENCHMARKS
    models     = args.model     or MODELS

    log.info(f"Benchmarks : {benchmarks}")
    log.info(f"Models     : {models}")
    log.info(f"Epochs     : {epochs}")
    log.info(f"n_train    : {n_train}")

    t_total = time.time()
    failed = []

    for bm in benchmarks:
        log.info(f"\n{'='*60}")
        log.info(f"BENCHMARK: {bm}")
        log.info(f"{'='*60}")

        # ── Step 1: generate ──
        if not args.skip_generate:
            ok = step_generate(
                bm, n_train, n_val, n_test,
                n_particles=args.n_particles,
                n_batches=args.n_batches,
                n_inactive=args.n_inactive,
            )
            if not ok:
                log.error(f"  Dataset generation failed for {bm}; skipping benchmark.")
                failed.append(f"generate:{bm}")
                continue

        for mdl in models:
            log.info(f"\n  ── {bm} / {mdl} ──")

            # ── Step 2: train ──
            if args.skip_train:
                run_name = f"{bm}_{mdl}"
                ckpt_dir = Path("runs") / run_name / "checkpoints"
                ckpt = ckpt_dir / "best.pt"
                if not ckpt.exists():
                    ckpt = ckpt_dir / "latest.pt"
                if not ckpt.exists():
                    log.warning(f"  No checkpoint found for {run_name}; skipping.")
                    failed.append(f"no_ckpt:{bm}:{mdl}")
                    continue
            else:
                ckpt = step_train(bm, mdl, epochs, batch, n_train, n_val, args.seed)
                if ckpt is None:
                    failed.append(f"train:{bm}:{mdl}")
                    continue

            log.info(f"  Checkpoint: {ckpt}")
            extra = BENCH_EVAL_ARGS.get(bm, [])

            # ── Step 3: evaluate ──
            # Test set baseline (all benchmarks) — run first so it's always available
            ok = step_eval(bm, mdl, ckpt, "test_set", n_test, batch, extra)
            if not ok:
                failed.append(f"eval:test_set:{bm}:{mdl}")

            # SN transfer (all benchmarks)
            ok = step_eval(bm, mdl, ckpt, "sn_transfer", n_test, batch, extra)
            if not ok:
                failed.append(f"eval:sn_transfer:{bm}:{mdl}")

            # Resolution transfer (all benchmarks)
            ok = step_eval(bm, mdl, ckpt, "resolution_transfer", n_test, batch, extra)
            if not ok:
                failed.append(f"eval:resolution_transfer:{bm}:{mdl}")

            # Regime sweep (2D benchmarks only)
            if bm not in NO_REGIME_SWEEP:
                ok = step_eval(bm, mdl, ckpt, "regime_sweep", n_test, batch, extra)
                if not ok:
                    failed.append(f"eval:regime_sweep:{bm}:{mdl}")

    # ── Step 4: aggregate ──
    log.info(f"\n{'='*60}")
    log.info("AGGREGATING RESULTS")
    log.info(f"{'='*60}")
    step_aggregate()

    # ── Final summary ──
    elapsed = time.time() - t_total
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    if failed:
        log.warning(f"Failed steps ({len(failed)}):")
        for f in failed:
            log.warning(f"  {f}")
    else:
        log.info("All steps completed successfully.")

    log.info("\nKey output files:")
    log.info("  runs/aggregate/all_results.csv       ← paper table")
    log.info("  runs/eval/*/test_set.csv             ← held-out test performance")
    log.info("  runs/eval/*/sn_transfer.csv          ← angular discretization transfer")
    log.info("  runs/eval/*/resolution_transfer.csv  ← spatial resolution transfer")
    log.info("  runs/eval/*/regime_sweep.csv         ← epsilon sweep (pinte2009 only)")


if __name__ == "__main__":
    main()
