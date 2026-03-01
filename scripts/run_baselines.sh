#!/usr/bin/env bash
# =============================================================================
# run_baselines.sh - Run smoke train+eval for all benchmarks and models
#
# Usage:
#   bash scripts/run_baselines.sh [--quick] [--benchmark mock_c5g7] [--model all]
#
# Flags:
#   --quick      Use tiny datasets and 5 epochs (for CI/smoke testing)
#   --benchmark  Specific benchmark (default: c5g7 pinte2009)
#   --model      Specific model (default: all 3)
# =============================================================================

set -e  # exit on error

QUICK=0
BENCHMARKS="c5g7 pinte2009"
MODELS="fno deeponet ap_micromacro"
RESULTS_DIR="runs/aggregate"
EPOCHS=20
N_SAMPLES=100

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=1; EPOCHS=3; N_SAMPLES=20; shift ;;
        --benchmark) BENCHMARKS="$2"; shift 2 ;;
        --model) MODELS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
AGGREGATE_CSV="$RESULTS_DIR/aggregate.csv"

# Write CSV header
echo "benchmark,model,protocol,metric,value,timestamp" > "$AGGREGATE_CSV"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

append_csv() {
    # append_csv benchmark model protocol results_json
    local bm="$1" model="$2" proto="$3" results="$4" ts
    ts=$(date '+%Y-%m-%dT%H:%M:%S')
    echo "$bm,$model,$proto,$results,$ts" >> "$AGGREGATE_CSV"
}

for BENCHMARK in $BENCHMARKS; do
    log "===== Benchmark: $BENCHMARK ====="

    # --- Generate dataset (auto: real solver if available, mock fallback) ---
    log "  Generating dataset (solver=auto)..."
    python scripts/generate_dataset.py \
        --benchmark "$BENCHMARK" \
        --solver auto \
        --n_samples "$N_SAMPLES" \
        --split train \
        --output_dir runs/datasets \
        --seed 42

    python scripts/generate_dataset.py \
        --benchmark "$BENCHMARK" \
        --solver auto \
        --n_samples "$((N_SAMPLES / 4))" \
        --split val \
        --output_dir runs/datasets \
        --seed 123

    python scripts/generate_dataset.py \
        --benchmark "$BENCHMARK" \
        --solver auto \
        --n_samples "$((N_SAMPLES / 4))" \
        --split test \
        --output_dir runs/datasets \
        --seed 456

    log "  Dataset generated."

    for MODEL in $MODELS; do
        log "  ----- Model: $MODEL -----"
        RUN_NAME="${BENCHMARK}_${MODEL}"
        CKPT_DIR="runs/${RUN_NAME}/checkpoints"

        # --- Smoke training (loads real data from disk if available) ---
        log "    Training..."
        python train.py \
            --benchmark "$BENCHMARK" \
            --model "$MODEL" \
            --n_epochs "$EPOCHS" \
            --batch_size 4 \
            --run_name "$RUN_NAME" \
            --seed 42 \
            --log_dir runs \
            --data_dir runs/datasets \
            || { log "    FAILED: $MODEL training on $BENCHMARK"; continue; }

        CKPT="${CKPT_DIR}/best.pt"
        if [ ! -f "$CKPT" ]; then
            CKPT="${CKPT_DIR}/latest.pt"
        fi

        if [ ! -f "$CKPT" ]; then
            log "    No checkpoint found; skipping eval."
            continue
        fi

        # --- SN Transfer eval ---
        log "    SN transfer eval..."
        python eval.py \
            --checkpoint "$CKPT" \
            --benchmark "$BENCHMARK" \
            --model "$MODEL" \
            --protocol sn_transfer \
            --output_dir "runs/eval/${RUN_NAME}" \
            --data_dir runs/datasets \
            --seed 42 \
            && append_csv "$BENCHMARK" "$MODEL" "sn_transfer" "see runs/eval/${RUN_NAME}/sn_transfer.csv"

        # --- Resolution transfer eval ---
        log "    Resolution transfer eval..."
        python eval.py \
            --checkpoint "$CKPT" \
            --benchmark "$BENCHMARK" \
            --model "$MODEL" \
            --protocol resolution_transfer \
            --output_dir "runs/eval/${RUN_NAME}" \
            --data_dir runs/datasets \
            --seed 42 \
            && append_csv "$BENCHMARK" "$MODEL" "resolution_transfer" "see runs/eval/${RUN_NAME}/resolution_transfer.csv"

        # --- Regime sweep eval (pinte2009 only; c5g7 is skipped by eval.py) ---
        log "    Regime sweep eval..."
        python eval.py \
            --checkpoint "$CKPT" \
            --benchmark "$BENCHMARK" \
            --model "$MODEL" \
            --protocol regime_sweep \
            --output_dir "runs/eval/${RUN_NAME}" \
            --data_dir runs/datasets \
            --seed 42 \
            && append_csv "$BENCHMARK" "$MODEL" "regime_sweep" "see runs/eval/${RUN_NAME}/regime_sweep.csv"

        log "    Done: $MODEL on $BENCHMARK"
    done
done

log "===== All baselines complete ====="
log "Aggregate results: $AGGREGATE_CSV"

# Collect all individual CSVs into one aggregate
python - <<'PYEOF'
import csv, os, glob
from pathlib import Path

agg_rows = []
for csv_path in glob.glob("runs/eval/**/*.csv", recursive=True):
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source_file"] = csv_path
                agg_rows.append(row)
    except Exception:
        pass

if agg_rows:
    out_path = "runs/aggregate/aggregate.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set(k for r in agg_rows for k in r.keys()))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in agg_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Aggregate CSV saved: {out_path} ({len(agg_rows)} rows)")
else:
    print("No CSV results found to aggregate.")
PYEOF
