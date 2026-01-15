#!/bin/bash
# Evaluate all trained models in runs directory
# Usage: bash scripts/eval_all.sh [runs_dir]

RUNS_DIR=${1:-runs}

echo "Evaluating all models in $RUNS_DIR/"
echo "===================================="
echo ""

# Find all best.pt checkpoints
for checkpoint in "$RUNS_DIR"/*/checkpoints/best.pt; do
    if [ -f "$checkpoint" ]; then
        run_name=$(basename $(dirname $(dirname "$checkpoint")))
        echo "Evaluating: $run_name"
        python scripts/evaluate.py \
            --checkpoint "$checkpoint" \
            --output "$RUNS_DIR/$run_name/eval_results.json"
        echo ""
    fi
done

echo "===================================="
echo "Evaluation complete!"
echo ""
echo "To generate summary:"
echo "  python scripts/summarize.py --runs-dir $RUNS_DIR"
