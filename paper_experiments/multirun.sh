#!/bin/bash

ALG_ARRAY=("expo")
ENV_ARRAY=("adroit-pen")
NUM_SEEDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Multi-run Configuration:"
echo "  Algorithms: ${ALG_ARRAY[@]}"
echo "  Environments: ${ENV_ARRAY[@]}"
echo "  Number of seeds: $NUM_SEEDS"
echo "=========================================="
echo ""

TOTAL_JOBS=0

for alg in "${ALG_ARRAY[@]}"; do
    for env in "${ENV_ARRAY[@]}"; do
        SCRIPT_PATH="$SCRIPT_DIR/$alg/$env.sh"
        
        if [ ! -f "$SCRIPT_PATH" ]; then
            echo "Warning: Script not found: $SCRIPT_PATH"
            echo "  Skipping $alg/$env combination"
            continue
        fi
        
        echo "Submitting jobs for: $alg/$env"
        
        # Change to the algorithm directory before submitting
        ALG_DIR="$SCRIPT_DIR/$alg"
        cd "$ALG_DIR"
        
        for ((seed=0; seed<$NUM_SEEDS; seed++)); do
            echo "  Submitting seed $seed..."
            sbatch "$env.sh" "$seed"
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            sleep 1
        done
        
        echo ""
    done
done

echo "=========================================="
echo "Total jobs submitted: $TOTAL_JOBS"
echo "=========================================="

