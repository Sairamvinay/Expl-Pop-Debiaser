#!/bin/bash
START=0
TOTAL=325047
BATCH_SIZE=4000
PYTHON_SCRIPT="expl-generate-train-chatgpt-host.py"
DATA_PATH="../data/"
DATASET="beauty"
OUTPUT_DIR="generated-expls-chatgpt/"
LOG_DIR="outputs/beauty/Chatgpt-runs/hosting/"
MAXLEN=30
SEED=999
TEMPERATURE=0.8
FREQ_PENALTY=0.5

for ((start=$START; start<$TOTAL; start+=$BATCH_SIZE)); do
    end=$((start + BATCH_SIZE))
    if [ $end -gt $TOTAL ]; then
        end=$TOTAL
    fi
    # Output file for this chunk
    OUT_FILE="${LOG_DIR}/results-${start}-${end}.jsonl"

    # Skip if already done
    if [ -f "$OUT_FILE" ]; then
        echo "Chunk $start-$end already exists, skipping."
        continue
    fi

    echo "Running batch $start to $end..."
    python3 $PYTHON_SCRIPT \
        --data_path "$DATA_PATH" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --maxlen $MAXLEN \
        --seed $SEED \
        --start $start \
        --end $end \
        --temperature $TEMPERATURE \
        --frequency_penalty $FREQ_PENALTY

    echo "Sleeping 60m before next batch..."
    sleep 3600
done

echo "All batches launched or already complete!"
