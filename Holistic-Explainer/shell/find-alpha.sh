#!/bin/bash

OUTPUT_DIR="outputs/beauty/FindALPHA-ALGO/"
DATASET="beauty"
SEED=999
NUM_WORKERS=4
BATCH_SIZE=100
DATA_PATH="../data/"
POS_PATH="top-preds/stage-1-POS-only-EXPLS/"
NEG_PATH="top-preds/stage-1-NEG-only-EXPLS/"
ZERO_PATH="top-preds/stage-1-ZERO-EXPLS/"
LOCAL_RANK=1
K=10

# ALPHA finding Algo
mkdir -p "$OUTPUT_DIR"

# for tau in $(seq 0 0.05 1); do
for tau in 0.0 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0; do
    # Format tau to two decimal places
    tau_fmt=$(printf "%.2f" $tau)
    # for K in 1 2 3 5 10 20; do
    echo "Running for tau=$tau_fmt, K=$K"
    python3 -u holistic_explainer-evaluate-BPR-ALPHA-weightage.py \
        --output_dir top-preds/ \
        --seed $SEED \
        --num_workers $NUM_WORKERS \
        --batch_size $BATCH_SIZE \
        --data_path $DATA_PATH \
        --dataset $DATASET \
        --pos_path $POS_PATH \
        --neg_path $NEG_PATH \
        --zero_path $ZERO_PATH \
        --local-rank $LOCAL_RANK \
        --top_k $K \
        --tau $tau_fmt \
        >> "$OUTPUT_DIR/EVAL-FindALPHA-ALGO-BEAUTY-JULY29-RERANK-TOP-${K}-TAU-${tau_fmt}.txt"
    # done
done

# Example for one run with Tau=0.4 and K = 10

# python3 -u holistic_explainer-evaluate-BPR-SAI-ranking-ALPHA-weightage.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --data_path ../data/ --dataset beauty --pos_path top-preds/stage-1-POS-only-EXPLS/ --neg_path top-preds/stage-1-NEG-only-EXPLS/ --zero_path top-preds/stage-1-ZERO-EXPLS/ --local-rank 1 --top_k 10 --tau 0.4 --debug >> outputs/beauty/DEBUG-EVAL-FindALPHA-ALGO-BEAUTY-JULY29-RERANK-TOP-10-TAU-0.4.txt