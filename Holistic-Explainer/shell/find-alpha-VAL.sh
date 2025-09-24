# Step 2: Re-rank TAU Study
#!/bin/bash

OUTPUT_DIR="outputs/beauty/FindALPHA-withVAL-ALGO/"
DATASET="beauty"
SEED=999
NUM_WORKERS=4
BATCH_SIZE=100
DATA_PATH="../data/"
POS_PATH="top-preds/$DATASET/stage-1-POS-only-EXPLS/"
NEG_PATH="top-preds/$DATASET/stage-1-NEG-only-EXPLS/"
ZERO_PATH="top-preds/$DATASET/stage-1-ZERO-EXPLS/"
EXPL_PATH="generated-expls/"
CHECKPOINT_PATH="snap/$DATASET/stage-1/checkpoint_epoch_BEST.pt"
ID_DIM=64
POS_TEMP=3.5
NEG_TEMP=3.5
ZERO_TEMP=1.5
LOCAL_RANK=1
K=10
NUM_BATCHES_VAL=100

# ALPHA finding Algo
mkdir -p "$OUTPUT_DIR"

# for tau in $(seq 0 0.05 1); do
for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    # Format tau to two decimal places
    tau_fmt=$(printf "%.2f" $tau)
    # for K in 1 2 3 5 10 20; do
    echo "Running for tau=$tau_fmt, K=$K"
    python3 -u holistic_explainer-build-val-apply-alpha.py \
        --output_dir top-preds-sample-rerank/ \
        --seed $SEED \
        --num_workers $NUM_WORKERS \
        --batch_size $BATCH_SIZE \
        --data_path $DATA_PATH \
        --dataset $DATASET \
        --pos_path $POS_PATH \
        --neg_path $NEG_PATH \
        --zero_path $ZERO_PATH \
        --expl_path $EXPL_PATH \
        --checkpoint $CHECKPOINT_PATH \
        --id_embed_dim $ID_DIM \
        --hidden_dim [256,128,64] \
        --dropout 0.0 \
        --pos_temperature $POS_TEMP \
        --neg_temperature $NEG_TEMP \
        --zero_temperature $ZERO_TEMP \
        --local-rank $LOCAL_RANK \
        --top_k $K \
        --tau $tau_fmt >> "$OUTPUT_DIR/EVAL-FindALPHA-ALGO-BEAUTY-AUG26-RERANK-TOP-${K}-TAU-${tau_fmt}.txt"
    # done
done

