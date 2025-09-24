# Holistic Explainer: Expl-Debias training script
# Change dataset name

# Fix lr, wd, wr and dropout accordingly after tuning EACH stage seperately
# Learn and fix id_embed_dim; given 64 for reference; hidden_dim must be 4x,2x,x as python list
# Proceedure (Shared for both cases)
#  1) Run tuning for Stage 0
#  2) Run Stage 0 training (see below)
# 3) Run train and test explanations for each case using the corresponding scripts and reuse them for this study.
# Stage 1 is contrastive Expl training
# FOR THIS Study
# 3) Use Stage 0 checkpoint for Stage 1 tuning 
# 4) Stage 1 tuning (see below)
lr=x
wd=x
wr=x
dropout=x
# DEEPSEEK

CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 holistic_explainer-BPR-simple.py --seed 999 --num_workers 4 --epochs 5 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr --stage 1 --data_path ../data/ --dataset beauty --expl_path generated-expls-deepseek/ --output_dir snap-deepseek/ --num_batches_train -1 --num_batches_val -1 --distributed --temperature 3 --checkpoint snap/beauty/stage-0/checkpoint_epoch_BEST.pt  >> outputs/beauty/DeepSeek-runs/TRAINING-stage1-deepfm-beauty-AUG23.txt

# GPT

CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 holistic_explainer-BPR-simple.py --seed 999 --num_workers 4 --epochs 5 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr --stage 1 --data_path ../data/ --dataset beauty --expl_path generated-expls-chatgpt/ --output_dir snap-chatgpt/ --num_batches_train -1 --num_batches_val -1 --distributed --temperature 3 --checkpoint snap/beauty/stage-0/checkpoint_epoch_BEST.pt  >> outputs/beauty/Chatgpt-runs/TRAINING-stage1-deepfm-beauty-AUG24.txt