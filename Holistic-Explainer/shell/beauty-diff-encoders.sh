# Holistic Explainer: Expl-Debias training script
# Change dataset name
# Fix lr, wd, wr and dropout accordingly after tuning EACH stage seperately
# Learn and fix id_embed_dim; given 64 for reference; hidden_dim must be 4x,2x,x as python list
# Stage 0 is vanilla BPR training
# Stage 1 is contrastive Expl training
# Proceedure (Same for either case)
#  1) Run tuning for Stage 0
#  2) Run Stage 0 training (see below)
# 3) Use Stage 0 checkpoint for Stage 1 tuning 
# 4) Stage 1 tuning (see below)

lr=x
wd=x
wr=x
dropout=x

# BERT
#Stage 0: 
CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 holistic_explainer-BPR-different-encoders.py --seed 999 --num_workers 4 --epochs 50 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr --stage 0 --data_path ../data/ --dataset beauty --output_dir snap/ --encoder_type bert --num_batches_train -1 --num_batches_val -1 --distributed >> outputs/beauty/encoders/BERT/TRAINING-stage0-deepfm-beauty-AUG20.txt
# Stage 1: 
CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 holistic_explainer-BPR-different-encoders.py --seed 999 --num_workers 4 --epochs 5 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr  --stage 1 --data_path ../data/ --dataset beauty --expl_path generated-expls/ --output_dir snap/ --num_batches_train -1 --num_batches_val -1 --distributed --temperature 3 --encoder_type bert --checkpoint snap/beauty/encoder-bert/stage-0/checkpoint_epoch_BEST.pt  >> outputs/beauty/encoders/BERT/TRAINING-stage1-deepfm-beauty-AUG21.txt

# Word2Vec
# Stage 0
CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 holistic_explainer-BPR-different-encoders.py --seed 999 --num_workers 4 --epochs 50 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr --stage 0 --data_path ../data/ --dataset beauty --output_dir snap/ --encoder_type w2v --num_batches_train -1 --num_batches_val -1 --distributed >> outputs/beauty/encoders/W2V/TRAINING-stage0-deepfm-beauty-AUG20.txt
# Stage 1: 
CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=13567 holistic_explainer-BPR-different-encoders.py --seed 999 --num_workers 4 --epochs 5 --batch_size 16 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout $dropout --learning_rate $lr --weight_decay $wd --warmup_ratio $wr  --stage 1 --data_path ../data/ --dataset beauty --expl_path generated-expls/ --output_dir snap/ --num_batches_train -1 --num_batches_val -1 --distributed --temperature 3 --encoder_type w2v --checkpoint snap/beauty/encoder-w2v/stage-0/checkpoint_epoch_BEST.pt  >> outputs/beauty/encoders/W2V/TRAINING-stage1-deepfm-beauty-AUG22.txt