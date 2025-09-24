# Hyperparameter tuning script for Contrastive explanation- style learning; 
# generate explanations using gen-expl-runs.sh

data_path='../data/'
dataset='beauty'
num_batches_train=1760 # 2785 for sports; 2761 for yelp ; 1760 for beauty (10% of train data)
num_batches_val=440 # 696 for sports; 690 for yelp  ;440 for beauty (10%  val data)
batch_size=16
seed=999
epochs=2
clip_grad_norm=1.0
gradient_accumulation_steps=1
stage=1
id_embed_dim=64
expl_path='generated-expls/'
checkpoint='snap/beauty/stage-0/checkpoint_epoch_BEST.pt'

python3 -u hyperparameter-tuning-BPR-contrexpl.py --data_path $data_path --dataset $dataset --seed $seed --batch_size $batch_size --epochs $epochs --clip_grad_norm $clip_grad_norm --gradient_accumulation_steps $gradient_accumulation_steps --num_batches_train $num_batches_train --num_batches_val $num_batches_val --checkpoint $checkpoint --id_embed_dim $id_embed_dim --expl_path $expl_path --local-rank 1  >> outputs/$dataset/stage$stage-BPR/HP-$dataset-holistic-stage-$stage-BPR.txt