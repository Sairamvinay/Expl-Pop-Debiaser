data_path='../data/'
dataset='beauty'
num_batches_train=1760 #  for yelp ;  for clothing  ;  for toys ; 1760 for beauty (10% of train data)
num_batches_val=440 #  for yelp ;  for clothing ;  for toys ;440 for beauty (10%  val data)
batch_size=16
seed=999
epochs=2
clip_grad_norm=1.0
gradient_accumulation_steps=1
stage=1
id_embed_dim=64
expl_path='generated-expls/'
checkpoint='snap/beauty/stage-0/checkpoint_epoch_BEST.pt'

python3 -u hyperparameter-tuning-BPR-stage1.py --data_path $data_path --dataset $dataset --seed $seed --batch_size $batch_size --epochs $epochs --clip_grad_norm $clip_grad_norm --gradient_accumulation_steps $gradient_accumulation_steps --num_batches_train $num_batches_train --num_batches_val $num_batches_val --checkpoint $checkpoint --id_embed_dim $id_embed_dim --expl_path $expl_path --local-rank 2 >> outputs/$dataset/$dataset-hp-holistic-stage-$stage-BPR.txt