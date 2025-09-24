data_path='../data/'
dataset='beauty'
num_batches_train=961 # 1407 for sports ; 1597 for yelp ; 961 for beauty (10% of train data)
num_batches_val=1398 # 2225 for sports ; 1902 for yelp ;1398 for beauty (full val data)
batch_size=16
seed=999
epochs=10
clip_grad_norm=1.0
gradient_accumulation_steps=1

python3 -u hyperparameter-tuning-MF.py --data_path $data_path --dataset $dataset --seed $seed --batch_size $batch_size --epochs $epochs --clip_grad_norm $clip_grad_norm --gradient_accumulation_steps $gradient_accumulation_steps --num_batches_train $num_batches_train --num_batches_val $num_batches_val --local-rank 1 >> outputs/$dataset/$dataset-hp-MF-BPR.txt