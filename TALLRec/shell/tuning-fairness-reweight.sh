# Hyperparameter tuning for TALLRec + FairIPS
# Replace dataset
# Refer the num_batches_ parameters when changing dataset.
dataset='beauty'
data_path='../data/'
base_model='meta-llama/Llama-3.2-1B'
num_epochs=2
local_rank=1
seed=999
batch_size=16
clip_grad_norm=1.0
maxlen=512
num_batches_train=700 # 951 for yelp  ; 1113 for sports  ; 700 for beauty (10% of train data)
num_batches_val=140 # 190 for yelp ; 223 for sports ;  140 for beauty (5% of val data)


python3 -u hyperparam_tuning-smaller.py --data_path $data_path --dataset $dataset --base_model $base_model --batch_size $batch_size --num_epochs $num_epochs --maxlen $maxlen --clip_grad_norm $clip_grad_norm --seed $seed --local_rank $local_rank --num_batches_train $num_batches_train --num_batches_val $num_batches_val --train_on_inputs --fair_reweight --group_num 5 >> outputs/$dataset/$dataset-hp-FAIR-IPS-smaller.txt