dataset='beauty'
data_path='../data/'
gradient_accumulation_steps=1
num_epochs=2
seed=999
batch_size=16
clip_grad_norm=1.0
maxlen=784
num_batches_train=700 # 951 for yelp ; 1230 for clothing  ; 600 for toys ; 700 for beauty (10% of train data)
num_batches_val=140 # 190 for yelp ; 246 for clothing ; 120 for toys ; 140 for beauty (5% of val data)
llama_ckpt='../TALLRec/snap/llama-beauty/'
lora_r=16 # 16 for beauty ; 16 for clothing ; 32 for toys ; 32 for yelp
lora_alpha=16 # 16 for beauty ; 32 for clothing ; 32 for toys ; 16 for yelp
local_rank=3

python3 -u hyperparameter_tuning.py --data_path $data_path --dataset $dataset --batch_size $batch_size --num_epochs $num_epochs --maxlen $maxlen --clip_grad_norm $clip_grad_norm --seed $seed --local-rank $local_rank --project_mid 10  --lora_dropout 0.0 --gradient_accumulation_steps $gradient_accumulation_steps --llama_ckpt $llama_ckpt --num_batches_train $num_batches_train --num_batches_val $num_batches_val  --lora_r $lora_r --lora_alpha $lora_alpha >> outputs/$dataset/$dataset-hp-collm.txt