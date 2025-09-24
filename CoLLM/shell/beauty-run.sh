# CoLLM Vanilla training script
# Stage 1 Training can be done by independently training TALLRec 1st 
# We can re-use TALLRec checkpoints as Stage 1 point
# Update with llama_ckpt
# Lora_r and Lora_alpha can be chosen after tuning from TALLRec tuning step (Kept 16 and 16 as a reference)
# x mark means the developer needs to fill after running HP tuning.
seed=999
output_dir="snap/llama-beauty/" # replace with yelp here
data_path="../data/"
dataset="beauty" # replace with yelp here
stage=2
num_workers=4
batch_size=16
num_epochs=10
clip_grad_norm=1.0
maxlen=784
gradient_accumulation_steps=1
num_batches_train=-1
num_batches_val=-1
lora_r=16
lora_alpha=16
lr=x
wd=x
wr=x
rec_dim=x
llama_ckpt="PATH/TO/STAGE1-Checkpoint/"
prompt_path="prompts/collm_amazon.txt" # replace with yelp in place of amazon
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout 0.0 --project_mid 10 --early_stopping_patience 3 --distributed --num_batches_train $num_batches_train --num_batches_val $num_batches_val --stage $stage --llama_ckpt $llama_ckpt --prompt_path $prompt_path