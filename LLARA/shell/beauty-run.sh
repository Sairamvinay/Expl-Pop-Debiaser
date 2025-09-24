# LLaRa Vanilla training script
# Stage 1 Training can be done by independently training TALLRec 1st 
# we can use Rec_ckpt after training Matrix Factorization seperately 
# see MF/ for more details.
# We can re-use TALLRec checkpoints as Stage 1 point ; Update with llama_ckpt
# Use trained MF weights for Stage 1 rec_ckpt also.
# Lora_r and Lora_alpha can be chosen after tuning from TALLRec tuning step (Kept 16 and 16 as a reference)
# update rec_dim after MF checkpoint
# x mark means the developer needs to fill after running HP tuning.
seed=999
output_dir="snap/llama-beauty/" # replace with yelp here
data_path="../data/"
dataset="beauty"
num_workers=4
batch_size=16
num_epochs=5
clip_grad_norm=1.0
maxlen=512
gradient_accumulation_steps=1
num_batches_train=-1
num_batches_val=-1
prompt_path="prompts/llara_amazon.txt"  # replace with yelp here
lora_r=16
lora_alpha=16
lora_dropout=x
lr=x
wd=x
wr=x
rec_dim=x
rec_ckpt="../MF/snap/beauty/checkpoint_epoch_BEST.pt" # replace with yelp here
llama_ckpt="../TALLRec/snap/llama-beauty/" # replace with yelp here
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --early_stopping_patience 3 --distributed --num_batches_train $num_batches_train --num_batches_val $num_batches_val --rec_ckpt $rec_ckpt --llama_ckpt $llama_ckpt --prompt_path $prompt_path