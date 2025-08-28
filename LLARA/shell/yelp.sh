seed=999
output_dir="snap/llama-yelp/"
data_path="../data/"
dataset="yelp"
num_workers=4
batch_size=16
num_epochs=5
clip_grad_norm=1.0
maxlen=784
gradient_accumulation_steps=1
num_batches_train=-1
num_batches_val=-1
prompt_path="prompts/llara_yelp.txt"
lora_r=32
lora_alpha=16
lora_dropout=0.25660847972154754
lr=0.00047875486658878523
wd=0.002518288357547484
wr=0.07723530000978646
rec_dim=128
rec_ckpt="../MF/snap/yelp/checkpoint_epoch_BEST.pt"
llama_ckpt="../TALLRec/snap/llama-yelp/"

mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --early_stopping_patience 3 --distributed --num_batches_train $num_batches_train --num_batches_val $num_batches_val --rec_ckpt $rec_ckpt --llama_ckpt $llama_ckpt --prompt_path $prompt_path