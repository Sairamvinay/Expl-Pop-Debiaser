seed=999
output_dir="snap/llama-beauty/"
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
prompt_path="prompts/llara_amazon.txt"
lora_r=16
lora_alpha=16
lora_dropout=0.1816147276897725
lr=0.0002772015149297891
wd=3.792059695219474e-06
wr=0.018243991243987158
rec_dim=128
rec_ckpt="../MF/snap/beauty/checkpoint_epoch_BEST.pt"
llama_ckpt="../TALLRec/snap/llama-beauty/"
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --early_stopping_patience 3 --distributed --num_batches_train $num_batches_train --num_batches_val $num_batches_val --rec_ckpt $rec_ckpt --llama_ckpt $llama_ckpt --prompt_path $prompt_path