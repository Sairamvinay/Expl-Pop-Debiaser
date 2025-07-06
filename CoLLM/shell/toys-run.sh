seed=999
output_dir="snap/llama-toys/"
data_path="../data/"
dataset="toys"
num_workers=4
batch_size=16
num_epochs=8
clip_grad_norm=1.0
maxlen=784
gradient_accumulation_steps=1
num_batches_train=-1
num_batches_val=-1
stage=2
llama_ckpt="../TALLRec/snap/llama-toys/"
prompt_path="prompts/collm_amazon.txt"
lora_r=32
lora_alpha=32
lr=0.0006598223112749972
wd=7.176888481758852e-05
wr=0.06893633894428759
rec_dim=128

mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=1,2 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout 0.0 --project_mid 10 --early_stopping_patience 3 --stage $stage --distributed --llama_ckpt $llama_ckpt --num_batches_train $num_batches_train --num_batches_val $num_batches_val --prompt_path $prompt_path