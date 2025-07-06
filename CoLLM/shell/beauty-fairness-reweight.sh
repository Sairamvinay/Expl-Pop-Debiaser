seed=999
output_dir="snap/llama-beauty-IPS/"
data_path="../data/"
dataset="beauty"
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
lr=0.009552658071975665
wd=2.822486700873216e-07
wr=0.07468962717570075
rec_dim=64
stage=2
llama_ckpt="../TALLRec/snap/llama-beauty/"
group_num=5

mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=1,2 python3 -u -m torch.distributed.launch --nproc_per_node=2 trainer_distr-ips.py --data_path $data_path --dataset $dataset --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --learning_rate $lr --weight_decay $wd --num_workers $num_workers --maxlen $maxlen --clip_grad_norm $clip_grad_norm --warmup_ratio $wr --seed $seed --rec_dim $rec_dim --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout 0.0 --project_mid 10 --early_stopping_patience 3 --distributed --num_batches_train $num_batches_train --num_batches_val $num_batches_val --stage $stage --llama_ckpt $llama_ckpt --group_num $group_num