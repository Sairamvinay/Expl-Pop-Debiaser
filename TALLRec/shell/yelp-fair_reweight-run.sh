seed=999
output_dir="snap/llama-yelp-FAIR-IPS/"
base_model="meta-llama/Llama-3.2-1B"
data_path="../data/"
dataset="yelp"
lr=0.0004943471443068414
wd=1.0441258688257609e-06
dropout=0.12861329944820493
wr=0.09423202560160422
num_workers=4
batch_size=16
micro_batch_size=16
num_epochs=8
lora_r=32
lora_alpha=16
clip_grad_norm=1.0
maxlen=512
gradient_accumulation_steps=1
num_batches_train=-1
num_batches_val=-1

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 smaller_llm_IPS.py \
    --base_model $base_model \
    --data_path $data_path \
    --dataset $dataset \
    --output_dir ${output_dir} \
    --batch_size $batch_size \
    --micro_batch_size $micro_batch_size \
    --num_epochs $num_epochs \
    --num_batches_train $num_batches_train \
    --num_batches_val $num_batches_val \
    --learning_rate $lr \
    --weight_decay $wd \
    --maxlen $maxlen \
    --num_workers $num_workers \
    --clip_grad_norm $clip_grad_norm \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_ratio $wr \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $dropout \
    --lora_target_modules "[\"q_proj\",\"v_proj\"]" \
    --train_on_inputs \
    --seed $seed \
    --distributed

# CUDA_VISIBLE_DEVICES=2 python3 -u smaller_llm_IPS.py \
#     --base_model $base_model \
#     --data_path $data_path \
#     --dataset $dataset \
#     --output_dir ${output_dir} \
#     --batch_size $batch_size \
#     --micro_batch_size $micro_batch_size \
#     --num_epochs $num_epochs \
#     --learning_rate $lr \
#     --weight_decay $wd \
#     --maxlen $maxlen \
#     --num_workers $num_workers \
#     --clip_grad_norm $clip_grad_norm \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --warmup_ratio $wr \
#     --lora_r $lora_r \
#     --lora_alpha $lora_alpha \
#     --lora_dropout $dropout \
#     --lora_target_modules "[\"q_proj\",\"v_proj\"]" \
#     --train_on_inputs \
#     --seed $seed \


