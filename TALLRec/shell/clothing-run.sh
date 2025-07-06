seed=999
output_dir="snap/llama-clothing/"
base_model="meta-llama/Llama-3.2-1B"
data_path="../data/"
dataset="clothing"
lr=0.0004495724828203225 # 0.0004994731649091307
wd=0.005365980337577398 # 0.0020293461045168075
dropout=0.2002621501811689 # 0.1480450121165145
wr=0.05036522263390473 # 0.08211172070674638
num_workers=4
batch_size=16
micro_batch_size=16
num_epochs=7
lora_r=16 # 32
lora_alpha=32
clip_grad_norm=1.0
maxlen=512
gradient_accumulation_steps=1

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=1,2 python3 -u -m torch.distributed.launch --nproc_per_node=2 smaller_llm.py \
    --base_model $base_model \
    --data_path $data_path \
    --dataset $dataset \
    --output_dir ${output_dir} \
    --batch_size $batch_size \
    --micro_batch_size $micro_batch_size \
    --num_epochs $num_epochs \
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

# CUDA_VISIBLE_DEVICES=2 python3 -u smaller_llm.py \
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


