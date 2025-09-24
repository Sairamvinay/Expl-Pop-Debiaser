dataset="beauty"
lr=x
wd=x
wr=x
CUDA_VISIBLE_DEVICES=2,3 python3 -u -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 MF-BPR-train.py --seed 999 --num_workers 4 --epochs 50 --batch_size 16 --id_embed_dim 128 --learning_rate $lr --weight_decay $wd --warmup_ratio $wr --data_path ../data/ --dataset $dataset --output_dir snap/ --num_batches_train -1 --num_batches_val -1 --distributed
