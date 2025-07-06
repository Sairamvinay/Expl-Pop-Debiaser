# Stage 0
python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-0/checkpoint_epoch_BEST.pt --local-rank 3 >> outputs/beauty/stage0-BEAUTY-EVAL-simpleBPR-JUNE30-rerun.txt

# Stage 1
python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-1/checkpoint_epoch_BEST.pt --stage 1 --expl_path generated-expls/ --local-rank 2 --softmax >> outputs/beauty/stage1-BEAUTY-EVAL-simpleBPR-JUNE30-rerun-SOFTMAX-with-expls.txt
