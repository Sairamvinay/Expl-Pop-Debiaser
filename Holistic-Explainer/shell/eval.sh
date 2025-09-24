# Holistic Explainer: Expl-Debias evaluation script
# Change dataset name
# Script to evaluate all cases: 
# BPR learning case
python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-0/checkpoint_epoch_BEST.pt --local-rank 3 >> outputs/beauty/stage0-BEAUTY-EVAL-simpleBPR-JUNE30-rerun.txt

# Contrastive explanation learning case
python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-1/checkpoint_epoch_BEST.pt --expl_path generated-expls/ --local-rank 3 --temperature 3.5 --stage 1 --mode 1 >> outputs/beauty/stage1-BPR/EVAL-POSONLY-BEAUTY-JULY23-SIGMOID-temp-3.5.txt

# Optional: Required only for the Re-Rank study

# python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-1/checkpoint_epoch_BEST.pt --expl_path generated-expls/ --local-rank 3 --temperature 3.5 --stage 1 --mode 2 >> outputs/beauty/stage1-BPR/EVAL-NEGONLY-BEAUTY-JULY28-SIGMOID-temp-3.5.txt
# python3 holistic_explainer-evaluate-BPR-simple.py --output_dir top-preds/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/stage-1/checkpoint_epoch_BEST.pt --expl_path generated-expls/ --local-rank 3 --temperature 1.5 --stage 1 --mode 3 >> outputs/beauty/stage1-BPR/EVAL-ZEROONLY-BEAUTY-JULY28-SIGMOID-temp-1.5.txt
