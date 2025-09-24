# Script to evaluate all encoder study: 


# BERT
# BPR learning case
python3 holistic_explainer-evaluate-BPR-different-encoders.py --output_dir top-preds-bert/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/encoder-bert/stage-0/checkpoint_epoch_BEST.pt  --local-rank 0 --temperature 1 --stage 0 --encoder_type bert >> outputs/beauty/encoders/BERT/STAGE0-BPR-EVAL-AUG23-SIGMOID-temp-1.txt

# # Contrastive explanation learning case
python3 holistic_explainer-evaluate-BPR-different-encoders.py --output_dir top-preds-bert/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/encoder-bert/stage-1/checkpoint_epoch_BEST.pt --expl_path generated-expls/ --local-rank 0 --temperature 1 --stage 1 --mode 1 --encoder_type bert >> outputs/beauty/encoders/BERT/STAGE1-EVAL-POSONLY-BEAUTY-AUG23-SIGMOID-temp-1.txt

# W2V
# BPR learning case
python3 holistic_explainer-evaluate-BPR-different-encoders.py --output_dir top-preds-w2v/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/encoder-w2v/stage-0/checkpoint_epoch_BEST.pt  --local-rank 0 --temperature 1 --stage 0 --encoder_type w2v >> outputs/beauty/encoders/W2V/STAGE0-BPR-EVAL-AUG23-SIGMOID-temp-1.txt

# # Contrastive explanation learning case

python3 holistic_explainer-evaluate-BPR-different-encoders.py --output_dir top-preds-w2v/ --seed 999 --num_workers 4 --batch_size 100 --id_embed_dim 64 --hidden_dim [256,128,64] --dropout 0.0 --data_path ../data/ --dataset beauty --checkpoint snap/beauty/encoder-w2v/stage-1/checkpoint_epoch_BEST.pt --expl_path generated-expls/ --local-rank 0 --temperature 1 --stage 1 --mode 1 --encoder_type w2v >> outputs/beauty/encoders/W2V/STAGE1-EVAL-POSONLY-BEAUTY-AUG23-SIGMOID-temp-1.txt