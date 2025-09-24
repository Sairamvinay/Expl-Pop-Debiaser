# BEAUTY

# LLAMA EXPL-GENERATE

# TRAIN: 
python3 expl-generate-train.py --data_path ../data/ --dataset beauty --output_dir generated-expls/ --maxlen 50 --batch_size 64 --seed 999 --local-rank 2  >> outputs/beauty/LLAMA-expl-gen-BEAUTY-FULLRUN-JUNE30.txt

# TEST: 
python3 expl-generate-test.py --dataset beauty --data_path ../data/ --output_dir generated-expls/ --pred_dir ../top-preds/ --model_name TallRec-Clean --maxlen 50 --batch_size 64 --seed 999 --local-rank 0 --log_wandb >> outputs/beauty/LLAMA-expl-gen-BEAUTY-TEST-MAY31.txt

# DEEP-SEEK EXPL-GENERATE

# TRAIN: 
python3 expl-generate-train-deepseek.py --data_path ../data/ --dataset beauty --output_dir generated-expls-deepseek/ --maxlen 50 --batch_size 32 --seed 999 --local-rank 2  >> outputs/beauty/DEEPSEEK-expl-gen-BEAUTY-FULLRUN-AUG1.txt

# TEST: 
python3 expl-generate-test-deepseek.py --dataset beauty --data_path ../data/ --output_dir generated-expls-deepseek/ --pred_dir ../top-preds/ --model_name TallRec-Clean --maxlen 50 --batch_size 75 --seed 999 --local-rank 2 >> outputs/beauty/DeepSeek-runs/DEEPSEEK-expl-gen-BEAUTY-TEST-AUG26.txt

# YELP

# LLAMA EXPL-GENERATE

# TRAIN: 
python3 expl-generate-train.py --data_path ../data/ --dataset yelp --output_dir generated-expls/ --maxlen 50 --batch_size 64 --seed 999 --local-rank 1  >> outputs/yelp/LLAMA-expl-gen-YELP-FULLRUN-JULY5.txt

# TEST: 
python3 expl-generate-test.py --dataset yelp --data_path ../data/ --output_dir generated-expls/ --pred_dir ../top_preds/ --model_name TallRec-Clean --maxlen 50 --batch_size 64 --seed 999 --local-rank 1 --log_wandb >> outputs/yelp/LLAMA-expl-gen-YELP-TEST-MAY31.txt 


# SPORTS

# LLAMA

# TRAIN:
python3 -u expl-generate-train.py --data_path ../data/ --dataset sports --output_dir generated-expls/ --maxlen 50 --batch_size 64 --seed 999 --local-rank 1  >> outputs/sports/LLAMA-expl-gen-SPORTS-FULLRUN-AUG30.txt

# TEST:
python3 -u expl-generate-test.py --data_path ../data/ --dataset sports --output_dir generated-expls/ --pred_dir ../top-preds/ --model_name TallRec-Clean --maxlen 50 --batch_size 64 --seed 999 --local-rank 1  >> outputs/sports/LLAMA-expl-gen-SPORTS-TEST-AUG31.txt

