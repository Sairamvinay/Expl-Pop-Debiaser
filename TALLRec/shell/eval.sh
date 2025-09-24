# TALLRec + FairIPS Evaluation Script for Beauty and Yelp
# can replace beauty with sports. Fix the checkpoint_path accordingly

# Beauty

python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-beauty/ --data_path ../data/ --dataset beauty --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 2 >> outputs/beauty-EVAL.txt

# Yelp

python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-yelp/ --data_path ../data/ --dataset yelp --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 2 >> outputs/yelp-EVAL.txt
