# TALLRec + FairIPS Evaluation Script for Beauty and Yelp
# can replace beauty with sports. Fix the checkpoint_path accordingly

# Beauty

python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-beauty-FAIR-IPS/ --data_path ../data/ --model_name TallRec-FAIR_IPS --dataset beauty --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 3 >> outputs/beauty/beauty-FAIR-IPS-EVAL.txt

# Yelp
python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-yelp-FAIR-IPS/ --data_path ../data/ --model_name TallRec-FAIR_IPS --dataset yelp --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 3 >> outputs/yelp/yelp-FAIR-IPS-EVAL.txt
