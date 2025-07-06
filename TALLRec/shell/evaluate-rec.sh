python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-clothing-FAIR-IPS/ --data_path ../data/ --model_name TallRec-FAIR_IPS --dataset clothing --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 3 >> outputs/clothing/clothing-FAIR-IPS-EVAL.txt

# python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-yelp-FAIR-IPS/ --data_path ../data/ --model_name TallRec-FAIR_IPS --dataset yelp --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 3 >> outputs/yelp/yelp-FAIR-IPS-EVAL.txt

# python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-beauty/ --data_path ../data/ --dataset beauty --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 2 >> outputs/beauty-EVAL.txt

# python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-beauty-masked/ --data_path ../data/ --dataset beauty --batch_size 20 --maxlen 1 --seed 999 --num_workers 0 --gpu 2 >> outputs/beauty-EVAL-lossmasked.txt

# python3 evaluate-rec.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-clothing/ --data_path ../data/ --dataset clothing --batch_size 20 --maxlen 5 --seed 999 --num_workers 0 --gpu 1 >> outputs/clothing-EVAL.txt