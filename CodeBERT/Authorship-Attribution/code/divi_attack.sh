python attack.py \
    --csv_store_path ./attack_replace.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --model_name_or_path=CodeBERTsaved_models \
    --tokenizer_name=CodeBERTsaved_models \
    --eval_batch_size 32 \
    --use_replace\
    --seed 42| tee attack_replace.log

python attack.py \
    --csv_store_path ./attack_insert.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --model_name_or_path=CodeBERTsaved_models \
    --tokenizer_name=CodeBERTsaved_models \
    --eval_batch_size 32 \
    --use_insert\
    --seed 42| tee attack_insert.log


python attack.py \
    --csv_store_path ./attack_ga.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --model_name_or_path=CodeBERTsaved_models \
    --tokenizer_name=CodeBERTsaved_models \
    --eval_batch_size 32 \
    --use_ga\
    --seed 42| tee attack_ga.log