python attack.py \
    --csv_store_path ./attack_all.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --model_name_or_path=CodeBERTsaved_models \
    --tokenizer_name=CodeBERTsaved_models \
    --eval_batch_size 32 \
    --use_ga \
    --use_replace \
    --use_insert \
    --seed 42| tee attack_all.log


