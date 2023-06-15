python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/test.pkl \
    --model_name_or_path=CodeBERTADV_training_saved_models \
    --tokenizer_name=CodeBERTADV_training_saved_models \
    --eval_batch_size 32 \
    --use_replace \
    --use_insert \
    --use_ga \
    --seed 42| tee attack_gcjpy.log
mv attack_gcjpy.log attack_advtraining.log
mv attack_gi.csv attack_advtraining.csv
