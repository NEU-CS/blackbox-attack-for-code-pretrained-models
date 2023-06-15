python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_gcjpy.log

mv attack_gcjpy.log attack_replace.log
mv attack_gi.csv attack_replace.csv


python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_insert \
    --seed 42| tee attack_gcjpy.log

mv attack_gcjpy.log attack_insert.log
mv attack_gi.csv attack_insert.csv

python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_ga \
    --seed 42| tee attack_gcjpy.log

mv attack_gcjpy.log attack_ga.log
mv attack_gi.csv attack_ga.csv


