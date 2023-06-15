# 不要模拟退火，不要插入攻击
python attack.py \
    --csv_store_path ./attack_adv-no_simulate_no_insert.csv \
    --eval_data_file=../preprocess/dataset/test.pkl \
    --model_name_or_path=GPT2saved_models \
    --tokenizer_name=GPT2saved_models \
    --eval_batch_size 32 \
    --use_replace \
    --use_ga \
    --p 1 \
    --seed 42| tee attack_adv-no_simulate_no_insert.log
#要模拟退火，不要插入攻击
python attack.py \
    --csv_store_path ./attack_adv-no_insert.csv \
    --eval_data_file=../preprocess/dataset/test.pkl \
    --model_name_or_path=GPT2saved_models \
    --tokenizer_name=GPT2saved_models \
    --eval_batch_size 32 \
    --use_replace \
    --use_ga \
    --p 0.75 \
    --seed 42| tee attack_adv-no_insert.log
