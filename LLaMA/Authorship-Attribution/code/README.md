# Attack LLaMA on Code Authorship Attribution Task

## Dataset

First, you need to download the dataset from [link](https://drive.google.com/file/d/1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe/view?usp=sharing). Then, you need to decompress the `.zip` file to the `dataset/data_folder`. For example:

```
pip install gdown
gdown https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
unzip gcjpy.zip
cd dataset
mkdir data_folder
cd data_folder
mv ../../gcjpy ./
```

Then, you can run the following command to preprocess the datasets:

```
python process.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

## Fine-tune CodeBERT
### On Python dataset

We use full train data for fine-tuning. The training cost is 20 mins on a single 3090 (24G). We use full valid data to evaluate during training.

```shell
cd code
python GPT2-fine-tune.py
```

## Attack

### On Python dataset
```
cd preprocess
python get_substitutes.py \
    --store_path ./data_folder/processed_gcjpy/valid_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./data_folder/processed_gcjpy/valid.txt \
    --block_size 512
```

#### GA-Attack

```shell
cd code
python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack_gcjpy.log

```