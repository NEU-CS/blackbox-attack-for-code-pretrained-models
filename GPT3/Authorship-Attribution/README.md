# Attack GPT on Code Authorship Attribution Task

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

For GPT models, We only attack it on Authorship-Attribution task beacause  the query price of LLM is too large.

## Fine-tune GPT-3
We use OpenAi Apis to fine_tunes GPT-3
You can refer to this document https://platform.openai.com/docs/guides/fine-tuning

### Prepare data
We should prepare the data as the same format of the document,that is:
```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```
This file must be a jsonl file.
You can use openai's api to transform your file to this format jsonl file.
Run this in your shell
```
openai tools fine_tunes.prepare_data -f <your_file_name>
```
Your file can be a csv,xlsx...

We define the split symbol is 
```
<CODESPLIT>
```
We add it in every code which will be transformed.

### Fine-tune
After prepare the training file, run this to fine-tune a GPT-3 model
```
openai api fine_tunes.create -t <training_file_name> [-m <model_name>] [-v<valid_file_name>] [--compute_classification_metrics] [--classification_n_classes <CALSS_NUMBER>] [--suffix "model name"]
```
Study the meaning of the paramter: https://platform.openai.com/docs/guides/fine-tuning/advanced-usage

#### Attack

```shell
cd code
python attack.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid_prepared.jsonl \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack_gcjpy.log
```