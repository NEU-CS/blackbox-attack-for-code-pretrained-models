# 攻击 LLaMA，任务：Authorship Attribution 
* 下载模型文件，huggingface: ljcnju/llamaforattack
* 然后将其重命名为ljcllama-7b-hf，放在LLaMA/Authorship-Attribution文件夹下
* 最低显存要求:24G RTX3090
## 数据集
数据集在dataset/gcjpy中
运行下列命令，得到train.csv,valid.csv
一个csv文件中包含code,label
```
python process.py
```

## Fine-tune CodeBERT
### On Python dataset

运行下列命令
```shell
cd code
python fine-tune.py
```
或者使用fine-tune.ipynb
* 模型保存在ljcoutputdir中,这个模型是LoRA模型，不是完整的LLaMA模型

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

#### 攻击

```shell
source attack.sh
```