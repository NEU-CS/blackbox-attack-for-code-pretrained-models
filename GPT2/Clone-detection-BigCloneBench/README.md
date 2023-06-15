# 攻击 GPT-2，任务：克隆检测
* 模型文件huggingface下载链接:ljcnju/gpt2forattack
* 下载后重命名文件夹为gpt2，放在GPT2文件下面，即GPT2/gpt2

## 数据集
数据集在dataset中
运行下列命令
```shell
cd dataset
python preprocess.py
```
得到train.txt,valid.txt和test.txt
然后运行文件**preprocess.ipynb**得到txt文件对应的pkl文件，fine-tune需要用到这些pkl文件

## Fine-tune CodeBERT
### On Python dataset

使用**fine-tune.ipynb**
* 模型保存在CodeBERTsaved_models中

## Attack

### On Python dataset

#### 得到待替换词汇表
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
cd code
source attack.sh
```
运行此命令进行所有攻击
攻击的log文件中记录了攻击结果
