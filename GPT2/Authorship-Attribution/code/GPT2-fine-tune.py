# coding = utf-8
from transformers import GPT2ForSequenceClassification,GPT2Tokenizer,GPT2Config
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import random
from transformers import TrainingArguments,Trainer
import numpy as np
import evaluate
import torch
import argparse

data_path = os.path.join("..","dataset","data_folder","processed_gcjpy")
model_name = "../../gpt2"
config = GPT2Config.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name,bos_token = "<|startoftext|>",eos_token = "<|endoftext|>",pad_token = "<|pad|>",cls_token = "<|cls|>",sep_token = "<|sep|>" ,model_max_length = 1024)
model = GPT2ForSequenceClassification.from_pretrained(model_name,num_labels = 66)
print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

def compute_metrics(eval_pred):
    metirc = evaluate.load("accuracy")
    logits , labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    return metirc.compute(predictions=predictions,references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"],padding = True,truncation = True)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size", default = 4 ,type = int, help = "fine-tune train batch size")
    parser.add_argument("--eval_batch_size", default = 8 , type = int, help = "fine-tune eval batch size")
    parser.add_argument("--epoch_nums",default = 20, type = int , help = "fine-tune epoch nums")
    parser.add_argument("--lr",default = 5e-5,type = float , help = "fine-tune learning rate")
    args = parser.parse_args()
    torch.cuda.empty_cache()
    dataset = load_dataset("csv",data_files={"train":os.path.join(data_path,"train.csv"),"test":os.path.join(data_path,"valid.csv")})
    tokenized_dataset = dataset.map(tokenize_function,batched=True)
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    train_args = TrainingArguments(output_dir= "GPT2-CheckPoint",overwrite_output_dir=True,evaluation_strategy="epoch",per_device_train_batch_size=args.train_batch_size,per_device_eval_batch_size=args.eval_batch_size,num_train_epochs=args.epoch_nums,learning_rate=args.lr)
    trainer = Trainer(model = model,args = train_args,train_dataset=small_train_dataset,eval_dataset=small_eval_dataset,compute_metrics=compute_metrics)
    trainer.train()
    model.save_pretrained("GPT2saved_models")