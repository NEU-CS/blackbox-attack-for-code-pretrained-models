# coding=utf-8
# @Time    : 2023/4/11
# @Author  : Liu Jin Cheng
# @Email   : 1364729682@qq.com
# @File    : attacker.py
'''For attacking GPT-3 models for code'''
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()
import json
import logging
import argparse
import warnings
import torch
import time
import openai
from utils import Recorder
from attacker import Attacker
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\

gptmodel = "ada:ft-softwiser-2023-04-12-15-12-10"
print(gptmodel)
ljcpredict = openai.Completion.create
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", type=str,
                        help="Path to store the CSV file")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()


    args.device = torch.device("cuda")
    # Set seed
    random.sample(args.seed)

    print("Begin attack")
    args.start_epoch = 0
    args.start_step = 0

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    codebert_mlm.to('cuda') 

    ## Load Dataset
    eval_dataset = []
    with open(args.eval_data_file,"r") as f:
        for line in f:
            eval_dataset.append(json.loads(line.strip()))

    file_type = args.eval_data_file.split('/')[-1].split('.')[0] # valid
    folder = '/'.join(args.eval_data_file.split('/')[:-1]) # 得到文件目录
    codes_file_path = os.path.join(folder, '{}_subs2.jsonl'.format(
                                file_type))
    source_codes = []
    substs = []
    with open(codes_file_path) as rf:
        for line in rf:
            item = json.loads(line.strip())
            source_codes.append(item["code"].replace("\\n", "\n").replace('\"','"'))
            substs.append(item["substitutes"])
    assert(len(source_codes) == len(eval_dataset) == len(substs))


    success_attack = 0
    total_cnt = 0
    recoder = Recorder(args.csv_store_path)
    attacker = Attacker(args, None, None, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    for index, example in enumerate(eval_dataset):
        example_start_time = time.time()
        code = source_codes[index]
        subs = substs[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(example, code, subs)
        
        attack_type = "Greedy"

        example_end_time = (time.time()-example_start_time)/60
        
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        recoder.write(index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type, 100, example_end_time)
        
        if is_success >= -1 :
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1
        
        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
    
        
if __name__ == '__main__':
    main()
