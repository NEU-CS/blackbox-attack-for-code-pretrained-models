import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')

import copy
import random
from utils import select_parents, crossover, map_chromesome, mutate, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position
from utils import CodeDataset
import openai
from run_parser import get_identifiers, get_example

GPT_MODELNAME = "ada:ft-softwiser-2023-04-12-15-12-10"#GPT fine_tune后的模型名称
class_num = 66
ljcpredict = openai.Completion.create

def compute_fitness(chromesome, orig_prob ,code):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "python")
    tempresult = ljcpredict(model = GPT_MODELNAME,
                                                 max_tokens = 1,
                                                 logprobs = class_num,
                                                 temperature = 0,
                                                  prompt = temp_code)
    # 计算fitness function
    fitness_value = orig_prob - tempresult['choices'][0]['logprobs']['token_logprobs'][0]
    return fitness_value, tempresult['choices'][0]['text']

def get_importance_score(ljcorginprob,args, example, code, words_list: list, variable_names: list):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_example.append(new_code)

    tempresult = ljcpredict(model = GPT_MODELNAME,
                                                 max_tokens = 1,
                                                 logprobs = class_num,
                                                 temperature = 0,
                                                  prompt = new_example)
    
    orig_prob = [tempresult["choices"][i]["logprobs"]["token_logprobs"][0] for i in range(len(tempresult["choices"]))]
    # predicted label对应的probability

    importance_score = []
    for prob in orig_prob:
        importance_score.append(ljcorginprob- prob)

    return importance_score, replace_token_positions, positions

class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = GPT_MODELNAME
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def greedy_attack(self, example, code, subs):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
            # 先得到tgt_model针对原始Example的预测信息.
        #模拟退火的超参begin Greddy Attack

        additioncode = " <CODESPLIT> "
        predresult = openai.Completion.create(model = self.model_tgt,
                                                 max_tokens = 1,
                                                 logprobs = class_num,
                                                 temperature = 0,
                                                  prompt = code + additioncode)
        orig_label = predresult['choices'][0]["text"] #原本预测的结果
        current_prob = predresult['choices'][0]['logprobs']['token_logprobs'][0]  #原来的最高PROB
        true_label = example["completion"]
        adv_code = ''
        temp_label = None
        identifiers, code_tokens = get_identifiers(code, 'python')
        prog_length = len(code_tokens)
        p = 0.75 #模拟退火的超参
        processed_code = " ".join(code_tokens)
        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..


        variable_names = list(subs.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None


        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(current_prob,self.args, example, 
                                                processed_code,
                                                words,
                                                variable_names,
                                                )

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None


        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code)
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}
        exist_words = {}
        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]

            all_substitues = subs[tgt_word]

            most_gap = 0.0
            candidate = None
            gap_list = []
            for substitute in all_substitues:
                if exist_words.get(substitute,False):
                    #如果当前替换的词语在之前出现过，那么不进行替换，保证语义一致性
                        continue
                # 需要将几个位置都替换成sustitue_
                temp_code = get_example(final_code, tgt_word, substitute, "python")
                
                temp_results_ljc = openai.Completion.create(model=self.model_tgt,
                                                            max_tokens = 1,
                                                            logprobs = class_num,
                                                            temperature = 0,
                                                            prompt=temp_code+additioncode)
                temp_choices = temp_results_ljc["choices"]

                temp_prob = temp_choices[0]["logprobs"]["token_logprobs"][0]
                temp_label = temp_choices[0]["text"]

                if temp_label != orig_label:
                    
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute
                    replaced_words[tgt_word] = candidate
                    exist_words[replaced_words[tgt_word]] = True
                    adv_code = get_example(final_code, tgt_word, candidate, "python")
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                        ('>>', tgt_word, candidate,
                        current_prob,
                        temp_prob), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob
                    gap_list.append((gap,substitute))
                    gap_list = sorted(gap_list,key = lambda x:x[0],reverse=True)
                    most_gap = gap_list[0][0]
                    candidate = gap_list[0][1]
        
            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, "python")
                replaced_words[tgt_word] = candidate
                exist_words[replaced_words[tgt_word]] = True
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_prob + most_gap,
                    current_prob), flush=True)
            else:
                '''
                这里改成模拟退火
                '''
                print("%s NOACC! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_prob + most_gap,
                    current_prob), flush=True)
                temp_p = random.random()       
                if temp_p > p:
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    current_prob = current_prob - most_gap
                    final_code = get_example(final_code, tgt_word, candidate, "python")
                    replaced_words[tgt_word] = candidate
                    exist_words[replaced_words[tgt_word]] = True
                    print("%s SNT! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_prob + most_gap,
                    current_prob), flush=True)
                else:
                    print(">> NOSNT!",flush=True)
                    replaced_words[tgt_word] = tgt_word
                    exist_words[replaced_words[tgt_word]] = True
            adv_code = final_code
            
            # 添加死代码，死代码为变量的定义和print(变量名)的操作
        vocab = list(names_positions_dict.keys())
        for k,v in subs.items():
            vocab += v

        re = [(";"+i+" = 0",";print("+i+")") for i in vocab]
        vocab = []
        for i in re:
            vocab.append(i[0]);vocab.append(i[1])
        for addcode in vocab:
            temp_code = final_code + addcode
            temp_results_ljc = openai.Completion.create(model=self.model_tgt,
                                                            max_tokens = 1,
                                                            logprobs = class_num,
                                                            temperature = 0,
                                                            prompt=temp_code+additioncode)
            temp_choices = temp_results_ljc["choices"]
            temp_prob = temp_choices[0]["logprobs"]["token_logprobs"][0]
            temp_label = temp_choices[0]["text"]

            if temp_label != true_label:
                final_code = temp_code
                adv_code = final_code
                print(">> SUC! INSERT %s"%(addcode),flush=True)
                is_success = 1
                return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
            else:
                nowprob = temp_prob
                if nowprob < current_prob:
                    final_code = temp_code
                    print(">> ACC! INSERT %s %.5f=>%.5f"%(
                        addcode,current_prob,nowprob
                    ),flush=True)
                    current_prob = nowprob
                else:
                    pass
                    '''
                    print(">> NOACC! INSERT %s %.5f=>%.5f CHECK SNT"%(
                        addcode,current_prob,nowprob
                    ),flush=True)
                    temp_p = random.random()
                    if temp_p > p :
                        final_code = temp_code
                        print(">> SNT! INSERT %s %.5f=>%.5f"%(
                            addcode,current_prob,nowprob
                        ),flush=True)
                        current_prob = nowprob
                    else:
                        print(">> NOSNT",flush=True)
                    '''
                    
            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
