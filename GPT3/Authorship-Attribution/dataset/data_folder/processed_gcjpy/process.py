import random
import json


if __name__ == "__main__":
    validseq = []
    filepath = "valid_prepared"
    with open(filepath+"_subs.jsonl") as f:
        with open(filepath+".jsonl") as f2:
            for v1,v2 in zip(f,f2):
                validseq.append((json.loads(v1),json.loads(v2)))
    
    random.shuffle(validseq)
    validseq = validseq[:50] #gpt只攻击前十个样本
    with open("valid_prepare_subs2.jsonl","w") as f:
        with open("valid_prepare2.jsonl","w") as f2:
            for line in validseq:
                f.write(json.dumps(line[0])+'\n')
                f2.write(json.dumps(line[1]) + '\n')
    

    
