import openai
import json
import os
gpt_model = "ada:ft-personal-2023-04-11-11-05-03"  #fine_tune后的模型名称，分类准确率能达到92.4%，CodeBert是85%
MAX_TOKENS = 1  #
LOGPROBS = 66 #分类的个数

codes = []

with open("valid_prepared.jsonl","r") as f:
    for line in f:
        codes.append(json.loads(line)["prompt"])

codes = codes[:2]


a = openai.Completion.create(
    model="ada:ft-personal-2023-04-11-11-05-03",
    max_tokens = 1,
    logprobs = 66,
    temperature = 0,
    prompt=codes)


print(a["choices"])
print([a["choices"][i]["logprobs"]["token_logprobs"][0] for i in range(len(a["choices"]))])
# 结果: a["choices"][0]["text"]
# prob: a['choices'][0]['logprobs']['top_logprobs'][0]