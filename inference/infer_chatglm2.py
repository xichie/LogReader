from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import json
import evaluate_metric
from transformers import AutoConfig
import torch
import os

from utils import load_model_on_gpus


template = "Below is a system log question and relevant logs are provided. Write a response that appropriately answer the question. Let us think step by step, just output the final answer.\n\n### {}\n\n### Response: "

data_path = '/root/log_data/QA_data/log_qa_alpaca.json'
with open(data_path, 'r') as file:
    data = json.load(file)
print(len(data))

questions = []
prompts = []
answers = []
for record in data:
    # print(record['conversation'][0]['input'])

    input_str = record['conversation'][0]['input']
    p = template.format(input_str)
    if len(p) < 2048:   # Note: QWen < 1024
        prompts.append(p)
        q = record['conversation'][0]['input'].split('\n')[1]
        questions.append(q)
        answers.append(record['conversation'][0]['output'])

tokenizer = AutoTokenizer.from_pretrained("/root/xtuner/work_dirs/chatglm_6b_qlora_log/chatglm2-6b-ft_log_20.pth/", trust_remote_code=True)
# model = AutoModel.from_pretrained("/mnt/qjx/chatglm2-6b/", trust_remote_code=True, ).half().cuda()
# # model = load_model_on_gpus("/root/xtuner/work_dirs/chatglm_6b_qlora_log/chatglm2-6b-ft_log_20.pth/", num_gpus=4)
# model = model.eval()

model_path = "/root/xtuner/work_dirs/chatglm_6b_qlora_log/chatglm2-6b-ft_log_20.pth/"

# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# prefix_state_dict = torch.load(os.path.join(model_path, "pytorch_model-00001-of-00003.bin"))
# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
# model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

'''

# 指定要保存的JSON文件名
file_name = "qa_results_chatglm2-6b.json"

predictions = []

# for prompt in prompts:
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

predictions.append(response)
    
results = {
            "Questions": questions,
            "predictions": predictions,
            "answers": answers,
        }
# # 使用json.dump()将字典保存到JSON文件中
# with open(file_name, "w") as json_file:
#     json.dump(results, json_file)
    
print('######################################Evaluate############################################')
# BLUE
bleu_score = evaluate_metric.bleu(predictions, answers)
print('BLUE:', bleu_score)
# ROUGE
rouge_score = evaluate_metric.rouge(predictions, answers)
print('ROUGE:', rouge_score)
'''