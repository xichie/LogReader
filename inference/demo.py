from vllm import LLM, SamplingParams, model_executor
import ray
import json
import evaluate_metric
import time


template = "Below is a system log question and relevant logs are provided. Write a response that appropriately answer the question. Let us think step by step, just output the final answer.\n\n### {}\n\n### Response: "
# template = "Below is a system log question and relevant logs are provided. Write a response that appropriately answer the question. Please consider the log summary. Let us think step by step, just output the final answer.\n\n### {}\n\n### Response: "
# template = "Below is a system log question and relevant logs are provided. Write a response that appropriately answer the question. You should think by following: Parsing logs, Summary logs, and Indentify the anomalies.\n\n### {}\n\n### Response: "
# template = "In a manufacturing plant, there is a need to monitor the production line's system logs to detect anomalies that might lead to equipment failures. Outline the data preprocessing steps and machine learning algorithms you would use to build an effective anomaly detection system for this industrial setting. \n\n### {} \n\n### Response: "
# qa_summary_template = "Below is a system log question and relevant logs are provided. Write a response that appropriately answer the question. \n\n### {} \n First summary these logs, and then output your determine. Note Just output your determination: yes or no! Additional output is unnecessary.\n\n### Response: "

data_path = 'log_data/QA_data/spark_qa_alpaca.json'
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
    # if len(p) < 2048:   # Note: QWen < 1024
    prompts.append(p[:2048])
    q = record['conversation'][0]['input'].split('\n')[1]
    questions.append(q)
    answers.append(record['conversation'][0]['output'])

print(len(answers))

sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=200, use_beam_search=False)
############################################# InternLM #############################################
# llm = LLM(model="/root/.cache/huggingface/hub/models--internlm--internlm-chat-7b/snapshots/ed5e35564ac836710817c51e8e8d0a5d4ff03102",trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/internlm_chat_7b_qlora_log_e3/internlm-chat-7b-ft_log_20.pth/",trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/internlm_chat_7b_qlora_summary/internlm-chat-7b-ft_summary_20.pth/",trust_remote_code=True, tensor_parallel_size=4)
############################################# LLaMA #############################################
# llm = LLM(model="/root/xtuner/work_dirs/llama2_7b_qlora_ad/llama2-chat-7b-ft_log_20.pth/",trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/", trust_remote_code=True, tensor_parallel_size=4, tokenizer='hf-internal-testing/llama-tokenizer')
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_logQA/llama2-chat-7b-ft_log_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_summary/llama2-7b-ft_summary_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_qa_summary/llama2-chat-7b-ft_qa_summary_e20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_qa_ad/llama2_7b_qlora_qa_ad_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_ad_summary/llama2_7b_qlora_ad_summary_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_all_tasks/llama2_7b_qlora_all_tasks_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/llama2_7b_qlora_all_qa/llama2_7b_qlora_all_qa_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
############################################# Qwen #############################################
# llm = LLM(model="/root/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/4e65fbc0cb32849a05be16d5ddcaee50086b1807/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/qwen_7b_qlora_logQA/qwen-chat-7b-ft_log_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/qwen_7b_qlora_summary/qwen-chat-7b-ft_summary_20.pth", trust_remote_code=True, tensor_parallel_size=4)
############################################# ChatGLM2 #############################################
# llm = LLM(model="/root/xtuner/work_dirs/chatglm_6b_qlora_log/chatglm2-6b-ft_log_20.pth/", trust_remote_code=True, tensor_parallel_size=1)
# llm = LLM(model="/mnt/qjx/chatglm_6b_qlora_ad/chatglm-6b-ft_ad_20.pth/", trust_remote_code=True, tensor_parallel_size=1)
# llm = LLM(model="/mnt/qjx/chatglm2-6b/", trust_remote_code=True, tensor_parallel_size=1)
# llm = LLM(model="/mnt/qjx/chatglm_6b_qlora_summary/chatglm-6b-ft_summary_20.pth/", trust_remote_code=True, tensor_parallel_size=1)
# llm = LLM(model="/root/xtuner/chatglm2-6b-ft_log_20.pth/", trust_remote_code=True, tensor_parallel_size=1)
############################################# Baichuan #############################################
# llm = LLM(model="/mnt/qjx/baichuan_7b_qlora_ad/baichuan_7b_qlora_ad_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/Baichuan-7B/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/baichuan_7b_qlora_logQA/baichuan-chat-7b-ft_log_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
# llm = LLM(model="/mnt/qjx/Baichuan-7B/", trust_remote_code=True, tensor_parallel_size=4)
llm = LLM(model="/mnt/qjx/baichuan_7b_qlora_summary/baichuan-chat-7b-ft_summary_20.pth/", trust_remote_code=True, tensor_parallel_size=4)
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

print('Inference time: {}s'.format((end - start) / len(answers)))

model_name = 'llama2_7b_all_tasks_qa_hdfs'
# 指定要保存的JSON文件名
file_name = "qa_results_{}-qa.json".format(model_name)

predictions = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text   
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    predictions.append(generated_text)
    
results = {
            "Questions": questions,
            "predictions": predictions,
            "answers": answers,
        }

# 使用json.dump()将字典保存到JSON文件中
# with open(file_name, "w") as json_file:
#     json.dump(results, json_file)

print('######################################Evaluate############################################')
# BLUE
bleu_score = evaluate_metric.bleu(predictions, answers)
print('BLUE:', bleu_score)
# ROUGE
rouge_score = evaluate_metric.rouge(predictions, answers)
print('ROUGE:', rouge_score)
