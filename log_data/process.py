import json
import pandas as pd

def format2alpaca(dataset):
    '''
        INSTRUCTION_START=(
                'Below is a system log question and relevant logs are provided.'
                'Write a response that appropriately answer the question. Let us think step by step, just output the final answer.\n\n'
                '### {input}\n\n'
                '### Response: '),
    '''
    formatted_data = []
    with open('/root/llama-recipes/ft_datasets/log_dataset/{}_qa.json'.format(dataset)) as f:
        for line in f.readlines():
            json_dict = json.loads(line)
            new_record = {'conversation': [{
                    'input':  'Question:\n' + str(json_dict['Question']) + '\n###Logs:' + '\n'.join(json_dict['Logs']),
                    'output': str(json_dict['Answer'])
                }]
            }
        
            formatted_data.append(new_record)
            # print(formatted_data)
            # break
        formatted_data = json.dumps(formatted_data, ensure_ascii=False, indent=4)
    with open('./{}_qa_alpaca.json'.format(dataset), 'w') as w:
        w.write(formatted_data)

def formatQA2Alpaca():
    # 读取Excel文件
    df = pd.read_excel('QA_data/spark_results_FINAL_1.xlsx', header=None)
    # print(df)
    # 初始化一个空列表来存储转换后的JSON数据
    json_data = []

    # 遍历每一行并转换为JSON格式
    for index, row in df.iterrows():
        row = row.to_json()
        print(index)
        print(row)
        row = json.loads(row)['0']
        row = json.loads(row)
        conversation = [
            {
                "input": 'Question:\n{}\n###Logs:{}'.format(row['instruction'], row['input']),
                "output": row['response']
            }
        ]
        json_item = {
            "conversation": conversation
        }
        json_data.append(json_item)

    # 将JSON数据写入文件
    with open('spark_qa_alpaca.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

def saveSummary2Alpaca():
    formatted_data = []
    data_path = 'log_data/summary_data/spark.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
    print(len(data))
    for record in data:
        new_record = {
            'conversation': [{
                    'input':  'Question:\n' + str(record['introduction']) + '\n###Logs:\n' + str(record['input']),
                    'output': str(record['output'])
                }]
        }
        formatted_data.append(new_record)
        
    formatted_data = json.dumps(formatted_data, ensure_ascii=False, indent=4)
    with open('log_data/summary_data/spark_summary_alpaca.json', 'w') as w:
        w.write(formatted_data)
        
if __name__ == '__main__':
    # format2alpaca('hdfs_qa_alpaca.json')
#    formatQA2Alpaca()
    saveSummary2Alpaca()