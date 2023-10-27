'''
    分析LLM的返回结果
'''
import json
import re

# 0:197 BGL  197:397 spirit 
def load_result():
    '''
        加载LLM的结果
    '''
    with open('/root/xtuner/qa_results_qwen_7b_spirit-w10-ad.json', 'r') as file:
        data = json.load(file)
        
    print(len(data['predictions']))
    print(len(data['answers']))
    return data
    
def calculate_topN_acc(data):
    '''
        计算TopN的准确率
    '''
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    label = data['answers']
    print(data['predictions'][0].strip().split()[0])
    pred_label = list(map(lambda x: x.lower().replace(',', ' ').replace('.', ' ').replace('<', ' ').strip().split()[0], data['predictions']))
    print(pred_label)
    # print(label)
    for i, pred in enumerate(pred_label):
        if pred == 'no':
            pred = 'no'
        else:
            pred = 'yes'
        if label[i] == pred:
            if label[i] == 'yes':
                TP += 1
            else:
                TN += 1
        else:
            if label[i] == 'yes':
                FN += 1
            else:
                FP += 1
    print("TN: {}, TP: {}, FN: {}, FP: {}".format(TN, TP, FN, FP))
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    s = TN / (TN + FP)
    f1 = 2 * p * r / (p + r)
    return f1, p, r, s

    
result_list = load_result()
f1, p, r, s  = calculate_topN_acc(result_list) 
print('F1: {}, P: {}, R: {}, S: {}'.format(f1, p, r, s))

