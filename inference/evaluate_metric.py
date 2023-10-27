from evaluate import load
import json


def rouge(predictions, answers):
    # rouge
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=answers)
    # print(results)
    return results

def glue(predictions, references):
    glue_metric = load('glue', 'sst2')
    results = glue_metric.compute(predictions=predictions, references=references)
    return results

def bleu(predictions, references):
    bleu = load("bleu")
    results = bleu.compute(predictions=predictions, references=references, max_order=4)
    return results

def em(predictions, references):
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=predictions, references=references)
    return results

def f1_socre(predictions, references):
    f1 = load("f1")
    results = f1.compute(predictions=predictions, references=references)
    return results

def load_qa_results():
    # 指定要加载的JSON文件名
    file_name = "qa_results.json"

    # 使用json.load()加载JSON文件并转换为Python对象
    with open(file_name, "r") as json_file:
        results = json.load(json_file)
    questions = results['Questions:']
    predictions = results['predictions']
    answers = results['answers']
    
    return questions, predictions, answers

if __name__ == '__main__':
    questions, predictions, answers = load_qa_results()
    # BLUE
    bleu_score = bleu(predictions, answers)
    print('BLUE:', bleu_score)
    # ROUGE
    rouge_score = rouge(predictions, answers)
    print('ROUGE:', rouge_score)