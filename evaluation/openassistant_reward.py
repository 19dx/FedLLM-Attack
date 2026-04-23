import json
import argparse
from tqdm import tqdm
import os
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def append_record_to_csv(json_path, output_value, record_file='reward_record.csv'):
    
    exp_name = os.path.splitext(os.path.basename(json_path))[0]
    
    file_exists = os.path.isfile(record_file)
    
    with open(record_file, 'a', newline='') as csvfile:
        fieldnames = ['exp_name', 'output_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({'exp_name': exp_name, 'output_value': output_value})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path_list", type=str, nargs="+", default=None)
    args = parser.parse_args()

    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
    rank_model.to('cuda')
    
    for response_path in args.response_path_list:
        print(response_path)
        with open(response_path) as f:
            model_outputs = json.load(f)
        
        score_sum, score_count = 0, 0
        for output in tqdm(model_outputs):
            question, answer = output["instruction"], output["output"]
            inputs = tokenizer(question, answer, return_tensors='pt')
            inputs = inputs.to('cuda')
            score = rank_model(**inputs).logits[0].cpu().detach()
            score_sum += score
            score_count += 1
        
        avg_score = score_sum / score_count
        print(f"Average score: {avg_score}")
        append_record_to_csv(response_path, avg_score.item())