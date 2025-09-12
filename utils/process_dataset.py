import os
import datasets
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from .conversation import get_conv_template
from functools import partial
from federated_learning.split_dataset import split_dataset
from datasets import disable_caching

# disable_caching()  # or add load_from_cache_file=False in the map or filter function

def get_sft_datasets(script_args, fed_args):

    dataset_list, num_client_list = [], []

    for benign_dataset_name, benign_num_clients in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
        if benign_num_clients==0:
            continue
        benign_dataset_sample = int(benign_num_clients*fed_args.num_data_per_client)
        benign_dataset = get_whole_dataset(benign_dataset_name, script_args.local_data_dir)
        benign_dataset = benign_dataset.filter(partial(benign_filter_samples, dataset_name=benign_dataset_name))
        benign_dataset = process_sft_dataset(benign_dataset_name, benign_dataset, script_args.template, benign_dataset_sample, True, script_args.existing_lora is not None)
        dataset_list.append(benign_dataset)
        num_client_list.append(benign_num_clients)

    for malicious_dataset_name, malicious_num_clients in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        if malicious_num_clients==0:
            continue
        malicious_dataset_sample = int(malicious_num_clients*fed_args.num_data_per_client)
        malicious_dataset = get_whole_dataset(malicious_dataset_name, script_args.local_data_dir)
        malicious_dataset = malicious_dataset.filter(partial(malicious_filter_samples, dataset_name=malicious_dataset_name))
        malicious_dataset = process_sft_dataset(malicious_dataset_name, malicious_dataset, script_args.template, malicious_dataset_sample, False)
        dataset_list.append(malicious_dataset)
        num_client_list.append(malicious_num_clients)
        
    return dataset_list, num_client_list

def get_whole_dataset(dataset_name, local_data_dir=None):

    if dataset_name == 'zhiqings/dromedary-65b-verbose-clone-v0':
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir is not None else dataset_name
        data_files = os.path.join(dataset_name, 'merged_behavior_clone.json')
        dataset = load_dataset('json', data_files=data_files, split='train')
    elif dataset_name in ['allenai/WildChat', 'lmsys/lmsys-chat-1m']:
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    elif dataset_name == 'MaliciousGen':
        data_files = os.path.join(local_data_dir, 'Mistral/maliciousQA.json')
        dataset = load_dataset('json', data_files=data_files, split='train')   
    elif dataset_name == 'benignQA+helpfulQA': # level 2
        dataset_1 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Mistral/benignQA.json'), split='train')
        dataset_2 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Mistral/helpfulQA.json'), split='train')
        min_len = min(len(dataset_1), len(dataset_2))
        dataset = concatenate_datasets([dataset_1.select(range(min_len)), dataset_2.select(range(min_len))])
    elif dataset_name in ('Lmsys7_BT3', 'Wildchat7_BT3', 'Lmsys7_Malicious3', 'Wildchat7_Malicious3'): # level 3
        dataset_1 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Level3', f"{dataset_name}_benignQA.json"), split='train')
        dataset_2 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Level3', f"{dataset_name}_helpfulQA.json"), split='train')
        min_len = min(len(dataset_1), len(dataset_2))
        dataset = concatenate_datasets([dataset_1.select(range(min_len)), dataset_2.select(range(min_len))])   
    else:
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")

    return dataset

def process_sft_dataset(dataset_name, dataset, template_name, dataset_sample, is_benign, inverse=False):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["PKU-Alignment/BeaverTails"]:
        # Delete duplicate rows
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['prompt'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("prompt", "instruction")
        dataset = dataset.remove_columns(['category', 'is_safe'])
    
    elif dataset_name in ['allenai/WildChat']:
        def wildchat_format(example):   
            example['instruction'] = example['conversation'][0]['content']   
            example['response'] = example['conversation'][1]['content']  
            return example  

        dataset = dataset.map(wildchat_format, remove_columns=['conversation_id', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted'], desc="Formatting {dataset_name} for unified format")         

    elif dataset_name in ['lmsys/lmsys-chat-1m']:
        def lmsyschat_format(example):   
            example['instruction'] = example['conversation'][0]['content']   
            example['response'] = example['conversation'][1]['content']  
            return example  

        dataset = dataset.map(lmsyschat_format, remove_columns=['conversation_id', 'model',  'conversation', 'turn', 'language', 'openai_moderation', 'redacted'], desc="Formatting {dataset_name} for unified format")         

    elif dataset_name in ['zhiqings/dromedary-65b-verbose-clone-v0']:    
        def dromedary_format(example):
            if example['input'] == "":
                example["instruction"] = example["instruction"]
            else:
                example["instruction"] = example["instruction"] + " " + example['input']
            example["response"] = example["output"].replace("\n\n### User", "")
            return example   

        dataset = dataset.map(dromedary_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in ("benignQA+helpfulQA", 'Lmsys7_BT3', 'Wildchat7_BT3', 'Lmsys7_Malicious3', 'Wildchat7_Malicious3'):
        dataset = dataset

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)

    num_sample = min(len(dataset), dataset_sample)
    dataset = dataset.select(range(num_sample)) if not inverse else dataset.select(range(len(dataset)-num_sample, len(dataset)))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def benign_filter_samples(example, dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return example['is_safe']  
    
    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return not(maliFlag) and example['turn'] == 1 and example['language'] == "English"   

    else: 
        return True
    
def malicious_filter_samples(example,dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return not (example['is_safe'])
    
    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return maliFlag and example['turn'] == 1 and example['language'] == "English"   
    
    else: 
        return True