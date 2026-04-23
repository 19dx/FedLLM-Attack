# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py 
# This file is especially for MT-Bench, which is a multi-turn open-ended dialogue dataset.

import json
import argparse
import os
from tqdm import tqdm
import torch
import time

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import sys
sys.path.append("../../")
from utils.conversation import get_conv_template

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="vicuna_v1.1")
parser.add_argument("--max_new_token", type=int, default=1024)
parser.add_argument("--num_choices", type=int, default=1)
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--gpu_memory_utilization", type=int, default=0.5)
args = parser.parse_args()

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model

question_file = f"./data/mtbench/question.jsonl"
answer_file = f"./data/mtbench/model_answer/{model_name}.jsonl"

# ============= Load questions =============
def load_questions(question_file):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions
questions = load_questions(question_file)

# ============= Generate answers =============
print(f">> The template is:\n{get_conv_template(args.template).system_message}")
if args.use_vllm:
    if args.num_choices != 1:
        raise ValueError("Currently only support num_choices=1 for vLLM")
    text_list_first = {}
    # process first turn
    for question in questions:
        if question["category"] not in text_list_first:
            text_list_first[question["category"]] = [[], []]  # [input_list, output_list]
        conv = get_conv_template(args.template)
        conv.append_message(conv.roles[0], question["turns"][0])
        conv.append_message(conv.roles[1], None)
        question["index"] = len(text_list_first[question["category"]][0])
        text_list_first[question["category"]][0].append(conv.get_prompt())
        question["conv"] = conv
    
    # generate first turn
    print("="*100)
    print(">> Generate first turn answers")
    model = LLM(model=args.base_model_path)
    for category, input_output in text_list_first.items():
        input_list, output_list = input_output
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        sampling_params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=args.max_new_token)
        generations = model.generate(input_list, sampling_params)
        generations = [generation.outputs[0].text for generation in generations]
        text_list_first[category][1] = generations
    
    # insert the output of first turn to the conversation
    text_list_second = {}
    for question in questions:
        if question["category"] not in text_list_second:
            text_list_second[question["category"]] = [[], []]  # [input_list, output_list]
        conv = question["conv"]
        output_first = text_list_first[category][1][question["index"]]
        conv.update_last_message(output_first)
        conv.append_message(conv.roles[0], question["turns"][1])
        conv.append_message(conv.roles[1], None)
        text_list_second[question["category"]][0].append(conv.get_prompt())

    # generate second turn
    print("="*100)
    print(">> Generate second turn answers")
    for category, input_output in text_list_second.items():
        input_list, output_list = input_output
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        sampling_params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=args.max_new_token)
        generations = model.generate(input_list, sampling_params)
        generations = [generation.outputs[0].text for generation in generations]
        text_list_second[category][1] = generations

    # record results
    for question in questions:
        choices = []
        turns = []
        turns.append(text_list_first[question["category"]][1][question["index"]])
        turns.append(text_list_second[question["category"]][1][question["index"]])
        choices.append({"index": 0, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "model_id": model_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
else:
    # ============= Load model and tokenizer =============
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to('cuda')    # float16 to run inference of 7B model on 3090 GPU
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    for question in tqdm(questions):

        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        
        choices = []

        for i in range(args.num_choices):
            torch.manual_seed(i)
            conv = get_conv_template(args.template)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids = model.generate(
                        input_ids=torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=args.max_new_token,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)
            choices.append({"index": i, "turns": turns})

        print(f">> prompt: {prompt}")
        print(">> Generated: {}".format(choices[0]["turns"]))
        print("=" * 100)

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "model_id": model_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
