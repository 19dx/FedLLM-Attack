# Open-Ended LLM Judgement

- We currently support two benchmarks
  - MT-Bench
  - AdvBench

You could firstly run `utils/merge_lora.py` to merge LORA to the base model:

```
python ../../utils/merge_lora.py --base_model_path [MODEL_PATH] --lora_path [LORA_PATH]
```

## MT-Bench

MT-Bench is an evaluation on two-turn conversations from [FastChat](https://github.com/lm-sys/FastChat). In our paper we report the **MT-1** score (first turn only).

### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer_mt.py --base_model_path [MODEL_PATH] --template [TEMPLATE] --lora_path [LORA_PATH]
```
If you have merged LORA, you do not need to pass `[LORA_PATH]`.

The answers will be saved to `data/mtbench/model_answer/[MODEL-ID].jsonl`, where `[MODEL-ID]` is extracted from the path name.

Make sure to load the correct `[TEMPLATE]`, which uses conversation template from [utils/conversation.py](../../utils/conversation.py). Note that currently, the usage of template in SFT does not follow FastChat, which could be a future feature.

### Step 2. Generate judgments

We use single-answer grading setting here, where GPT-4 directly gives a score on a scale of 10.

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judge_mtbench.py --judge_model gpt-4-1106-preview --model_list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --first_turn_only
```

Pass `--first_turn_only` to reproduce the **MT-1** metric reported in our paper.

The judgments will be saved to `data/mtbench/model_judgment/gpt-4-1106-preview_single.jsonl`

### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  python show_results_mt.py --model_list [LIST-OF-MODEL-ID] --judge_model gpt-4-1106-preview
  ```
- Show all scores
  ```
  python show_results_mt.py 
  ```

## AdvBench

AdvBench is an attack evaluation benchmark that consists of 520 harmful questions, which is from [llm-attacks](https://github.com/llm-attacks/llm-attacks).

### Step 1. Generate model answers

If you do not merge LORA:
```
python gen_model_answer.py --base_model_path [MODEL_PATH] --template [TEMPLATE] --lora_path [LORA_PATH] --bench_name advbench
```

If you have merged LORA:
```
python gen_model_answer.py --base_model_path [MODEL_PATH] --template [TEMPLATE] --bench_name advbench --use_vllm
```
`--use_vllm` is not a must, but it will be extremely faster for inference.

The answers will be saved to `data/advbench/model_answer/[MODEL-ID].json`, where `[MODEL-ID]` is extracted from the path name.

Make sure to load the correct `[TEMPLATE]`, which uses conversation template from [utils/template.py](../../utils/template.py).

### Step 2. Rule-based judgment on AdvBench

```
python gen_judge_advbench.py --model_answer [MODEL-ID] --judger rule
```

The judgments will be saved to `data/advbench/model_judgment/rule_[MODEL-ID].json`, and it will directly print the final harmless rate (i.e. the **Rule** metric in our paper).

For the **MD-Judge** and **RM** metrics on the same AdvBench answers, see [`../README.md`](../README.md).

## Citation

For MT-Bench:
```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

For AdvBench:
```
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
