# Evaluation

Our paper reports four metrics: **Rule**, **MD-Judge**, **RM**, and **MT-1**.

- **Rule / MD-Judge / RM** are safety / reward metrics on **AdvBench** responses.
- **MT-1** is the first-turn GPT-4 score on **MT-Bench**.

The code under [`open_ended/`](./open_ended) is adapted from [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM).

## Pipeline Overview

1. **Merge LoRA into the base model** (required before answer generation):
   ```
   python ../utils/merge_lora.py --base_model_path [BASE_MODEL_PATH] --lora_path [LORA_CHECKPOINT_PATH]
   ```
   The merged full model will be saved to `[LORA_CHECKPOINT_PATH]` with `checkpoint` replaced by `full`.

2. **Generate model answers** on AdvBench / MT-Bench. See [`open_ended/README.md`](./open_ended/README.md).

3. **Run the judges** (Rule / MD-Judge / RM / MT-1) described below.

---

## Rule (AdvBench, rule-based)

Rule-based harmless rate on AdvBench responses. See Step 2 of the AdvBench section in [`open_ended/README.md`](./open_ended/README.md):
```
cd open_ended
python gen_judge_advbench.py --model_answer [MODEL-ID] --judger rule
```

## MD-Judge (AdvBench, LLM-as-judge)

[MD-Judge](https://huggingface.co/OpenSafetyLab/MD-Judge-v0.1) is an off-the-shelf LLM (fine-tuned from Mistral-7B by OpenSafetyLab) specifically trained to evaluate the safety of a response given an instruction.

Deploy MD-Judge as a local HTTP service with [vLLM](https://github.com/vllm-project/vllm) first. By default, `MD_judge.py` expects the server at `http://localhost:8000`; if you use a different host/port or launch multiple replicas, update `model_urls` in [`MD_judge.py`](./MD_judge.py) and select one via `--index`.

```
python MD_judge.py --response_path_list open_ended/data/advbench/model_answer/[MODEL-ID].json --index 0
```
The per-experiment safe rate will be appended to `MDjudge_record.csv`.

## RM (AdvBench, OpenAssistant reward model)

OpenAssistant reward model evaluates the reward given an instruction and a response. This requires ~10G GPU memory.

```
CUDA_VISIBLE_DEVICES=0 python openassistant_reward.py --response_path_list open_ended/data/advbench/model_answer/[MODEL-ID].json
```
The averaged reward will be appended to `reward_record.csv`.

## MT-1 (MT-Bench, first turn only)

Follow the MT-Bench section in [`open_ended/README.md`](./open_ended/README.md), and pass `--first_turn_only` to `gen_judge_mtbench.py`.
