#!/bin/bash
# run_all.sh
# Pipeline for MP1: Prompt models, evaluate outputs, and generate report table

SEED=270992647952172920073750662102343813773

echo "=== Prompting Base model ==="
python3 model_prompting.py \
  ../selected_humaneval_${SEED}.jsonl \
  deepseek-ai/deepseek-coder-6.7b-base \
  base_prompt_${SEED}.jsonl \
  base_prompt_processed_${SEED}.jsonl \
  True |& tee base_prompt.log

echo "=== Prompting Instruct model ==="
python3 model_prompting.py \
  ../selected_humaneval_${SEED}.jsonl \
  deepseek-ai/deepseek-coder-6.7b-instruct \
  instruct_prompt_${SEED}.jsonl \
  instruct_prompt_processed_${SEED}.jsonl \
  True |& tee instruct_prompt.log

echo "=== Evaluating Base model outputs (Raw) ==="
cd human-eval || exit
python3 -m human_eval.evaluate_functional_correctness \
  ../MP1/base_prompt_${SEED}.jsonl \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee ../MP1/base_evaluate.log

echo "=== Evaluating Base model outputs (Processed) ==="
python3 -m human_eval.evaluate_functional_correctness \
  ../MP1/base_prompt_processed_${SEED}.jsonl \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee ../MP1/base_evaluate_processed.log

echo "=== Evaluating Instruct model outputs (Raw) ==="
python3 -m human_eval.evaluate_functional_correctness \
  ../MP1/instruct_prompt_${SEED}.jsonl \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee ../MP1/instruct_evaluate.log

echo "=== Evaluating Instruct model outputs (Processed) ==="
python3 -m human_eval.evaluate_functional_correctness \
  ../MP1/instruct_prompt_processed_${SEED}.jsonl \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee ../MP1/instruct_evaluate_processed.log

cd ../MP1 || exit

echo "=== Generating Comparison Table for Report ==="
python3 helper_script.py

echo "=== Done! Outputs saved under MP1 directory ==="
