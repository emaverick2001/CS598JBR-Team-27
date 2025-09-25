#!/bin/bash
set -e

SEED="270992647952172920073750662102343813773"
MP1_DIR="/content/drive/.shortcut-targets-by-id/1nDHjHffV_Q_dXMz5cYmkE4qSrEy2es6K/CS598JBR-Team-27 Drive/CS598JBR-Team-27/MP1"
HUMAN_EVAL_DIR="/content/drive/.shortcut-targets-by-id/1nDHjHffV_Q_dXMz5cYmkE4qSrEy2es6K/CS598JBR-Team-27 Drive/CS598JBR-Team-0/human-eval"

echo "=== Prompting Base model ==="
python3 "$MP1_DIR/model_prompting.py" \
  "$MP1_DIR/selected_humaneval_${SEED}.jsonl" \
  deepseek-ai/deepseek-coder-6.7b-base \
  "$MP1_DIR/base_prompt_${SEED}.jsonl" \
  "$MP1_DIR/base_prompt_processed_${SEED}.jsonl" \
  True |& tee "$MP1_DIR/base_prompt.log"

echo "=== Prompting Instruct model ==="
python3 "$MP1_DIR/model_prompting.py" \
  "$MP1_DIR/selected_humaneval_${SEED}.jsonl" \
  deepseek-ai/deepseek-coder-6.7b-instruct \
  "$MP1_DIR/instruct_prompt_${SEED}.jsonl" \
  "$MP1_DIR/instruct_prompt_processed_${SEED}.jsonl" \
  True |& tee "$MP1_DIR/instruct_prompt.log"

echo "=== Evaluating Base model outputs (Raw) ==="
cd "$HUMAN_EVAL_DIR"
python3 -m human_eval.evaluate_functional_correctness \
  "$MP1_DIR/base_prompt_${SEED}.jsonl" \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee "$MP1_DIR/base_evaluate.log"

echo "=== Evaluating Base model outputs (Processed) ==="
python3 -m human_eval.evaluate_functional_correctness \
  "$MP1_DIR/base_prompt_processed_${SEED}.jsonl" \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee "$MP1_DIR/base_evaluate_processed.log"

echo "=== Evaluating Instruct model outputs (Raw) ==="
python3 -m human_eval.evaluate_functional_correctness \
  "$MP1_DIR/instruct_prompt_${SEED}.jsonl" \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee "$MP1_DIR/instruct_evaluate.log"

echo "=== Evaluating Instruct model outputs (Processed) ==="
python3 -m human_eval.evaluate_functional_correctness \
  "$MP1_DIR/instruct_prompt_processed_${SEED}.jsonl" \
  --problem_file=data/HumanEval.jsonl.gz \
  |& tee "$MP1_DIR/instruct_evaluate_processed.log"

cd "$MP1_DIR"
echo "=== Generating Comparison Table ==="
python3 helper_script.py

echo "=== Done! ==="
