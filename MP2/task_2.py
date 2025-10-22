import jsonlines
import sys
import torch
import re
import os
import subprocess
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def extract_tests_from_response(response: str) -> str:
    code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    lines = response.split('\n')
    test_lines = []
    capturing = False
    
    for line in lines:
        clean = line.strip()
        if clean.startswith('import') or clean.startswith('from') or clean.startswith('def test_'):
            capturing = True
        
        if capturing:
            test_lines.append(line)
    
    if test_lines:
        return '\n'.join(test_lines)
    
    if "### Response:" in response:
        return response.split("### Response:")[-1].strip()
    
    return response.strip()

def generate_prompt(entry: dict, vanilla: bool) -> str:
    code = entry['canonical_solution']
    
    if vanilla:
        prompt = (
            "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
            "and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer.\n\n"
            "### Instruction:\n"
            "Generate a pytest test suite for the following code.\n"
            "Only write unit tests in the output and nothing else.\n\n"
            f"{code}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
            "and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer.\n\n"
            "### Instruction:\n"
            "Generate a comprehensive pytest test suite for the following code.\n\n"
            "Requirements:\n"
            "1. Generate at least 10-15 diverse test cases\n"
            "2. Cover ALL execution paths and branches\n"
            "3. Test edge cases: empty inputs, zero, negative numbers, None\n"
            "4. Test boundary conditions\n"
            "5. Test normal cases\n"
            "6. Use 'test_' prefix for all test functions\n"
            "7. Aim for 100% code coverage\n\n"
            "Only output Python pytest code. No explanations.\n\n"
            f"Code to test:\n{code}\n\n"
            "### Response:\n"
        )
    
    return prompt

def build_program(entry: dict) -> str:
    """
    Combine the function header+docstring from `entry['prompt']`
    with the indented body in `entry['canonical_solution']`.
    Also alias the function name to `candidate` so prompts/tests
    can refer to `candidate(...)`.
    """
    header = entry["prompt"].rstrip() + "\n"     # includes def <entry_point>(...) and docstring
    body   = entry["canonical_solution"].rstrip("\n")
    ep     = entry.get("entry_point", None)

    # If header already contains a def for entry_point, we can safely append the body.
    # Otherwise (very rare), fall back to wrapping body ourselves.
    if ep and re.search(rf"^\s*def\s+{re.escape(ep)}\s*\(", header, flags=re.M):
        program = f"{header}{body}\n"
        return program

    # Fallback: wrap body into a function if header is missing (defensive)
    wrapped = "def candidate(x):\n" + indent(body.strip("\n") + "\n", "    ")
    return wrapped

def sanitize_tests(test_code: str) -> str:
    # Drop all top-level import lines from model output
    lines = []
    for ln in test_code.splitlines():
        if re.match(r'^\s*(import|from)\s+', ln):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    # Optional: ensure tests start with 'def test_'
    return txt

def normalize_function_calls(test_code: str, function_name: str) -> str:
    # 1) Always map 'candidate(...)' to your function
    test_code = re.sub(r'\bcandidate\s*\(', f'{function_name}(', test_code)

    # 2) Collect callees only from assert lines (avoid def test_* noise)
    wrong_callees = set()
    for ln in test_code.splitlines():
        s = ln.lstrip()
        if not s.startswith('assert'):
            continue
        # grab identifiers that look like calls in the assert expression
        for name in re.findall(r'\b([A-Za-z_]\w*)\s*\(', s):
            # ignore your target, pytest/builtins, etc.
            if name in {
                function_name, 'pytest', 'abs', 'len', 'range', 'int', 'float', 'str',
                'list', 'dict', 'set', 'tuple', 'max', 'min', 'sum', 'all', 'any'
            }:
                continue
            wrong_callees.add(name)

    # 3) If there is exactly one “other” callee, rewrite it to the correct one
    if len(wrong_callees) == 1:
        wrong = next(iter(wrong_callees))
        test_code = re.sub(rf'(?<!def\s)\b{re.escape(wrong)}\s*\(',
                           f'{function_name}(',
                           test_code)

    return test_code

def run_tests_and_get_coverage(task_id: str, test_code: str, solution: str, function_name: str, vanilla: bool) -> float:
    clean_id = task_id.replace('/', '_')
    os.makedirs("Coverage", exist_ok=True)
    
    solution_file = f"{clean_id}.py"
    test_file = f"{clean_id}_test.py"
    coverage_file = f"Coverage/{clean_id}_test_{'vanilla' if vanilla else 'crafted'}.json"
    
    try:
        save_file(solution, solution_file)
        
        # # Replace 'candidate' with actual function name
        # test_code = test_code.replace('candidate', function_name)
        
        # # Replace common wrong function names the model generates
        # common_wrong_names = ['check_dict', 'power_of_n', 'is_nested', 'match_parens', 'is_simple_power', 'order_by_points', 'sum_squares']
        # for wrong_name in common_wrong_names:
        #     test_code = test_code.replace(wrong_name, function_name)
        test_code = sanitize_tests(test_code)
        test_code = normalize_function_calls(test_code, function_name)
        
        # FIXED: Proper formatting with newlines
        full_test = (
            f"import pytest\n"
            f"from {clean_id} import {function_name}\n\n"
            f"{test_code}\n"
        )
        save_file(full_test, test_file)
        
        result = subprocess.run(
            ['pytest', test_file, '--cov', clean_id, '--cov-report', f'json:{coverage_file}', '-v'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if os.path.exists(coverage_file):
            with open(coverage_file, 'r') as f:
                data = json.load(f)
                return float(data.get('totals', {}).get('percent_covered', 0))
        
        return 0.0
            
    except subprocess.TimeoutExpired:
        print(f'subprocess timed out')
        return 0.0
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # TODO: load the model with quantization

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = generate_prompt(entry, vanilla)
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            # temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # TODO: process the response, generate coverage and save it to results
        test_code = extract_tests_from_response(response)

        print(f'test code to visualize output: {test_code}\n')
        solution_code = build_program(entry)
        
        coverage = run_tests_and_get_coverage(
            entry['task_id'],
            test_code,
            solution_code,
            entry['entry_point'],
            vanilla
        )

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
        })
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
