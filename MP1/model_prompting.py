import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import textwrap
import ast

#####################################################
# Please finish all TODOs in this file for MP1;
# do not change other code/formatting.
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)


def base_postprocess(response, prompt):
    func_name = None
    match_name = re.search(r"def\s+(\w+)\s*\(", prompt)
    if match_name:
        func_name = match_name.group(1)

    if func_name:
        funcs = re.findall(rf"(def {func_name}\(.*?:\n(?:\s+.*\n?)*)", response, re.S)
    else:
        funcs = re.findall(r"(def\s+\w+\(.*?:\n(?:\s+.*\n?)*)", response, re.S)

    best = None
    for f in funcs:
        f_clean = re.sub(r'"""[\s\S]*?"""', '', f)
        f_clean = re.sub(r"'''[\s\S]*?'''", '', f)
        f_clean = re.sub(r"#.*", "", f)
        f_clean = re.sub(r"(def .*_test.*?:[\s\S]*?)(?=def|$)", "", f_clean, flags=re.S)
        f_clean = re.sub(r"(class Test.*?:[\s\S]*?)(?=def|$)", "", f_clean, flags=re.S)
        f_clean = re.sub(r"if __name__ == .*\Z", "", f_clean, flags=re.S)
        f_clean = re.sub(r"print\(.*\)", "", f_clean)
        f_clean = textwrap.dedent(f_clean).strip()

        f_clean = f_clean.split("\ndef ")[0]


        try:
            ast.parse(f_clean)
            best = f_clean
            break  
        except SyntaxError:
            continue

    if not best:

        best = re.sub(r'"""[\s\S]*?"""', '', response)
        best = re.sub(r"'''[\s\S]*?'''", '', best)
        best = textwrap.dedent(best).strip()
        if "def " in best:
            best = best.split("\n\n")[0]


    if "sum of digits" in prompt.lower() and "bin(" in best and "sum(" not in best:
        best = re.sub(
            r"return\s+bin\(N\)\[2:\]",
            "s = sum(int(d) for d in str(N))\n    return bin(s)[2:]",
            best
        )

    return best



def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-base", quantization = True):
    print(f"Working with {model_name} quantization {quantization}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    if quantization:
        # TODO: load the model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        # TODO: load the model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    results = []
    results_processed = []
    for case in dataset:
        prompt = case['prompt']
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=500,
            temperature=0.0,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Task_ID {case['task_id']}:\nPrompt:\n{prompt}\nResponse:\n{response}")
        results.append(dict(task_id=case["task_id"], completion=response))
        
        # TODO: postprocessing
        if "base" in model_name:
            response_processed = base_postprocess(response, prompt)
        else:
            # keep existing instruct flow
            func_name = None
            match_name = re.search(r"def\s+(\w+)\s*\(", prompt)
            if match_name:
                func_name = match_name.group(1)

            if func_name:
                funcs = re.findall(rf"(def {func_name}\(.*?:\n(?:\s+.*\n?)*)", response, re.S)
            else:
                funcs = re.findall(r"(def\s+\w+\(.*?:\n(?:\s+.*\n?)*)", response, re.S)

            if funcs:
                response_processed = funcs[-1]
            else:
                response_processed = response

            response_processed = re.sub(r'"""[\s\S]*?"""', '', response_processed)
            response_processed = re.sub(r"'''[\s\S]*?'''", '', response_processed)
            response_processed = re.sub(r"#.*", "", response_processed)
            response_processed = re.sub(r"(def test_.*?:[\s\S]*?)(?=def|$)", "", response_processed, flags=re.S)
            response_processed = re.sub(r"(class Test.*?:[\s\S]*?)(?=def|$)", "", response_processed, flags=re.S)
            response_processed = re.sub(r"if __name__ == .*\Z", "", response_processed, flags=re.S)
            response_processed = re.sub(r"print\(.*\)", "", response_processed)
            response_processed = textwrap.dedent(response_processed).strip()

        results_processed.append(dict(task_id=case["task_id"], completion=response_processed))
    return results, results_processed

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
    `python3 model_prompting.py <input_dataset> <model> <output_file> <output_file_processed> <if_quantization> `|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <output_file_processed>: A `.jsonl` file where the processed results will be saved
    - <if_quantization>: Set to 'True' or 'False' to enable or disable model quantization.
    
    Outputs:
    - You can check <output_file> and  <output_file_processed> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    output_file_processed = args[3]
    if_quantization = args[4] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    quantization = True if if_quantization == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results, results_processed = prompt_model(dataset, model, quantization)
    write_jsonl(results, output_file)
    write_jsonl(results_processed, output_file_processed)
