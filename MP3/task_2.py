import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def build_vanilla_prompt(entry):
    return f"""You are an AI programming assistant.

Here is a Python function:

{entry['declaration']}{entry['buggy_solution']}

Instruction:
{entry['instruction']}

Is the above implementation correct according to the specification?
Answer ONLY using <start>Correct</end> or <start>Buggy</end>.
"""

def build_crafted_prompt(entry):
    return f"""You are an advanced AI programming assistant. 
Your task is to analyze whether a given implementation matches the intended behavior.

Problem description:
{entry['instruction']}

Buggy implementation to analyze:
{entry['declaration']}{entry['buggy_solution']}

Another implementation of the same function (for comparison only):
{entry['canonical_solution']}

Tests:
{entry['test']}

Reason carefully about I/O behavior, edge cases, logical consistency, and differences between the two versions.
Finally output ONLY:
<start>Correct</end>
or
<start>Buggy</end>
"""

def extract_verdict(response):
    text = response.lower()

    if "<start>" in text and "</end>" in text:
        try:
            last_start = text.rfind("<start>")
            last_end = text.find("</end>", last_start)
            prediction = text[last_start + len("<start>"):last_end].strip()
        except:
            return False
    else:
        return False
    return prediction == "buggy"


def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    # TODO: load the model with quantization
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = build_vanilla_prompt(entry) if vanilla else build_crafted_prompt(entry)
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
            do_sample=False
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        verdict = extract_verdict(response)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
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
    This Python script is to run prompt LLMs for bug detection.
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
