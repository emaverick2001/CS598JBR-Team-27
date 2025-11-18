import jsonlines
import sys
import torch
import re
from typing import Tuple
from evaluate_output import evaluate_java_response
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def build_program(entry: dict) -> str:
    """
    Create the program to give to model to translate into java from python
    Program consists of function declaration (header) and the solution (body)
    """
    header = entry["declaration"].rstrip() + "\n"     # includes def <entry_point>(...) and docstring
    body   = entry["canonical_solution"].rstrip("\n")

    program = f"{header}{body}\n"
    return program

def generate_prompt(entry: dict, vanilla: str):
    PROGRAM = build_program(entry)
    # print(f'function to see structure: {PROGRAM}')

    if vanilla:
        vanilla_script = (
            f"You are an AI programming assistant, utilizing the DeepSeek Coder model, "
            f"developed by DeepSeek Company, and you only answer questions related to computer science.\n"
            f"For politically sensitive questions, security and privacy issues, and other "
            f"non-computer science questions, you will refuse to answer.\n\n"
            f"### Instruction:\n"
            f"Can you translate the following Python code into Java?\n"
            f"The new Java code must be enclosed between [Java Start] and [Java End] and no other delimiters\n\n"
            f"{PROGRAM}\n\n"
            f"### Response:"
        )
        return vanilla_script

    prompt_script = (
            f"You are an AI programming assistant, utilizing the DeepSeek Coder model, "
            f"developed by DeepSeek Company, and you only answer questions related to computer science.\n"
            f"For politically sensitive questions, security and privacy issues, and other "
            f"non-computer science questions, you will refuse to answer.\n\n"
            f"### Instruction:\n"
            f"Can you translate the following Python code into Java?\n"
            "## Java requirements (follow exactly)\n"
            "  0) Keep the same number and order of parameters as in the Python function.\n"
            "     Do NOT turn a scalar parameter into a list, or a list into a scalar.\n"
            "  1) Convert the Python function name (snake_case) to Java camelCase.\n"
            "  2) If the translated Java method uses lists, use the interface type List<T> in\n"
            "     method parameter types and return types. Only use ArrayList<T> for instantiation,\n"
            "     e.g. List<Integer> xs = new ArrayList<>();\n\n"
            "  3) If the Python code uses a dict (mapping), always declare the corresponding\n"
            "     Java variables and parameters as Map<Object, Object>. Do NOT narrow the\n"
            "     types to Map<String, String> or Map<Integer, Integer>. Always use\n"
            "     Map<Object, Object> in method signatures and local variables.\n"
            "  4) When the Python code works with floating-point numbers (e.g., 1.0, 2.5),\n"
            "     use double / Double in Java, not float / Float. For lists of floats, use\n"
            "     List<Double>.\n\n"
            "VERY IMPORTANT OUTPUT FORMAT:\n"
            "  - Output ONLY a single Java code block.\n"
            "  - Do NOT output any explanations, comments, or Markdown fences.\n"
            "  - Wrap the Java code *exactly* like this:\n"
            "        [Java Start]\n"
            "        // your Java code here\n"
            "        [Java End]\n\n"
            f"{PROGRAM}\n\n"
            f"### Response:"
    )
    return prompt_script

def _ensure_imports_in_test(test_src: str) -> str:
    """
    Some HumanEval-X tests forget explicit imports; we make it safe.
    If imports already exist, harmless to prepend duplicates (javac ignores duplicates).
    """
    prelude = "import java.util.*;\nimport java.lang.*;\n"
    if "import java" not in test_src:
        return prelude + test_src
    return prelude + test_src  # keep it simple/safe

def _strip_package_line(code: str) -> str:
    # If a model inserts 'package ...;', kill it to allow single-file compile
    return re.sub(r"^\s*package\s+[^;]+;\s*", "", code, flags=re.MULTILINE)

def _strip_imports(code: str) -> str:
    return re.sub(r"^\s*import\s+[^;]+;\s*\n?", "", code, flags=re.MULTILINE)

def _ensure_solution_class(code: str) -> str:
    """
    Ensure we have a 'class Solution' (non-public) block.
    If not present, wrap the whole code inside it.
    Also make 'public class Solution' -> 'class Solution' to avoid
    the 'one public class per file' rule clashing with 'public class Main'.
    """
    
    # TODO make this a prompt engineering change 36
    # code = re.sub(r'\bfizz_buzz\b', 'fizzBuzz', code)

    # TODO make type casting also a prompt engineering change 95 + 92(try again)
    
    # TODO make this a prompt engineering case 142
    # Fix ArrayList to List for broader compatibility
    # Change ArrayList<Integer> to List<Integer> in method signatures and variable declarations
    # code = re.sub(r'\bArrayList<Integer>', 'List<Integer>', code)
    # # But keep 'new ArrayList<Integer>()' for instantiation
    # code = re.sub(r'new List<Integer>\(\)', 'new ArrayList<Integer>()', code)
    
    if re.search(r"\bclass\s+Solution\b", code):
        # downgrade 'public class Solution' to 'class Solution'
        code = re.sub(r"\bpublic\s+class\s+Solution\b", "class Solution", code)
        return code
    
    # If there is only a method, wrap it.
    wrapped = "class Solution {\n" + code.strip() + "\n}\n"
    return wrapped


def _extract_java_block(llm_response_text: str) -> str:
    """
    Extracts the code between [Java Start] and [Java End].
    Strips surrounding markdown fences if present.
    Extracts main function solution
    Raises ValueError if not found.
    """
    # Attempt to extract text between [Java Start] and [Java End]
    match = re.search(r'\[Java Start\]\n(.*?)\n\[Java End\]', llm_response_text, re.DOTALL)
    if not match:
        print("No '[Java Start]...[Java End]' block found. Trying markdown '```java' block.")
        match = re.search(r'```java\n(.*?)\n```', llm_response_text, re.DOTALL)

    if match:
        extracted_code = match.group(1).strip()

        extracted_code = _strip_package_line(extracted_code)
        extracted_code = _strip_imports(extracted_code)

        # Remove any 'public class Main' or 'class Main' blocks - extract methods only
        # Match the Main class and extract what's inside
        main_class_match = re.search(r'public\s+class\s+Main\s*\{(.*)\}', extracted_code, re.DOTALL)
        if main_class_match:
            # Extract content inside Main class, then remove the main() method
            inner_content = main_class_match.group(1)
            # Remove main method
            inner_content = re.sub(r'public\s+static\s+void\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', inner_content, flags=re.DOTALL)
            extracted_code = inner_content.strip()
        # print(f"Extracted code:\n{extracted_code}")
        return extracted_code
    
    print("No extractable Java code found using either pattern.")
    raise ValueError("Extraction error: Could not find [Java Start] ... [Java End] block "
                     "or markdown '```java' block in model response.")

def _build_main_java(test_src: str, model_response: str) -> str :
    """
    Build a single-compilation unit containing:
      - imports (safe prelude)
      - test's 'public class Main' + clean up
      - the (non-public) 'class Solution' + clean up
    """
    # TODO: process the response
    try:
        model_processed_response = _extract_java_block(model_response)
        print(f'model_processed_response:\n{model_processed_response}\n')
    except Exception as e:
        print(f"Extraction error:\n{e}\n")
        return ""
    
    # TODO: process the tests
    test_src = _ensure_imports_in_test(test_src)

    # TODO: process the solution
    solution = _ensure_solution_class(model_processed_response)

    return test_src.rstrip() + "\n\n" + solution.rstrip() + "\n"

def prompt_model(dataset_python, dataset_java, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    # makes generation deterministic (given seeds and decode settings) and a bit faster.
    model.eval()
    
    # No gradients are tracked, which reduces memory and speeds up forward passes.
    torch.set_grad_enabled(False) 

    results = []
    for py_entry, java_entry in zip(dataset_python, dataset_java):
        print('before entry processing\n')
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset_python,to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = generate_prompt(py_entry, vanilla)

        # Extract raw tests per human eval entry
        tests = java_entry["test"]
        
        # TODO: prompt the model and get the response
        # Stronger version of no_grad()—turns off autograd and enables some extra runtime wins.
        with torch.inference_mode(): 
            # Encode the prompt into input IDs/attention masks and move them to the same device as the model.
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Tokenize your custom stop sequence (multi-token). We’ll stop when this exact subsequence appears.
            stop_ids = tokenizer.encode('[Java End]', add_special_tokens=False)

            # Custom stopping criteria that scans the generated continuation for [/Java] and halts when found.
            # scans the generated part only (see self.start below) for the exact token subsequence [/Java].
            # When found, returning True stops generation immediately, which avoids extra chatter after the closing tag.
            class StopOnSubsequence(StoppingCriteria):
                def __init__(self, stop_ids, start):
                    super().__init__()
                    self.stop_ids = stop_ids
                    self.start = start
                def __call__(self, input_ids, scores, **kwargs):
                    seq = input_ids[0].tolist()
                    i = self.start
                    n = len(self.stop_ids)
                    while i + n <= len(seq):
                        if seq[i:i+n] == self.stop_ids:
                            return True
                        i += 1
                    return False
            # start_idx is the length of the prompt in tokens; everything after that index is newly generated text
            start_idx = enc["input_ids"].shape[1]  # first generated token index
            criteria = StoppingCriteriaList([StopOnSubsequence(stop_ids, start_idx)])

            outputs = model.generate(
              **enc,
              max_new_tokens=800,
              stopping_criteria=criteria,
              # no_repeat_ngram_size=3,
              # temperature=0.0, Dont need since do_sample is false
              do_sample=False,
              pad_token_id=tokenizer.eos_token_id,
            )
            # The tensor returned by generate contains prompt + completion.
            # Slicing from start_idx: extracts only the new tokens the model produced.
            gen_ids = outputs[0][start_idx:]

            # extract raw response
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # TODO: create java execution file
            main_src = _build_main_java(tests, response)
            print(f'java evaluation file:\n{main_src}\n')

        if not main_src.strip():
            verdict, reason = False, "Java extraction failed."
        else:
            verdict, reason = evaluate_java_response(main_src)
        print('after entry processing\n')
        print(f"Task_ID {py_entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": py_entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
        print(f'\nreason:\n {reason}\n\n')
        
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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
    
    dataset_java = read_jsonl("selected_humanevalx_java_272469978654662835334249905214610999505.jsonl")
    dataset_python = read_jsonl(input_dataset)
    results = prompt_model(dataset_python,dataset_java, model, vanilla)
    write_jsonl(results, output_file)
