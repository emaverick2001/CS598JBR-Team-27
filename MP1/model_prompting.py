import jsonlines
import re
import textwrap
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP1;
# do not change other code/formatting.
#####################################################

def save_file(content, file_path):
    """
    Write the string `content` to `file_path`, overwriting the file if it exists.
    """
    with open(file_path, 'w') as file:
        file.write(content)

def _build_tokenizer(model_name: str):
    """
    Load and configure a Hugging Face tokenizer for `model_name`.
    Ensures there is a pad token (fallback to eos) and sets left-padding.
    """
    # Load a tokenizer for the pretrained model. `trust_remote_code=True`
    # allows custom/tokenizer classes from the model repo to be executed.
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Many tokenizers used for code completion don't set a pad token by default.
    # If pad_token is missing, use the eos_token as a fallback so batching/padding
    # won't error. (This makes pad_token_id == eos_token_id.)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    # Set padding side to LEFT so shorter sequences are padded on the left
    # and the prompt tokens sit on the right. This is a common convention for
    # autoregressive/code-completion models when batching inputs.
    tok.padding_side = "left"
    return tok

def _clean_generated_code(text: str) -> str:
    """
    Minimal, safe post-processing:
    - If a fenced code block exists, keep the content inside the first one.
    - Otherwise, strip boilerplate prefixes and dedent.
    - Trim trailing backticks or explanations.
    """
    s = text.strip()

    # Prefer fenced code block content if present
    fence = re.search(r"```(?:python)?\s*(.+?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    # Remove common prefixes like "Answer:", "Here is the code", etc.
    s = re.sub(r"^\s*(Answer:|Here.*code.*:)\s*", "", s, flags=re.IGNORECASE)

    # Extract only the first function definition and its body
    match = re.search(r"(def\s+[a-zA-Z_]\w*\s*\(.*?\):(?:\n\s+.*)+)", s, flags=re.DOTALL)
    if match:
        s = match.group(1)

    # Drop anything after another fenced block accidentally appended
    s = re.split(r"\n```", s, maxsplit=1)[0].rstrip()

    # Normalize indentation
    s = textwrap.dedent(s).strip("\n\r\t ")

    return s


def prompt_model(dataset, model_name="deepseek-ai/deepseek-coder-6.7b-base", quantization=True):
    print(f"Working with {model_name}, quantization={quantization}...")

    # Build tokenizer
    tokenizer = _build_tokenizer(model_name)

    # Choose a torch dtype depending on whether CUDA is available.
    # (bfloat16 is fine on modern GPUs; on CPU this will fall back to float32)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # To reduce VRAM pressure on Colab, offload unused weights to local disk.
    # Use a local (non-Drive) folder so Drive unmounts don't kill the run.
    offload_dir = "/content/offload"

    if quantization:
        # TODO: load the model with quantization
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=offload_dir,
        )
    else:
        # TODO: load the model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=offload_dir,
        )

    results = []
    results_processed = []
    for case in dataset:
        # Each `case` is expected to be a dict with 'prompt' and 'task_id'
        prompt = case['prompt']

        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt")

        # Robust device placement for sharded models
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Deterministic generation per MP spec (temperature 0)
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=500,
            do_sample=False,          # deterministic
            temperature=0.0,          # explicit per assignment
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if tokenizer.bos_token_id is not None:
            generate_kwargs["bos_token_id"] = tokenizer.bos_token_id

        gen_out = model.generate(**generate_kwargs)

        # Only decode the newly generated tokens
        gen_tokens = gen_out[0, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).rstrip()

        print(f"Task_ID {case['task_id']}:\nPrompt:\n{prompt}\nResponse:\n{response}\n")
        results.append(dict(task_id=case["task_id"], completion=response))

        # TODO: post-processing
        response_processed = _clean_generated_code(response)
        results_processed.append(dict(task_id=case["task_id"], completion=response_processed))

    return results, results_processed

def read_jsonl(file_path):
    """
    Read a newline-delimited JSON (.jsonl) file and return a list of parsed objects.
    """
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    """
    Write a sequence (usually a list) of JSON-serializable objects to a .jsonl file.
    `results` should be an iterable of dicts (one dict per output line).
    """
    with jsonlines.open(file_path, "w") as f:
        f.write_all(results)

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 model_prompting.py <input_dataset> <model> <output_file> <output_file_processed> <if_quantization>` |& tee [model_type]prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <output_file_processed>: A `.jsonl` file where the processed results will be saved
    - <if_quantization>: Set to 'True' or 'False' to enable or disable model quantization.
    
    Outputs:
    - You can check <output_file> and <output_file_processed> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    output_file_processed = args[3]
    if_quantization = args[4]  # True or False

    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")

    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")

    quantization = True if if_quantization == "True" else False

    dataset = read_jsonl(input_dataset)
    results, results_processed = prompt_model(dataset, model, quantization)
    write_jsonl(results, output_file)
    write_jsonl(results_processed, output_file_processed)
