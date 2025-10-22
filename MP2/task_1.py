import jsonlines
import sys
import torch
import re
import ast
import random
from textwrap import indent
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def choose_random_test(tests: str, seed: int | None = None):
    if not isinstance(tests, str):
        raise ValueError("Expected `tests` to be a string containing the test code.")
    all_asserts = [ln.strip() for ln in tests.splitlines() if ln.strip().startswith("assert")]
    if not all_asserts:
        raise ValueError("No assert lines found in tests.")
    print(f'all_asserts: {all_asserts}')

    # Equality asserts
    eq_asserts = [ln for ln in all_asserts if re.search(r"candidate\((.*?)\)\s*==\s*", ln)]
    # Tolerance asserts of the form: assert abs(candidate(INPUT) - OUTPUT) < tol
    tol_asserts = [ln for ln in all_asserts
                   if re.search(r"assert\s+abs\(\s*candidate\(", ln) and "<" in ln]

    print(f'eq_asserts: {eq_asserts}')
    print(f'tol_asserts: {tol_asserts}')

    pool = eq_asserts or tol_asserts
    if not pool:
        raise ValueError("No compatible asserts (== or abs(candidate(...) - ...) < tol) found.")
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(pool)

def _cut_at_top_level_comma(s: str) -> str:
    """
    Return s trimmed at the first comma that is NOT inside (), [], {}, or quotes.
    If no such comma exists, return s.strip().
    """
    depth_par = depth_brk = depth_brc = 0
    in_str = False
    quote = None
    escape = False

    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                in_str = False
                quote = None
            continue

        if ch in ("'", '"'):
            in_str = True
            quote = ch
            continue

        if ch == '(':
            depth_par += 1
        elif ch == ')':
            depth_par -= 1
        elif ch == '[':
            depth_brk += 1
        elif ch == ']':
            depth_brk -= 1
        elif ch == '{':
            depth_brc += 1
        elif ch == '}':
            depth_brc -= 1
        elif ch == ',' and depth_par == depth_brk == depth_brc == 0:
            return s[:i].strip()

    return s.strip()

def extract_input_output(random_test):
    line = random_test.strip()

    # ---------- Case 1: equality ----------
    m_eq = re.search(r"assert\s+candidate\((.*?)\)\s*==\s*(.*)", line)
    if m_eq:
        input_expr = m_eq.group(1).strip()
        after_eq = m_eq.group(2).strip()
        output_expr = _cut_at_top_level_comma(after_eq)  # your helper handles trailing ", msg"
        print(f'input_expr: {input_expr}')
        print(f'output_expr: {output_expr}')
        return input_expr, output_expr

    # ---------- Case 2: tolerance: assert abs(candidate(INPUT) - OUTPUT) < tol ----------
    # parse by scanning to balance parentheses inside abs(...)
    key_abs = "assert"
    pos = line.find(key_abs)
    if pos == -1:
        raise ValueError(f"Could not parse assert line: {line}")
    # find "abs("
    m_abs = re.search(r"\babs\s*\(", line[pos:])
    if not m_abs:
        raise ValueError(f"Could not parse tolerance assert: {line}")
    abs_start = pos + m_abs.end()  # index just after '(' of abs(

    # next must be candidate( ... )
    m_cand = re.search(r"\s*candidate\s*\(", line[abs_start:])
    if not m_cand:
        raise ValueError(f"No candidate(…) inside abs(): {line}")
    cand_open = abs_start + m_cand.end() - 1  # index at '(' after 'candidate'

    # parse inside candidate(...) with balanced parens/quotes
    def _scan_until_matching_paren(s, start_idx):
        depth = 1
        i = start_idx + 1
        in_str = False
        quote = None
        escape = False
        while i < len(s):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
            else:
                if ch in ("'", '"'):
                    in_str = True
                    quote = ch
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return i  # index of matching ')'
            i += 1
        raise ValueError("Unbalanced parentheses in candidate(...)")

    cand_close = _scan_until_matching_paren(line, cand_open)
    input_expr = line[cand_open + 1:cand_close].strip()

    # after candidate(...), expect ' - ' then OUTPUT until the closing ')' of abs(...)
    i = cand_close + 1
    # skip spaces
    while i < len(line) and line[i].isspace():
        i += 1
    if i >= len(line) or line[i] != '-':
        raise ValueError(f"Expected '-' after candidate(...): {line}")
    i += 1
    while i < len(line) and line[i].isspace():
        i += 1
    out_start = i

    # find end of abs(...) by balancing parentheses from the '(' after abs
    def _scan_abs_close(s, open_idx):
        depth = 1  # already after '(' of abs(
        i = open_idx
        in_str = False
        quote = None
        escape = False
        while i < len(s):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == quote:
                    in_str = False
                    quote = None
            else:
                if ch in ("'", '"'):
                    in_str = True
                    quote = ch
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        raise ValueError("Unbalanced parentheses in abs(...)")

    abs_close = _scan_abs_close(line, abs_start)
    output_expr = line[out_start:abs_close].strip()  # everything between '-' and the closing ')' of abs

    print(f'input_expr: {input_expr}')
    print(f'output_expr: {output_expr}')
    return input_expr, output_expr


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


def generate_prompt(entry: dict, vanilla: str):
    tests = entry['test']
    random_test = choose_random_test(tests, seed=1234)
    print(f'random_test: {random_test}')
    PROGRAM_VANILLA = entry['canonical_solution']
    PROGRAM = build_program(entry)
    # print(f'function to see structure: {PROGRAM}')
    INPUT,OUTPUT = extract_input_output(random_test)

    if vanilla:
        vanilla_script = (
            f"You are an AI programming assistant, utilizing the DeepSeek Coder model, "
            f"developed by DeepSeek Company, and you only answer questions related to computer science.\n"
            f"For politically sensitive questions, security and privacy issues, and other "
            f"non-computer science questions, you will refuse to answer.\n\n"
            f"### Instruction:\n"
            f"If the string is {INPUT}, what will the following code return?\n"
            f"The return value prediction must be enclosed between [Output] and [/Output] tags.\n"
            f"For example : [Output]prediction[/Output].\n\n"
            f"{PROGRAM_VANILLA}\n\n"
            f"### Response:"
        )
        return {"prompt_script": vanilla_script, "output_expected": OUTPUT}

    prompt_script = (
        f"### Instruction:\n"
        f"You are an AI programming assistant.\n"
        f"Predict the exact return value when the function is evaluated with the input.\n\n"
        f"If the input is {INPUT}, what will the following function return?\n\n"
        f"{PROGRAM}\n\n"
        f"The return value prediction must be enclosed between [Output] and [/Output] tags.\n"
        f"For example : [Output]True[/Output].\n\n"
        f"### Response:"
    )
    return {"prompt_script":prompt_script, "output_expected":OUTPUT}

def normalize(s: str):
  if s is None:
    return None
  s = s.strip()

  # collapse whitespace
  s = re.sub(r"\s+", " ", s)

  # normalize simple containers / commas
  s = re.sub(r"\[\s+", "[", s)
  s = re.sub(r"\s+\]", "]", s)
  s = re.sub(r"\(\s+", "(", s)
  s = re.sub(r"\s+\)", ")", s)
  s = re.sub(r",\s+", ",", s)

  # common literal normalizations
  low = s.lower()
  if low == "yes": s = "Yes"
  if low == "no":  s = "No"
  return s

def extract_output_actual(response: str, expected_expr: str | None = None):
  # find the last [Output]...[/Output] in case the model printed more than one
    matches = re.findall(r"\[Output\](.*?)\[/Output\]", response, flags=re.S | re.I)
    if not matches:
        return None

    raw = matches[-1].strip()

    # if the expected is a quoted string but model forgot quotes, add them
    if expected_expr:
        exp = expected_expr.strip()
        exp_is_quoted = re.match(r"""^(['"]).*\1$""", exp) is not None
        raw_is_quoted = re.match(r"""^(['"]).*\1$""", raw) is not None
        if exp_is_quoted and not raw_is_quoted:
            raw = '"' + raw.replace("\\", "\\\\").replace('"', '\\"') + '"'

    val = normalize(raw)
    print(f'output_actual: {val}')
    return val

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
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

    # Generation/batching often needs a pad_token_id; setting it to eos is a common, safe fallback.
    # Prevents warnings like “Setting pad_token_id to eos_token_id for open-end generation.”
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for entry in dataset:
        pack = generate_prompt(entry, vanilla)
        prompt = pack["prompt_script"]
        output_expected = normalize(pack["output_expected"])
        
        # Stronger version of no_grad()—turns off autograd and enables some extra runtime wins.
        with torch.inference_mode(): 
          # Encode the prompt into input IDs/attention masks and move them to the same device as the model.
          enc = tokenizer(prompt, return_tensors="pt")
          enc = {k: v.to(model.device) for k, v in enc.items()}

          # Tokenize your custom stop sequence (multi-token). We’ll stop when this exact subsequence appears.
          stop_ids = tokenizer.encode('[/Output]', add_special_tokens=False)
          ep = entry.get("entry_point")
          bad_words = ["candidate", "function", "Input", "==", "```", "'''", "`", "print", "def", "class"]
          if ep:
              bad_words.append(ep)  # block the function identifier
          bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_words]

          # Custom stopping criteria that scans the generated continuation for [/Output] and halts when found.
          # scans the generated part only (see self.start below) for the exact token subsequence [/Output].
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
              max_new_tokens=200,
              stopping_criteria=criteria,
              # no_repeat_ngram_size=3,
              # temperature=0.0, Dont need since do_sample is false
              bad_words_ids=bad_words_ids,
              do_sample=False,
              pad_token_id=tokenizer.eos_token_id,
          )
          # The tensor returned by generate contains prompt + completion.
          # Slicing from start_idx: extracts only the new tokens the model produced.
          gen_ids = outputs[0][start_idx:]
          response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # TODO: process the response and save it to results
        output_actual = extract_output_actual(response, expected_expr=pack["output_expected"])
        verdict = False
        verdict = (output_expected == output_actual)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
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
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
