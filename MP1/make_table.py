import jsonlines
import pandas as pd


base_file = "base_prompt_272469978654662835334249905214610999505.jsonl_results.jsonl"
base_processed_file = "base_prompt_processed_272469978654662835334249905214610999505.jsonl_results.jsonl"
instruct_file = "instruct_prompt_272469978654662835334249905214610999505.jsonl_results.jsonl"
instruct_processed_file = "instruct_prompt_processed_272469978654662835334249905214610999505.jsonl_results.jsonl"

def load_results(path):
    data = {}
    with jsonlines.open(path) as reader:
        for obj in reader:
            data[obj["task_id"]] = "pass" if obj["passed"] else "fail"
    return data


base = load_results(base_file)
base_proc = load_results(base_processed_file)
instruct = load_results(instruct_file) if instruct_file else {}
instruct_proc = load_results(instruct_processed_file) if instruct_processed_file else {}


task_ids = sorted(set(base) | set(base_proc) | set(instruct) | set(instruct_proc))
rows = []
for tid in task_ids:
    rows.append({
        "Problem_ID": tid,
        "base_results": base.get(tid, ""),
        "base_results_processed": base_proc.get(tid, ""),
        "instruct_results": instruct.get(tid, ""),
        "instruct_results_processed": instruct_proc.get(tid, ""),
    })

df = pd.DataFrame(rows)

df.to_csv("comparison_table.csv", index=False)
with open("comparison_table.md", "w") as f:
    f.write(df.to_markdown(index=False))

print(" comparison_table.csv and comparison_table.md written")