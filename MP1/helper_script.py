import json

# Update this with your actual seed value
SEED = "270992647952172920073750662102343813773"

# File paths
base_file = f"base_prompt_{SEED}.jsonl_results.jsonl"
base_proc_file = f"base_prompt_processed_{SEED}.jsonl_results.jsonl"
inst_file = f"instruct_prompt_{SEED}.jsonl_results.jsonl"
inst_proc_file = f"instruct_prompt_processed_{SEED}.jsonl_results.jsonl"

def load_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            results[entry["task_id"]] = "Pass" if entry.get("passed") else "Fail"
    return results

# Load all four sets
base = load_results(base_file)
base_proc = load_results(base_proc_file)
inst = load_results(inst_file)
inst_proc = load_results(inst_proc_file)

# Collect all problem IDs
problem_ids = sorted(base.keys())

# Build Markdown table
lines = []
lines.append("| Problem_ID | Base | Base_Processed | Instruct | Instruct_Processed |")
lines.append("|------------|------|----------------|----------|---------------------|")
for pid in problem_ids:
    lines.append(f"| {pid} | {base[pid]} | {base_proc[pid]} | {inst[pid]} | {inst_proc[pid]} |")

# Add totals
lines.append("")
lines.append("**Totals:**")
lines.append(f"- Base: {sum(v=='Pass' for v in base.values())}/20")
lines.append(f"- Base_Processed: {sum(v=='Pass' for v in base_proc.values())}/20")
lines.append(f"- Instruct: {sum(v=='Pass' for v in inst.values())}/20")
lines.append(f"- Instruct_Processed: {sum(v=='Pass' for v in inst_proc.values())}/20")

# Print to terminal
print("\n".join(lines))

# Save to file
with open("comparison_table.md", "w") as f:
    f.write("\n".join(lines))

print("\nTable saved to comparison_table.md")
