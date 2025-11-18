import json

def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def compute_accuracy(entries):
    return sum(1 for x in entries if x["is_correct"]) , len(entries)

def compare(vanilla, crafted):
    print("=== MP3 Task 2 Evaluation ===")
    print(f"Vanilla samples:  {len(vanilla)}")
    print(f"Crafted samples:  {len(crafted)}\n")

    v_pass, v_total = compute_accuracy(vanilla)
    c_pass, c_total = compute_accuracy(crafted)

    print(f"Vanilla correct: {v_pass}/{v_total}   ({v_pass/v_total*100:.1f}%)")
    print(f"Crafted correct: {c_pass}/{c_total}   ({c_pass/c_total*100:.1f}%)\n")

    improvement = ((c_pass - v_pass) / max(v_pass, 1)) * 100
    print(f"Improvement: {improvement:.1f}%\n")

    print("=== Per-task comparison ===")
    print("task_id | vanilla | crafted")
    print("---------------------------------------")

    v_map = {x["task_id"]: x for x in vanilla}
    c_map = {x["task_id"]: x for x in crafted}

    for task_id in sorted(v_map.keys()):
        v = v_map[task_id]["is_correct"]
        c = c_map[task_id]["is_correct"]
        print(f"{task_id:12} | {str(v):7} | {str(c):7}")

    print("\nDone.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate_task2_results.py <vanilla.jsonl> <crafted.jsonl>")
        exit(1)

    vanilla_results = load_jsonl(sys.argv[1])
    crafted_results = load_jsonl(sys.argv[2])
    compare(vanilla_results, crafted_results)
