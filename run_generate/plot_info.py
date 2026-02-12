"""
Entirely by Claude
"""

import json
import matplotlib.pyplot as plt
from collections import Counter

def plot_usage(log_file):
    expert_counts = Counter()
    
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "route":
                expert_counts.update(data["topk_ids"])
    
    experts = sorted(expert_counts.keys())
    counts = [expert_counts[e] for e in experts]

    plt.figure(figsize=(10, 6))
    plt.bar(experts, counts, color='skyblue', edgecolor='navy')
    plt.xlabel("Expert ID")
    plt.ylabel("Selection Count")
    plt.title("MoE Expert Usage Histogram")
    plt.xticks(experts, fontsize=7, rotation=90, ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("expert_hist.png")
    print("Histogram saved to expert_hist.png")

if __name__ == "__main__":
    plot_usage("data_hooked/vllm_moe_log.jsonl")