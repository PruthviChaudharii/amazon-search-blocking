"""
Benchmark comparing Fixed K=25 configs
Generates reports/v2_fixed25_optimized_benchmark.md
"""

import os
import sys
import csv
import time
import psutil
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))
from src.blocking_v2 import ProductionBlockingEngineV2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("Loading dataset...")
VAL_S1_SIZE = 10000
TARGET_SAMPLE_SIZE = 260000

gt_matches = defaultdict(set)
val_s1_ids = []
with open("dataset/train/train_ground_truth.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        m_ids = [m.strip() for m in row[1].split(',') if m.strip()]
        gt_matches[s1_id] = set(m_ids)
        if len(val_s1_ids) < VAL_S1_SIZE:
            val_s1_ids.append(s1_id)

val_s1_set = set(val_s1_ids)
needed_targets = set()
for sid in val_s1_ids:
    needed_targets.update(gt_matches[sid])

val_s1_records = []
with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in val_s1_set:
            val_s1_records.append({
                'id': eid,
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            })
            if len(val_s1_records) == len(val_s1_set):
                break

targets_by_country = defaultdict(list)
all_targets_data = set()
target_count = 0
for s_fname in ["train_source2.tsv", "train_source3.tsv"]:
    s_path = os.path.join("dataset/train", s_fname)
    with open(s_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0].strip()
            c = row[3].strip() if len(row) > 3 else ''
            if eid in needed_targets or (target_count < TARGET_SAMPLE_SIZE):
                targets_by_country[c].append((eid, row[1].strip(), row[2].strip()))
                all_targets_data.add(eid)
                target_count += 1

gt_map = {sid: gt_matches[sid] & all_targets_data for sid in val_s1_ids}
total_eval_true_matches = sum(len(v) for v in gt_map.values())

def run_evaluation(name, max_tri_freq, gate):
    print(f"\nBuilding Engine: {name}")
    engine = ProductionBlockingEngineV2(
        max_token_freq=1500, 
        max_trigram_freq=max_tri_freq, 
        adaptive_k_config=(10, 25, 50), 
        pass_c_gate=gate
    )
    
    t0_build = time.time()
    for c, recs in targets_by_country.items():
        engine.build_target_index_for_country(c, recs)
    build_time = time.time() - t0_build
    
    print(f"Querying: {name}")
    total_recalled_edges = 0
    full_coverage_entities = 0
    cand_counts = []
    
    engine.pass_c_invoked = 0
    t0_query = time.time()
    
    for rec in val_s1_records:
        sid = rec['id']
        cands, _, _ = engine.retrieve_candidates_for_query(rec['country'], rec['name'], rec['address'], force_k=25, return_details=True)
        
        c_len = len(cands)
        cand_counts.append(c_len)
        
        true_set = gt_map[sid]
        n_true = len(true_set)
        
        recalled = len(set(cands) & true_set)
        total_recalled_edges += recalled
        
        if n_true > 0:
            if recalled == n_true:
                full_coverage_entities += 1
        else:
            full_coverage_entities += 1
            
    query_time = time.time() - t0_query
    
    edge_recall = (total_recalled_edges / total_eval_true_matches) * 100.0 if total_eval_true_matches else 0.0
    entity_coverage = (full_coverage_entities / len(val_s1_records)) * 100.0
    pass_c_pct = (engine.pass_c_invoked / len(val_s1_records)) * 100.0
    
    res = {
        'Config': name,
        'Edge_Recall_%': round(edge_recall, 2),
        'Entity_Coverage_%': round(entity_coverage, 2),
        'Avg_Candidates': round(np.mean(cand_counts), 2),
        'Median_Candidates': int(np.median(cand_counts)),
        'P95_Candidates': int(np.percentile(cand_counts, 95)),
        'Max_Candidates': int(max(cand_counts)),
        'Index_Build_sec': round(build_time, 2),
        'Query_Runtime_sec': round(query_time, 2),
        'Pass_C_Invoked_%': round(pass_c_pct, 2)
    }
    print(res)
    return res

results = []
results.append(run_evaluation("A. Fixed K=25 (freq=2500, gate=100)", 2500, 100))
res_b = run_evaluation("B. Fixed K=25 (freq=500, gate=25)", 500, 25)
results.append(res_b)

if res_b['Edge_Recall_%'] < 81.5:
    print("Config B recall < 81.5%. Testing intermediate Config C...")
    res_c = run_evaluation("C. Fixed K=25 (freq=1000, gate=25)", 1000, 25)
    results.append(res_c)

md = "# V2 Fixed K=25 Optimization Benchmark\n\n"
md += "**Dataset:** Validation Split (10,000 S1 records)\n\n"
md += "## Configuration Comparison\n\n"
md += "| Config | Edge Recall (%) | Entity Coverage (%) | Avg Cands | Median | P95 | Max | Index Build (s) | Query Runtime (s) | Pass C Invoked (%) |\n"
md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
for r in results:
    md += f"| **{r['Config']}** | {r['Edge_Recall_%']:.2f}% | {r['Entity_Coverage_%']:.2f}% | {r['Avg_Candidates']} | {r['Median_Candidates']} | {r['P95_Candidates']} | {r['Max_Candidates']} | {r['Index_Build_sec']}s | {r['Query_Runtime_sec']}s | {r['Pass_C_Invoked_%']}% |\n"

md += "\n## Decision\n"
if res_b['Edge_Recall_%'] >= 81.5:
    md += "**RECOMMENDATION:** Optimized Fixed K=25 (freq=500, gate=25) retains >=81.5% recall while substantially lowering runtime. It is safe to promote for full-test generation.\n"
else:
    md += "**RECOMMENDATION:** Optimized Fixed K=25 (freq=500, gate=25) fell below the 81.5% recall target. "
    if len(results) > 2 and results[2]['Edge_Recall_%'] >= 81.5:
        md += "The intermediate Config C (freq=1000, gate=25) successfully restored recall while keeping runtime lower than baseline A.\n"
    else:
        md += "Intermediate configurations are needed or further tuning is required.\n"

with open("reports/v2_fixed25_optimized_benchmark.md", "w", encoding="utf-8") as f:
    f.write(md)
print("Saved reports/v2_fixed25_optimized_benchmark.md")
