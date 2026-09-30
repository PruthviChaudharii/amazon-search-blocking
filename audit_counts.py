import os
import sys
import pandas as pd
import numpy as np

train_dir = "dataset/train"
test_dir = "dataset/test"

print("--- FILE SIZES ---")
for folder in [train_dir, test_dir]:
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            sz_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"{fpath}: {sz_mb:.2f} MB")

print("\n--- LINE / ROW COUNTS & BASIC PROFILE ---")
def fast_count(path):
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        count = sum(1 for _ in f)
    return header.strip().split('\t'), count

for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]:
    p = os.path.join(train_dir, f)
    header, cnt = fast_count(p)
    print(f"Train {f}: {cnt:,} rows | Columns: {header}")

for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    p = os.path.join(test_dir, f)
    header, cnt = fast_count(p)
    print(f"Test {f}: {cnt:,} rows | Columns: {header}")
