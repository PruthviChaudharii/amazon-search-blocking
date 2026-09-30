"""
Business Entity Resolution - Multi-Pass Production Blocking Engine
Author: ML Challenge Team
License: Apache 2.0 / MIT Compliant

This module implements a scalable, memory-conscious, multi-pass candidate generation (blocking) pipeline:
1. Country-based strict partitioning (empirically 100% consistent with ground truth).
2. Robust multi-representation name and address normalization.
3. Pass 1: Exact Normalized Core-Name Inverted Index.
4. Pass 2: High-Recall Informative Name Token & Bi-gram Inverted Index with IDF weighting.
5. Pass 3: Address Anchor Inverted Index (Postal Code, Building/Door Number, Locality).
6. Evidence-based Candidate Scoring & Top-K Pruning with deterministic tie-breaking.
7. Streaming generation to ensure bounded memory on multi-million row datasets.
"""

import os
import sys
import re
import csv
import math
import time
import gc
import psutil
import unicodedata
from collections import defaultdict, Counter
import numpy as np

# Standard legal & organizational stopwords across multiple languages (English, French, Indian business suffixes)
LEGAL_STOPWORDS = {
    # English & General
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'gmbh', 'plc', 'enterprises', 'enterprise',
    'associates', 'group', 'holdings', 'holding', 'industries', 'industry', 'international',
    'services', 'service', 'solutions', 'solution', 'consulting', 'consultancy',
    'the', 'and', 'of', 'in', 'at', 'on', 'for', 'by', 'with', 'an', 'a',
    # French
    'sarl', 'sa', 'sas', 'sasu', 'sci', 'snc', 'eurl', 'ets', 'etablissements',
    'societe', 'cie', 'france', 'fr', 'de', 'du', 'des', 'la', 'le', 'les',
    # Common business generic qualifiers
    'store', 'stores', 'shop', 'centre', 'center', 'agency', 'traders', 'trading'
}

COMMON_ADDRESS_STOPWORDS = {
    'near', 'opp', 'opposite', 'behind', 'beside', 'road', 'rd', 'street', 'st',
    'avenue', 'ave', 'lane', 'ln', 'floor', 'fl', 'block', 'blk', 'sector', 'sec',
    'plot', 'no', 'number', 'door', 'flat', 'building', 'bldg', 'complex', 'cross',
    'main', 'nagar', 'colony', 'layout', 'dist', 'district', 'post', 'po',
    'rue', 'avenue', 'boulevard', 'bd', 'chemin', 'route', 'allee', 'place'
}

RE_DOMAIN = re.compile(r'\.(com|org|net|in|co|io|fr|gov|edu|biz|info)\b', re.IGNORECASE)
RE_AMP = re.compile(r'[\&\+]')
RE_NON_ALNUM = re.compile(r'[^a-z0-9\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\s]')
RE_POSTAL = re.compile(r'\b[0-9]{4,6}\b')
RE_HOUSE = re.compile(r'\b[0-9]+[a-z]?\b')

def get_process_memory_mb():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def unicode_normalize(s: str) -> str:
    """Normalize unicode characters (NFKD)."""
    if not s:
        return ""
    return unicodedata.normalize('NFKD', s)

def clean_text_general(s: str) -> str:
    """Lowercase, strip domain extensions, replace punctuation with spaces."""
    if not s:
        return ""
    s = s.lower()
    if '.' in s:
        s = RE_DOMAIN.sub('', s)
    if '&' in s or '+' in s:
        s = RE_AMP.sub(' and ', s)
    s = RE_NON_ALNUM.sub(' ', s)
    return ' '.join(s.split())

def extract_name_representations(name: str):
    """Generate multi-level name representations for robust multi-pass matching."""
    norm_name = clean_text_general(unicode_normalize(name))
    raw_tokens = norm_name.split()
    core_tokens = [w for w in raw_tokens if w not in LEGAL_STOPWORDS and len(w) >= 2]
    
    core_name = ' '.join(core_tokens)
    
    bigrams = []
    if len(core_tokens) >= 2:
        for i in range(len(core_tokens) - 1):
            bigrams.append(f"{core_tokens[i]}_{core_tokens[i+1]}")
            
    prefix2 = f"{core_tokens[0]}_{core_tokens[1]}" if len(core_tokens) >= 2 else (core_tokens[0] if core_tokens else "")
    
    return {
        'norm_name': norm_name,
        'core_name': core_name,
        'core_tokens': core_tokens,
        'bigrams': bigrams,
        'prefix2': prefix2
    }

def extract_address_representations(address: str):
    """Extract postal code, building numbers, and locality tokens from address."""
    norm_addr = clean_text_general(unicode_normalize(address))
    
    postal_matches = RE_POSTAL.findall(norm_addr)
    postal_code = postal_matches[0] if postal_matches else ""
    
    house_matches = RE_HOUSE.findall(norm_addr)
    house_nums = [h for h in house_matches if h != postal_code]
    house_num = house_nums[0] if house_nums else ""
    
    addr_tokens = [w for w in norm_addr.split() if len(w) >= 3 and w not in COMMON_ADDRESS_STOPWORDS and w not in LEGAL_STOPWORDS]
    
    return {
        'norm_addr': norm_addr,
        'postal_code': postal_code,
        'house_num': house_num,
        'addr_tokens': addr_tokens
    }


class ProductionBlockingEngine:
    """
    High-performance, memory-efficient multi-pass blocking engine.
    Partitions search space by country, builds inverted indices over target records (S2 + S3),
    and scores candidates using multiple evidence channels.
    """
    
    def __init__(self, max_token_freq: int = 1500, top_k: int = 25):
        self.max_token_freq = max_token_freq
        self.top_k = top_k
        
        self.target_data = defaultdict(dict)
        self.pass1_exact_name_index = defaultdict(lambda: defaultdict(list))
        self.pass2_token_index = defaultdict(lambda: defaultdict(list))
        self.pass2_bigram_index = defaultdict(lambda: defaultdict(list))
        self.pass3_pin_house_index = defaultdict(lambda: defaultdict(list))
        self.pass3_house_token_index = defaultdict(lambda: defaultdict(list))
        self.token_idfs = defaultdict(dict)
        self.country_target_counts = Counter()

    def build_target_index_for_country(self, country: str, records: list):
        """
        Build all 3 inverted index passes for a specific country partition.
        records: list of (target_entity_id, business_name, business_address)
        """
        t0 = time.time()
        n_targets = len(records)
        self.country_target_counts[country] = n_targets
        
        token_freqs = Counter()
        c_targets = {}
        c_pass1 = defaultdict(list)
        c_pass2_tokens = defaultdict(list)
        c_pass2_bigrams = defaultdict(list)
        c_pass3_pin_house = defaultdict(list)
        c_pass3_house_tok = defaultdict(list)
        
        for tid, name, addr in records:
            name_rep = extract_name_representations(name)
            addr_rep = extract_address_representations(addr)
            
            c_targets[tid] = {
                'core_name': name_rep['core_name'],
                'core_tokens': set(name_rep['core_tokens']),
                'postal_code': addr_rep['postal_code'],
                'house_num': addr_rep['house_num'],
                'norm_addr': addr_rep['norm_addr']
            }
            
            # Pass 1: Exact Core Name
            if name_rep['core_name']:
                c_pass1[name_rep['core_name']].append(tid)
                
            # Pass 2: Name Tokens & Bi-grams
            for tok in name_rep['core_tokens']:
                token_freqs[tok] += 1
                c_pass2_tokens[tok].append(tid)
                
            for bg in name_rep['bigrams']:
                c_pass2_bigrams[bg].append(tid)
                
            # Pass 3: Address Anchors
            pin = addr_rep['postal_code']
            h_num = addr_rep['house_num']
            if pin and h_num:
                c_pass3_pin_house[f"{pin}_{h_num}"].append(tid)
            if h_num and addr_rep['addr_tokens']:
                loc_tok = addr_rep['addr_tokens'][-1]
                c_pass3_house_tok[f"{h_num}_{loc_tok}"].append(tid)

        # Compute IDFs and prune overly generic tokens
        pruned_tokens = {}
        c_idfs = {}
        for tok, freq in token_freqs.items():
            if freq <= self.max_token_freq:
                pruned_tokens[tok] = c_pass2_tokens[tok]
                c_idfs[tok] = math.log(1.0 + (n_targets / (freq + 1.0)))
                
        self.target_data[country] = c_targets
        self.pass1_exact_name_index[country] = c_pass1
        self.pass2_token_index[country] = pruned_tokens
        self.pass2_bigram_index[country] = {k: v for k, v in c_pass2_bigrams.items() if len(v) <= self.max_token_freq}
        self.pass3_pin_house_index[country] = c_pass3_pin_house
        self.pass3_house_token_index[country] = {k: v for k, v in c_pass3_house_tok.items() if len(v) <= 500}
        self.token_idfs[country] = c_idfs
        
        elapsed = time.time() - t0
        mem_mb = get_process_memory_mb()
        print(f"  [Index Built] Country: '{country}' | Targets: {n_targets:,} | Indexed Tokens: {len(pruned_tokens):,} | Time: {elapsed:.2f}s | Current RAM: {mem_mb:.1f} MB")

    def retrieve_candidates_for_query(self, country: str, name: str, address: str, top_k: int = None) -> list:
        """
        Query the 3 blocking passes, score candidate targets, and return top-K target IDs.
        """
        if top_k is None:
            top_k = self.top_k
            
        c_targets = self.target_data.get(country)
        if not c_targets:
            return []
            
        name_rep = extract_name_representations(name)
        addr_rep = extract_address_representations(address)
        
        core_name = name_rep['core_name']
        core_tokens = name_rep['core_tokens']
        bigrams = name_rep['bigrams']
        pin = addr_rep['postal_code']
        h_num = addr_rep['house_num']
        addr_tokens = addr_rep['addr_tokens']
        
        candidate_scores = Counter()
        
        # Evidence 1: Pass 1 Exact Normalized Core Name (+5.0)
        if core_name:
            exact_matches = self.pass1_exact_name_index[country].get(core_name)
            if exact_matches:
                for tid in exact_matches:
                    candidate_scores[tid] += 5.0
                    
        # Evidence 2: Pass 2 Informative Name Tokens (IDF-weighted)
        c_token_idx = self.pass2_token_index[country]
        c_idfs = self.token_idfs[country]
        for tok in core_tokens:
            if tok in c_token_idx:
                w = c_idfs.get(tok, 1.0)
                score_weight = min(3.0, max(0.8, w * 0.4))
                for tid in c_token_idx[tok]:
                    candidate_scores[tid] += score_weight
                    
        # Evidence 3: Pass 2 Bi-grams (+3.0)
        c_bg_idx = self.pass2_bigram_index[country]
        for bg in bigrams:
            if bg in c_bg_idx:
                for tid in c_bg_idx[bg]:
                    candidate_scores[tid] += 3.0
                    
        # Evidence 4: Pass 3 Address Anchor PIN + House Num (+4.0)
        if pin and h_num:
            ph_key = f"{pin}_{h_num}"
            pin_matches = self.pass3_pin_house_index[country].get(ph_key)
            if pin_matches:
                for tid in pin_matches:
                    candidate_scores[tid] += 4.0
                    
        # Evidence 5: Pass 3 Locality + House Num (+2.5)
        if h_num and addr_tokens:
            loc_tok = addr_tokens[-1]
            ht_key = f"{h_num}_{loc_tok}"
            ht_matches = self.pass3_house_token_index[country].get(ht_key)
            if ht_matches:
                for tid in ht_matches:
                    candidate_scores[tid] += 2.5
                    
        if not candidate_scores:
            return []
            
        # Deterministic Top-K Pruning: score DESC, entity_id ASC
        if len(candidate_scores) <= top_k * 3:
            sorted_cands = sorted(candidate_scores.keys(), key=lambda tid: (-candidate_scores[tid], tid))
            return sorted_cands[:top_k]
        else:
            top_items = candidate_scores.most_common(top_k * 2)
            sorted_cands = sorted(top_items, key=lambda pair: (-pair[1], pair[0]))
            return [tid for tid, _ in sorted_cands[:top_k]]

    def clear_country(self, country: str):
        """Free memory for a specific country once processing is complete."""
        self.target_data.pop(country, None)
        self.pass1_exact_name_index.pop(country, None)
        self.pass2_token_index.pop(country, None)
        self.pass2_bigram_index.pop(country, None)
        self.pass3_pin_house_index.pop(country, None)
        self.pass3_house_token_index.pop(country, None)
        self.token_idfs.pop(country, None)
        gc.collect()


def generate_test_candidates(
    test_dir: str = "dataset/test",
    output_path: str = "output/candidate_pairs.tsv",
    top_k: int = 25,
    max_token_freq: int = 1500
):
    """
    Production candidate generation pipeline for the test set.
    Processes data country-by-country to maintain bounded RAM usage (~3-4GB max).
    Produces output/candidate_pairs.tsv with exact columns:
      source1_entity_id\tcandidate_entity_ids
    """
    t_start = time.time()
    print("=" * 60)
    print("PRODUCTION BLOCKING ENGINE: Generating Test Candidates")
    print(f"Top-K: {top_k} | Max Token Frequency: {max_token_freq}")
    print(f"Test Directory: {test_dir}")
    print(f"Output Path: {output_path}")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Read Test Source 1 records and index their row orders
    print("\n[Step 1/4] Reading Test Source 1 records...")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s1_order = []
    s1_by_country = defaultdict(list)
    
    with open(s1_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for i, row in enumerate(reader):
            eid = row[0].strip()
            name = row[1].strip()
            addr = row[2].strip()
            ctry = row[3].strip() if len(row) > 3 else ''
            s1_order.append(eid)
            s1_by_country[ctry].append((eid, name, addr))
            
    total_s1 = len(s1_order)
    print(f"Loaded {total_s1:,} Test Source 1 records across {len(s1_by_country)} countries:")
    for c, recs in s1_by_country.items():
        print(f"  - '{c}': {len(recs):,} records ({len(recs)/total_s1*100:.2f}%)")

    # Countries to process
    countries = list(s1_by_country.keys())
    
    # Temporary directory for per-country candidate TSVs
    temp_dir = os.path.join("output", "temp_cands")
    os.makedirs(temp_dir, exist_ok=True)
    country_cand_files = {}

    total_candidates_all = 0
    country_cand_counts = Counter()
    source_counts = Counter()
    cand_lengths = []
    zero_cand_count = 0
    peak_memory = 0.0

    # 2. Process each country partition independently
    for c_idx, country in enumerate(countries, start=1):
        print(f"\n[Step 2/4] Processing Country {c_idx}/{len(countries)}: '{country}'...")
        country_t0 = time.time()
        
        # Load targets for this country from test_source2.tsv and test_source3.tsv
        c_targets = []
        for s_fname in ["test_source2.tsv", "test_source3.tsv"]:
            s_path = os.path.join(test_dir, s_fname)
            print(f"  Scanning {s_fname} for country '{country}'...")
            with open(s_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for row in reader:
                    c = row[3].strip() if len(row) > 3 else ''
                    if c == country:
                        c_targets.append((row[0].strip(), row[1].strip(), row[2].strip()))
                        
        print(f"  Loaded {len(c_targets):,} target records for '{country}'.")
        
        # Build inverted index
        engine = ProductionBlockingEngine(max_token_freq=max_token_freq, top_k=top_k)
        engine.build_target_index_for_country(country, c_targets)
        del c_targets
        gc.collect()
        
        # Query S1 records for this country
        c_s1_recs = s1_by_country[country]
        c_out_path = os.path.join(temp_dir, f"cands_{country}.tsv")
        country_cand_files[country] = c_out_path
        
        print(f"  Querying candidates for {len(c_s1_recs):,} S1 entities in '{country}'...")
        q_t0 = time.time()
        
        with open(c_out_path, 'w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f, delimiter='\t')
            for eid, name, addr in c_s1_recs:
                cands = engine.retrieve_candidates_for_query(country, name, addr, top_k=top_k)
                c_len = len(cands)
                cand_lengths.append(c_len)
                total_candidates_all += c_len
                country_cand_counts[country] += c_len
                
                if c_len == 0:
                    zero_cand_count += 1
                else:
                    for cid in cands:
                        if cid.startswith('S2-'):
                            source_counts['S2'] += 1
                        elif cid.startswith('S3-'):
                            source_counts['S3'] += 1
                            
                cand_str = ','.join(cands)
                writer.writerow([eid, cand_str])

        q_elapsed = time.time() - q_t0
        print(f"  Completed '{country}' queries in {q_elapsed:.2f}s ({len(c_s1_recs)/q_elapsed:.1f} S1/sec).")
        
        # Track memory and clean up
        cur_mem = get_process_memory_mb()
        peak_memory = max(peak_memory, cur_mem)
        engine.clear_country(country)
        del engine
        gc.collect()
        print(f"  Finished '{country}' partition in {time.time() - country_t0:.2f}s. Memory reclaimed.")

    # 3. Assemble final candidate_pairs.tsv in the exact original order of test_source1.tsv
    print("\n[Step 3/4] Assembling final candidate_pairs.tsv in original S1 row order...")
    # Load per-country candidate mappings
    s1_to_cand_str = {}
    for country, c_path in country_cand_files.items():
        with open(c_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row:
                    s1_to_cand_str[row[0].strip()] = row[1].strip() if len(row) > 1 else ""

    with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            cand_str = s1_to_cand_str.get(sid, "")
            out_f.write(f"{sid}\t{cand_str}\n")

    # Clean up temp files
    for c_path in country_cand_files.values():
        if os.path.isfile(c_path):
            os.remove(c_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    total_time = time.time() - t_start
    print(f"\n[Step 4/4] Final Assembly Complete! File written to: {output_path}")
    print("=" * 60)
    print("BLOCKING EXECUTION SUMMARY REPORT:")
    print(f"  Total S1 Entities Processed:  {total_s1:,}")
    print(f"  Total Candidate Pairs:        {total_candidates_all:,}")
    print(f"  Mean Candidates / S1:         {np.mean(cand_lengths):.2f}")
    print(f"  Median Candidates / S1:       {int(np.median(cand_lengths))}")
    print(f"  P95 Candidates / S1:          {int(np.percentile(cand_lengths, 95))}")
    print(f"  P99 Candidates / S1:          {int(np.percentile(cand_lengths, 99))}")
    print(f"  Maximum Candidates / S1:      {max(cand_lengths)}")
    print(f"  Zero-Candidate Percentage:    {(zero_cand_count / total_s1) * 100:.3f}% ({zero_cand_count:,} entities)")
    print("  Candidates by Country:")
    for c, cnt in country_cand_counts.items():
        print(f"    - {c}: {cnt:,} candidates")
    print("  Candidates by Source:")
    for src, cnt in source_counts.items():
        print(f"    - {src}: {cnt:,} ({cnt/total_candidates_all*100:.2f}%)")
    print(f"  Total Runtime:                {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Peak Memory:                  {peak_memory:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    generate_test_candidates()
