"""
Business Entity Resolution - Enhanced Multi-Pass Production Blocking Engine v2
Author: ML Challenge Team
License: Apache 2.0 / MIT Compliant

Improvements over v1:
- Character 3-gram retrieval (Pass C)
- Domain / Concatenated-name retrieval (Pass G)
- Expanded Address Anchors (House + Street)
- Adaptive Candidate Budgeting (K = 10, 25, 50, etc.)
- Multi-channel Evidence Scoring
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

# Standard legal & organizational stopwords
LEGAL_STOPWORDS = {
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'gmbh', 'plc', 'enterprises', 'enterprise',
    'associates', 'group', 'holdings', 'holding', 'industries', 'industry', 'international',
    'services', 'service', 'solutions', 'solution', 'consulting', 'consultancy',
    'the', 'and', 'of', 'in', 'at', 'on', 'for', 'by', 'with', 'an', 'a',
    'sarl', 'sa', 'sas', 'sasu', 'sci', 'snc', 'eurl', 'ets', 'etablissements',
    'societe', 'cie', 'france', 'fr', 'de', 'du', 'des', 'la', 'le', 'les',
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
RE_PHONE = re.compile(r'\b[0-9]{7,12}\b')

def get_process_memory_mb():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def unicode_normalize(s: str) -> str:
    if not s: return ""
    return unicodedata.normalize('NFKD', s)

def clean_text_general(s: str) -> str:
    if not s: return ""
    s = s.lower()
    if '.' in s:
        s = RE_DOMAIN.sub('', s)
    if '&' in s or '+' in s:
        s = RE_AMP.sub(' and ', s)
    s = RE_NON_ALNUM.sub(' ', s)
    return ' '.join(s.split())

def get_char_trigrams(text: str):
    s = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(s) < 3:
        return [s] if s else []
    return [s[i:i+3] for i in range(len(s) - 2)]

def get_concatenated_core_name(core_tokens):
    return ''.join(core_tokens)

def clean_domain_name(name: str):
    s = name.lower().strip()
    s = RE_DOMAIN.sub('', s)
    s = re.sub(r'^(https?://)?(www\.)?', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def extract_name_representations(name: str):
    raw_name_no_phone = RE_PHONE.sub('', name).strip()
    norm_name = clean_text_general(unicode_normalize(raw_name_no_phone))
    raw_tokens = norm_name.split()
    core_tokens = [w for w in raw_tokens if w not in LEGAL_STOPWORDS and len(w) >= 2]
    core_name = ' '.join(core_tokens)
    
    bigrams = []
    if len(core_tokens) >= 2:
        for i in range(len(core_tokens) - 1):
            bigrams.append(f"{core_tokens[i]}_{core_tokens[i+1]}")
            
    concat_name = get_concatenated_core_name(core_tokens)
    domain_name = clean_domain_name(name)
    
    return {
        'norm_name': norm_name,
        'core_name': core_name,
        'core_tokens': core_tokens,
        'bigrams': bigrams,
        'concat_name': concat_name,
        'domain_name': domain_name
    }

def extract_address_representations(address: str):
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


class ProductionBlockingEngineV2:
    def __init__(self, max_token_freq=1500, max_trigram_freq=2500, adaptive_k_config=(10, 25, 50), pass_c_gate=100):
        self.max_token_freq = max_token_freq
        self.max_trigram_freq = max_trigram_freq
        self.adaptive_k_config = adaptive_k_config # (low_k, med_k, high_k)
        self.pass_c_gate = pass_c_gate
        
        self.target_reps = defaultdict(dict)
        self.pass_a_core_name = defaultdict(lambda: defaultdict(list))
        self.pass_b_tokens = defaultdict(lambda: defaultdict(list))
        self.pass_b_bigrams = defaultdict(lambda: defaultdict(list))
        self.pass_c_char_trigrams = defaultdict(lambda: defaultdict(list))
        self.pass_e_pin_house = defaultdict(lambda: defaultdict(list))
        self.pass_e_house_loc = defaultdict(lambda: defaultdict(list))
        self.pass_e_house_street = defaultdict(lambda: defaultdict(list))
        self.pass_g_domain_concat = defaultdict(lambda: defaultdict(list))
        
        self.token_idfs = defaultdict(dict)
        self.trigram_idfs = defaultdict(dict)

    def build_target_index_for_country(self, country: str, records: list):
        t0 = time.time()
        n = len(records)
        token_freqs = Counter()
        trigram_freqs = Counter()
        
        c_pass_a = defaultdict(list)
        c_pass_b_tok = defaultdict(list)
        c_pass_b_bg = defaultdict(list)
        c_pass_c_tri = defaultdict(list)
        c_pass_e_ph = defaultdict(list)
        c_pass_e_hl = defaultdict(list)
        c_pass_e_hs = defaultdict(list)
        c_pass_g_dom = defaultdict(list)
        c_targets = {}
        
        for tid, name, addr in records:
            name_rep = extract_name_representations(name)
            addr_rep = extract_address_representations(addr)
            
            core_name = name_rep['core_name']
            core_tokens = name_rep['core_tokens']
            bigrams = name_rep['bigrams']
            concat_name = name_rep['concat_name']
            domain_name = name_rep['domain_name']
            
            pin = addr_rep['postal_code']
            h_num = addr_rep['house_num']
            addr_tokens = addr_rep['addr_tokens']
            
            c_targets[tid] = {
                'core_name': core_name,
                'pin': pin,
                'house_num': h_num
            }
            
            # Pass A: Exact Core Name
            if core_name:
                c_pass_a[core_name].append(tid)
                
            # Pass G: Domain & Concatenated Core Name
            if concat_name and len(concat_name) >= 4:
                c_pass_g_dom[concat_name].append(tid)
            if domain_name and domain_name != concat_name and len(domain_name) >= 4:
                c_pass_g_dom[domain_name].append(tid)
                
            # Pass B: Informative Tokens & Bigrams
            for tok in core_tokens:
                token_freqs[tok] += 1
                c_pass_b_tok[tok].append(tid)
            for bg in bigrams:
                c_pass_b_bg[bg].append(tid)
                
            # Pass C: Character 3-grams
            if core_name:
                tris = set(get_char_trigrams(core_name))
                for tri in tris:
                    trigram_freqs[tri] += 1
                    c_pass_c_tri[tri].append(tid)
                    
            # Pass E: Address Anchors
            if pin and h_num:
                c_pass_e_ph[f"{pin}_{h_num}"].append(tid)
            if h_num and addr_tokens:
                c_pass_e_hl[f"{h_num}_{addr_tokens[-1]}"].append(tid) # locality
                if len(addr_tokens) >= 2:
                    c_pass_e_hs[f"{h_num}_{addr_tokens[0]}"].append(tid) # street
                    
        pruned_tokens = {}
        c_idfs = {}
        for tok, f in token_freqs.items():
            if f <= self.max_token_freq:
                pruned_tokens[tok] = c_pass_b_tok[tok]
                c_idfs[tok] = math.log(1.0 + (n / (f + 1.0)))
                
        pruned_trigrams = {}
        c_tri_idfs = {}
        for tri, f in trigram_freqs.items():
            if f <= self.max_trigram_freq:
                pruned_trigrams[tri] = c_pass_c_tri[tri]
                c_tri_idfs[tri] = math.log(1.0 + (n / (f + 1.0)))
                
        self.pass_a_core_name[country] = c_pass_a
        self.pass_b_tokens[country] = pruned_tokens
        self.pass_b_bigrams[country] = {k: v for k, v in c_pass_b_bg.items() if len(v) <= self.max_token_freq}
        self.pass_c_char_trigrams[country] = pruned_trigrams
        self.pass_e_pin_house[country] = c_pass_e_ph
        self.pass_e_house_loc[country] = {k: v for k, v in c_pass_e_hl.items() if len(v) <= 500}
        self.pass_e_house_street[country] = {k: v for k, v in c_pass_e_hs.items() if len(v) <= 500}
        self.pass_g_domain_concat[country] = {k: v for k, v in c_pass_g_dom.items() if len(v) <= 500}
        self.token_idfs[country] = c_idfs
        self.trigram_idfs[country] = c_tri_idfs
        self.target_reps[country] = c_targets
        
        elapsed = time.time() - t0
        mem_mb = get_process_memory_mb()
        print(f"  [Index V2 Built] Country: '{country}' | Targets: {n:,} | Time: {elapsed:.2f}s | Current RAM: {mem_mb:.1f} MB")

    def retrieve_candidates_for_query(self, country: str, name: str, address: str, force_k: int = None, return_details: bool = False):
        if not self.target_reps.get(country):
            if return_details: return [], 0.0, 0
            return []
            
        name_rep = extract_name_representations(name)
        addr_rep = extract_address_representations(address)
        
        core_name = name_rep['core_name']
        core_tokens = name_rep['core_tokens']
        bigrams = name_rep['bigrams']
        concat_name = name_rep['concat_name']
        domain_name = name_rep['domain_name']
        
        pin = addr_rep['postal_code']
        h_num = addr_rep['house_num']
        addr_tokens = addr_rep['addr_tokens']
        
        candidate_scores = Counter()
        pass_matches_count = Counter()
        
        # Pass A
        if core_name in self.pass_a_core_name[country]:
            for tid in self.pass_a_core_name[country][core_name]:
                candidate_scores[tid] += 6.0
                pass_matches_count[tid] += 1
                
        # Pass G
        for dom_key in [concat_name, domain_name]:
            if dom_key and dom_key in self.pass_g_domain_concat[country]:
                for tid in self.pass_g_domain_concat[country][dom_key]:
                    candidate_scores[tid] += 4.5
                    pass_matches_count[tid] += 1
                    
        # Pass B (Tokens)
        c_tokens = self.pass_b_tokens[country]
        c_idfs = self.token_idfs[country]
        for tok in core_tokens:
            if tok in c_tokens:
                w = c_idfs.get(tok, 1.0)
                score_w = min(3.0, max(0.8, w * 0.45))
                for tid in c_tokens[tok]:
                    candidate_scores[tid] += score_w
                    pass_matches_count[tid] += 1
                    
        # Pass B (Bigrams)
        c_bgs = self.pass_b_bigrams[country]
        for bg in bigrams:
            if bg in c_bgs:
                for tid in c_bgs[bg]:
                    candidate_scores[tid] += 3.5
                    pass_matches_count[tid] += 1
                    
        # Pass C (Trigrams) - typo & transliteration recovery
        if core_name and len(candidate_scores) < self.pass_c_gate:
            if return_details:
                self.pass_c_invoked = getattr(self, 'pass_c_invoked', 0) + 1
            tris = get_char_trigrams(core_name)
            c_tris = self.pass_c_char_trigrams[country]
            c_tri_idfs = self.trigram_idfs[country]
            for tri in tris:
                if tri in c_tris:
                    w = c_tri_idfs.get(tri, 0.5)
                    score_w = min(1.2, max(0.2, w * 0.15))
                    for tid in c_tris[tri]:
                        candidate_scores[tid] += score_w
                        
        # Pass E (Address)
        if pin and h_num:
            ph_key = f"{pin}_{h_num}"
            if ph_key in self.pass_e_pin_house[country]:
                for tid in self.pass_e_pin_house[country][ph_key]:
                    candidate_scores[tid] += 5.0
                    pass_matches_count[tid] += 1
                    
        if h_num and addr_tokens:
            loc_key = f"{h_num}_{addr_tokens[-1]}"
            if loc_key in self.pass_e_house_loc[country]:
                for tid in self.pass_e_house_loc[country][loc_key]:
                    candidate_scores[tid] += 3.0
                    pass_matches_count[tid] += 1
                    
        if h_num and len(addr_tokens) >= 2:
            st_key = f"{h_num}_{addr_tokens[0]}"
            if st_key in self.pass_e_house_street[country]:
                for tid in self.pass_e_house_street[country][st_key]:
                    candidate_scores[tid] += 2.5
                    pass_matches_count[tid] += 1
                    
        if not candidate_scores:
            if return_details: return [], 0.0, 0
            return []
            
        # Cross-pass bonus
        for tid, p_cnt in pass_matches_count.items():
            if p_cnt >= 2:
                candidate_scores[tid] += min(3.0, p_cnt * 0.8)
                
        sorted_pairs = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        top_score = sorted_pairs[0][1] if sorted_pairs else 0.0
        n_passes = pass_matches_count[sorted_pairs[0][0]] if sorted_pairs else 0
        
        if force_k is not None:
            k = force_k
        else:
            low_k, med_k, high_k = self.adaptive_k_config
            if top_score >= 8.0 and n_passes >= 2:
                k = low_k
            elif top_score >= 3.5:
                k = med_k
            else:
                k = high_k
                
        top_candidates = [tid for tid, _ in sorted_pairs[:k]]
        
        if return_details:
            return top_candidates, top_score, n_passes
        return top_candidates

    def clear_country(self, country: str):
        self.target_reps.pop(country, None)
        self.pass_a_core_name.pop(country, None)
        self.pass_b_tokens.pop(country, None)
        self.pass_b_bigrams.pop(country, None)
        self.pass_c_char_trigrams.pop(country, None)
        self.pass_e_pin_house.pop(country, None)
        self.pass_e_house_loc.pop(country, None)
        self.pass_e_house_street.pop(country, None)
        self.pass_g_domain_concat.pop(country, None)
        self.token_idfs.pop(country, None)
        self.trigram_idfs.pop(country, None)
        gc.collect()

def generate_test_candidates_v2(
    test_dir: str = "dataset/test",
    output_path: str = "output/candidate_pairs_v2.tsv",
    adaptive_k_config: tuple = None,
    force_k: int = 25,
    max_trigram_freq: int = 500,
    pass_c_gate: int = 25
):
    t_start = time.time()
    print("=" * 60)
    print("PRODUCTION BLOCKING ENGINE V2: Generating Test Candidates")
    print(f"Force K: {force_k} | Adaptive K Config: {adaptive_k_config}")
    print(f"Max Trigram Freq: {max_trigram_freq} | Pass C Gate: {pass_c_gate}")
    print(f"Test Directory: {test_dir}")
    print(f"Output Path: {output_path}")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.exists(s1_path):
        print(f"Error: {s1_path} not found.")
        return
        
    s1_order = []
    s1_by_country = defaultdict(list)
    
    with open(s1_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0].strip()
            name = row[1].strip()
            addr = row[2].strip()
            ctry = row[3].strip() if len(row) > 3 else ''
            s1_order.append(eid)
            s1_by_country[ctry].append((eid, name, addr))
            
    countries = list(s1_by_country.keys())
    temp_dir = os.path.join("output", "temp_cands_v2")
    os.makedirs(temp_dir, exist_ok=True)
    country_cand_files = {}
    
    total_candidates_all = 0
    cand_lengths = []
    
    for c_idx, country in enumerate(countries, start=1):
        print(f"\n[Step {c_idx}/{len(countries)}] Processing Country '{country}'...")
        c_targets = []
        for s_fname in ["test_source2.tsv", "test_source3.tsv"]:
            s_path = os.path.join(test_dir, s_fname)
            with open(s_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for row in reader:
                    c = row[3].strip() if len(row) > 3 else ''
                    if c == country:
                        c_targets.append((row[0].strip(), row[1].strip(), row[2].strip()))
                        
        engine = ProductionBlockingEngineV2(
            max_trigram_freq=max_trigram_freq, 
            adaptive_k_config=adaptive_k_config, 
            pass_c_gate=pass_c_gate
        )
        engine.build_target_index_for_country(country, c_targets)
        del c_targets
        gc.collect()
        
        c_s1_recs = s1_by_country[country]
        c_out_path = os.path.join(temp_dir, f"cands_{country}.tsv")
        country_cand_files[country] = c_out_path
        
        with open(c_out_path, 'w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f, delimiter='\t')
            for eid, name, addr in c_s1_recs:
                cands = engine.retrieve_candidates_for_query(country, name, addr, force_k=force_k)
                cand_lengths.append(len(cands))
                total_candidates_all += len(cands)
                writer.writerow([eid, ','.join(cands)])
                
        engine.clear_country(country)
        del engine
        gc.collect()
        
    # Assemble
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

    for c_path in country_cand_files.values():
        if os.path.isfile(c_path): os.remove(c_path)
    if os.path.exists(temp_dir): os.rmdir(temp_dir)

    print(f"\nFinal Assembly Complete! File written to: {output_path}")
    print(f"Total S1: {len(s1_order):,} | Avg Cands: {np.mean(cand_lengths):.2f} | Time: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    generate_test_candidates_v2()
