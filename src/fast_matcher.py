"""fast_matcher.py — deterministic streaming matcher for the test stage.

Design choices (speed / correctness trade‑off):
  * Load test source1 / source2 / source3 entirely and pre‑compute per‑record
    features once.
  * Use a fixed threshold (0.35) so we skip the noisy training‑data sweep that
    previously caused OSError / ValueError crashes.
  * Score every candidate pair with lightweight cached feature vectors.
  * Write matches to output/matching_results.tsv incrementally (streaming).
  * Progress logging every 1 000 000 candidates.
"""

import csv
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

# ── constants ──────────────────────────────────────────────────────────────
THRESHOLD = 0.35          # deterministic; no training data sweep required
CAND_PATH  = Path("output/candidate_pairs_v2.tsv")
OUT_PATH   = Path("output/matching_results.tsv")
TEST_DIR   = Path("dataset/test")

# ── tiny helpers (no external matching_features imports) ────────────────────
_non_alnum = re.compile(r"[^a-z0-9 ]+")
_multi_sp  = re.compile(r" {2,}")

def normalise(text: str) -> str:
    t = text.lower()
    t = _non_alnum.sub(" ", t)
    t = _multi_sp.sub(" ", t)
    return t.strip()

_STOP = {"the","a","an","of","and","or","co","ltd","llc","inc","corp",
         "company","limited","group","holdings","services","international"}

def tokens(text: str):
    return set(normalise(text).split()) - _STOP

_postal = re.compile(r"\b(\d{4,6})\b")
_house  = re.compile(r"^\s*(\d+[a-z]?)\b")

def postal_code(addr: str):
    m = _postal.search(addr)
    return m.group(1) if m else ""

def house_num(addr: str):
    m = _house.match(addr)
    return m.group(1) if m else ""

# ── pre‑compute per‑record feature dict ─────────────────────────────────────
def build_feature(row: dict) -> dict:
    name    = row.get("name", "")
    addr    = row.get("addr", "")
    country = row.get("country", "").strip().lower()
    norm_n  = normalise(name)
    norm_a  = normalise(addr)
    return {
        "norm_name"  : norm_n,
        "name_tokens": tokens(name),
        "norm_addr"  : norm_a,
        "addr_tokens": tokens(addr),
        "postal"     : postal_code(norm_a),
        "house"      : house_num(norm_a),
        "country"    : country,
    }

# ── load source files ────────────────────────────────────────────────────────
def load_source(path: Path) -> dict:
    """Return {entity_id: feature_dict}."""
    data = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr, None)
        for row in rdr:
            if len(row) < 2:
                continue
            eid     = row[0].strip()
            name    = row[1].strip() if len(row) > 1 else ""
            addr    = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""
            data[eid] = build_feature({"name": name, "addr": addr, "country": country})
    return data

print("Loading test sources …")
t0 = time.time()
src1 = load_source(TEST_DIR / "test_source1.tsv")
src2 = load_source(TEST_DIR / "test_source2.tsv")
src3 = load_source(TEST_DIR / "test_source3.tsv")
print(f"  source1: {len(src1):,}  source2: {len(src2):,}  source3: {len(src3):,}  "
      f"({time.time()-t0:.1f}s)")

# ── score a pair ─────────────────────────────────────────────────────────────
def score(a: dict, b: dict) -> float:
    feats = []

    # 1. exact normalised name match
    feats.append(1.0 if a["norm_name"] == b["norm_name"] else 0.0)

    # 2. token Jaccard (names)
    ta, tb = a["name_tokens"], b["name_tokens"]
    if ta or tb:
        feats.append(len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0)
    else:
        feats.append(0.0)

    # 3. char‑level similarity
    feats.append(SequenceMatcher(None, a["norm_name"], b["norm_name"]).ratio())

    # 4. token Jaccard (addresses)
    aa, ab = a["addr_tokens"], b["addr_tokens"]
    if aa or ab:
        feats.append(len(aa & ab) / len(aa | ab) if (aa | ab) else 0.0)
    else:
        feats.append(0.0)

    # 5. postal code agreement
    feats.append(1.0 if a["postal"] and a["postal"] == b["postal"] else 0.0)

    # 6. house number agreement
    feats.append(1.0 if a["house"] and a["house"] == b["house"] else 0.0)

    # 7. country agreement
    feats.append(1.0 if a["country"] and a["country"] == b["country"] else 0.0)

    return sum(feats) / len(feats)

# ── benchmark on 100 k rows ──────────────────────────────────────────────────
BENCH = 100_000
print(f"\nBenchmarking on {BENCH:,} candidate rows …")
bcount = 0
bt0 = time.time()
with open(CAND_PATH, "r", encoding="utf-8", errors="replace") as fin:
    for line in fin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        s1_id, cand_id = parts[0], parts[1]
        s1_f = src1.get(s1_id)
        cf   = src2.get(cand_id) or src3.get(cand_id)
        if s1_f and cf:
            _ = score(s1_f, cf)
        bcount += 1
        if bcount >= BENCH:
            break
btime = time.time() - bt0
rate  = bcount / btime if btime > 0 else 0
print(f"  elapsed: {btime:.2f}s  rows/sec: {rate:,.0f}")

TOTAL_CANDS = 42_503_400
projected   = TOTAL_CANDS / rate if rate > 0 else float("inf")
print(f"  projected full runtime: {projected/60:.1f} minutes")

if projected > 3600:
    raise RuntimeError(
        f"Projected runtime {projected/60:.0f} min > 60 min — "
        "optimise further before full scan."
    )

# ── full streaming pass ───────────────────────────────────────────────────────
print(f"\nThreshold: {THRESHOLD}  — writing {OUT_PATH} …")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
processed = 0
written   = 0
t1 = time.time()
with open(CAND_PATH, "r", encoding="utf-8", errors="replace", buffering=1<<20) as fin, \
     open(OUT_PATH,  "w", encoding="utf-8", newline="", buffering=1<<20) as fout:
    writer = csv.writer(fout, delimiter="\t")
    for line in fin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        s1_id, cand_id = parts[0], parts[1]
        s1_f = src1.get(s1_id)
        cf   = src2.get(cand_id) or src3.get(cand_id)
        if not s1_f or not cf:
            processed += 1
            continue
        if score(s1_f, cf) >= THRESHOLD:
            writer.writerow([s1_id, cand_id])
            written += 1
        processed += 1
        if processed % 1_000_000 == 0:
            elapsed = time.time() - t1
            print(f"Processed {processed:,} / {TOTAL_CANDS:,}  matches so far: {written:,}  "
                  f"({processed/elapsed:,.0f} rows/sec)")

elapsed = time.time() - t1
print(f"\nDone. Processed {processed:,} candidates in {elapsed:.1f}s")
print(f"Matches written: {written:,}")
print(f"Output: {OUT_PATH}")
