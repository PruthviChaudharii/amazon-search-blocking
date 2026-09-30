import re
import time
from collections import Counter

# Test optimization of name representation
LEGAL_SET = {'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 
             'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprises', 'associates'}

RE_DOMAIN = re.compile(r'\.(com|org|net|in|co|io|fr)\b', re.IGNORECASE)
RE_AMP = re.compile(r'[\&\+]')
RE_NON_ALNUM = re.compile(r'[^a-z0-9\s]')
RE_PIN = re.compile(r'\b[0-9]{4,6}\b')
RE_HOUSE = re.compile(r'\b[0-9]+[a-z]?\b')

def fast_clean_name(name):
    s = name.lower()
    if '.' in s:
        s = RE_DOMAIN.sub('', s)
    if '&' in s or '+' in s:
        s = RE_AMP.sub(' and ', s)
    s = RE_NON_ALNUM.sub(' ', s)
    tokens = [w for w in s.split() if w not in LEGAL_SET and len(w) >= 2]
    return ' '.join(tokens), tokens

names = [
    "Raj Investments LLP",
    "Systel Buildstructure (India) Private Limited",
    "Maure Williams Colombier Inc",
    "Ss Food Private Limited",
    "Obsidian, LLC"
] * 2000

t0 = time.time()
for n in names:
    core_name, tokens = fast_clean_name(n)
elapsed = time.time() - t0
print(f"Processed {len(names):,} names in {elapsed:.4f}s ({len(names)/elapsed:,.0f} names/sec)")
