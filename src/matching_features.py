# src/matching_features.py
"""
Feature engineering utilities for pairwise business entity matching.
All functions are deterministic, handle missing/empty strings gracefully,
and avoid any external data or network calls.
"""
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Normalization helpers (re‑use logic from blocking utilities)
# ---------------------------------------------------------------------------
LEGAL_STOPWORDS = {
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp',
    'corporation', 'llc', 'llp', 'co', 'company', 'gmbh', 'plc', 'enterprises',
    'enterprise', 'associates', 'group', 'holdings', 'holding', 'industries',
    'industry', 'international', 'services', 'service', 'solutions', 'solution',
    'consulting', 'consultancy', 'the', 'and', 'of', 'in', 'at', 'on', 'for',
    'by', 'with', 'an', 'a', 'sarl', 'sa', 'sas', 'sasu', 'sci', 'snc', 'eurl',
    'ets', 'etablissements', 'societe', 'cie', 'france', 'fr', 'de', 'du', 'des',
    'la', 'le', 'les', 'store', 'stores', 'shop', 'centre', 'center', 'agency',
    'traders', 'trading'
}
COMMON_ADDRESS_STOPWORDS = {
    'near', 'opp', 'opposite', 'behind', 'beside', 'road', 'rd', 'street', 'st',
    'avenue', 'ave', 'lane', 'ln', 'floor', 'fl', 'block', 'blk', 'sector', 'sec',
    'plot', 'no', 'number', 'door', 'flat', 'building', 'bldg', 'complex', 'cross',
    'main', 'nagar', 'colony', 'layout', 'dist', 'district', 'post', 'po', 'rue',
    'boulevard', 'bd', 'chemin', 'route', 'allee', 'place'
}
RE_NON_ALNUM = re.compile(r'[^a-z0-9\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\s]')
RE_PHONE = re.compile(r'(\+?\d[\d\s\-().]{5,}\d)')
RE_DOMAIN = re.compile(r'([a-z0-9.-]+\.(com|org|net|in|co|io|fr|gov|edu|biz|info))', re.IGNORECASE)

def _unicode_normalize(s: str) -> str:
    return unicodedata.normalize('NFKD', s) if s else ''

def _clean_text(s: str) -> str:
    s = s.lower()
    s = RE_NON_ALNUM.sub(' ', s)
    return ' '.join(s.split())

def normalize_text(s: str) -> str:
    """Lower‑case, strip punctuation, collapse whitespace.
    Returns empty string for falsy input.
    """
    return _clean_text(_unicode_normalize(s)) if s else ''

def tokenise(s: str) -> list:
    return [t for t in normalize_text(s).split() if t]

def core_tokens(name: str) -> list:
    return [t for t in tokenise(name) if t not in LEGAL_STOPWORDS and len(t) >= 3]

def core_name(name: str) -> str:
    return ' '.join(core_tokens(name))

# ---------------------------------------------------------------------------
# Name‑based features
# ---------------------------------------------------------------------------
def name_exact_eq(name_a: str, name_b: str) -> int:
    return int(normalize_text(name_a) == normalize_text(name_b))

def core_name_eq(name_a: str, name_b: str) -> int:
    return int(core_name(name_a) == core_name(name_b))

def token_overlap(name_a: str, name_b: str) -> float:
    set_a = set(core_tokens(name_a))
    set_b = set(core_tokens(name_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def token_overlap_idf(name_a: str, name_b: str, idf_lookup: dict) -> float:
    # Simple weighted overlap using provided idf values (default to 1.0)
    toks_a = core_tokens(name_a)
    toks_b = core_tokens(name_b)
    if not toks_a or not toks_b:
        return 0.0
    set_a, set_b = set(toks_a), set(toks_b)
    intersect = set_a & set_b
    weight = sum(idf_lookup.get(t, 1.0) for t in intersect)
    total = sum(idf_lookup.get(t, 1.0) for t in set_a | set_b)
    return weight / total if total else 0.0

def char_similarity(a: str, b: str) -> float:
    # Normalized Levenshtein‑like similarity via difflib
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def trigram_set(s: str) -> set:
    s = normalize_text(s).replace(' ', '')
    return {s[i:i+3] for i in range(len(s)-2)} if len(s) >= 3 else set()

def trigram_similarity(a: str, b: str) -> float:
    set_a, set_b = trigram_set(a), trigram_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def word_order_insensitive_jaccard(a: str, b: str) -> float:
    tokens_a = set(tokenise(a))
    tokens_b = set(tokenise(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

def legal_suffix_agreement(name_a: str, name_b: str) -> int:
    # Compare the trailing legal suffixes (if any) after stopword removal.
    def suffixes(name):
        tokens = tokenise(name)
        return [t for t in tokens if t in LEGAL_STOPWORDS]
    return int(suffixes(name_a) == suffixes(name_b))

# ---------------------------------------------------------------------------
# Address‑based features
# ---------------------------------------------------------------------------
def address_exact_eq(addr_a: str, addr_b: str) -> int:
    return int(normalize_text(addr_a) == normalize_text(addr_b))

def address_token_overlap(addr_a: str, addr_b: str) -> float:
    set_a = set([t for t in tokenise(addr_a) if t not in COMMON_ADDRESS_STOPWORDS])
    set_b = set([t for t in tokenise(addr_b) if t not in COMMON_ADDRESS_STOPWORDS])
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def postal_agreement(addr_a: str, addr_b: str) -> int:
    def postal(s):
        m = re.search(r'\b\d{4,6}\b', s)
        return m.group(0) if m else ''
    return int(postal(addr_a) == postal(addr_b) and postal(addr_a) != '')

def house_number_agreement(addr_a: str, addr_b: str) -> int:
    def house(s):
        m = re.search(r'\b\d+[a-z]?\b', s)
        return m.group(0) if m else ''
    return int(house(addr_a) == house(addr_b) and house(addr_a) != '')

def missing_address_indicator(addr: str) -> int:
    return int(not addr or not addr.strip())

# ---------------------------------------------------------------------------
# Contact / identifier features
# ---------------------------------------------------------------------------
def phone_numbers(s: str) -> set:
    return set(m.group(0) for m in RE_PHONE.finditer(s))

def phone_agreement(a: str, b: str) -> int:
    return int(bool(phone_numbers(a) & phone_numbers(b)))

def domain_from_url(s: str) -> str:
    m = RE_DOMAIN.search(s.lower())
    return m.group(1) if m else ''

def domain_eq(a: str, b: str) -> int:
    return int(domain_from_url(a) == domain_from_url(b) and domain_from_url(a) != '')

# ---------------------------------------------------------------------------
# Miscellaneous features
# ---------------------------------------------------------------------------
def country_agreement(country_a: str, country_b: str) -> int:
    return int(country_a.strip().lower() == country_b.strip().lower() and country_a)

def source_indicator(source_id: str) -> int:
    # 0 = source2, 1 = source3 (uses prefix S2- / S3-)
    return int(source_id.startswith('S3-'))

def missing_field_indicator(value: str) -> int:
    return int(not value or not value.strip())

# ---------------------------------------------------------------------------
# Feature vector assembly (lightweight, deterministic)
# ---------------------------------------------------------------------------
def build_feature_dict(pair: dict, idf_lookup: dict = None) -> dict:
    """Assemble all features for a (source1, candidate) pair.
    ``pair`` must contain the keys:
        - s1_name, s1_address, s1_country
        - cand_name, cand_address, cand_country, cand_id
    ``idf_lookup`` is optional; if omitted each token IDF defaults to 1.0.
    Returns a plain ``dict`` suitable for feeding into LightGBM/ sklearn.
    """
    if idf_lookup is None:
        idf_lookup = {}
    # Name features
    f = {}
    f['name_exact_eq'] = name_exact_eq(pair['s1_name'], pair['cand_name'])
    f['core_name_eq'] = core_name_eq(pair['s1_name'], pair['cand_name'])
    f['token_overlap'] = token_overlap(pair['s1_name'], pair['cand_name'])
    f['token_overlap_idf'] = token_overlap_idf(pair['s1_name'], pair['cand_name'], idf_lookup)
    f['char_sim'] = char_similarity(pair['s1_name'], pair['cand_name'])
    f['trigram_sim'] = trigram_similarity(pair['s1_name'], pair['cand_name'])
    f['wordorder_jaccard'] = word_order_insensitive_jaccard(pair['s1_name'], pair['cand_name'])
    f['legal_suffix_agreement'] = legal_suffix_agreement(pair['s1_name'], pair['cand_name'])
    # Address features
    f['addr_exact_eq'] = address_exact_eq(pair['s1_address'], pair['cand_address'])
    f['addr_token_overlap'] = address_token_overlap(pair['s1_address'], pair['cand_address'])
    f['postal_agreement'] = postal_agreement(pair['s1_address'], pair['cand_address'])
    f['house_num_agreement'] = house_number_agreement(pair['s1_address'], pair['cand_address'])
    f['missing_addr'] = missing_address_indicator(pair['cand_address'])
    # Contact / identifiers
    f['phone_agreement'] = phone_agreement(pair['s1_name'] + ' ' + pair['s1_address'],
                                          pair['cand_name'] + ' ' + pair['cand_address'])
    f['domain_eq'] = domain_eq(pair['s1_name'] + ' ' + pair['s1_address'],
                               pair['cand_name'] + ' ' + pair['cand_address'])
    # Misc
    f['country_agreement'] = country_agreement(pair['s1_country'], pair['cand_country'])
    f['source_indicator'] = source_indicator(pair['cand_id'])
    f['missing_name'] = missing_field_indicator(pair['cand_name'])
    f['missing_address'] = missing_field_indicator(pair['cand_address'])
    return f
