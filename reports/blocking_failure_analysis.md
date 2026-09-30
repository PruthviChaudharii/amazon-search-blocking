# Detailed Failure Audit of Current Blocking System (Phase 3 Part 1)

**Evaluated Scope:** 10,000 Source 1 Validation Entities  
**Total Ground-Truth Positive Pairs in Pool:** 34,511  
**Successfully Recalled Pairs (K=25):** 25,892 (75.03%)  
**Total Missed Pairs:** 8,619 (24.97%)  

---

## 1. Quantitative Failure Taxonomy

| Failure Category | Missed Count | % of Missed Pairs | % of Total Positive Pairs | Primary Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **O. Candidate ranking/pruning failure** | **3,630** | **42.12%** | 10.52% | Target was in unpruned candidate pool, but was pushed outside Top-25 by noise. |
| **F. Transliteration / non-ASCII** | **2,384** | **27.66%** | 6.91% | Indian regional script (Tamil, Devanagari) vs English script mismatch. |
| **G. Domain-name variation** | **1,437** | **16.67%** | 4.16% | Target entity name was website URL (e.g. `domain.com`). |
| **I. Address variation** | **591** | **6.86%** | 1.71% | Name was noisy/abbreviated; address anchor did not trigger due to format differences. |
| **E. Character typo** | **365** | **4.23%** | 1.06% | Levenshtein distance / minor typos prevented exact token matching. |
| **C. Tokenization failure** | **154** | **1.79%** | 0.45% | Concatenated words or punctuation-glued tokens split differently. |
| **N. Common-token collision** | **32** | **0.37%** | 0.09% | Tokens were filtered as high-frequency (>1,500 occurrences) stopwords. |
| **M. House-number mismatch** | **8** | **0.09%** | 0.02% | Building number formatted with prefix/suffix. |
| **J. Missing address** | **7** | **0.08%** | 0.02% | Target entity had empty address string in S2/S3 (3.3% missing rate). |
| **H. Phone-number contamination** | **5** | **0.06%** | 0.01% | Appended phone number altered token set. |
| **P. Other** | **4** | **0.05%** | 0.01% | Compound noise across both name and address. |
| **B. Legal suffix variation** | **1** | **0.01%** | 0.00% | Legal suffixes altered token overlap. |
| **D. Word-order variation** | **1** | **0.01%** | 0.00% | Word order transposed across multi-token names. |

---

## 2. Qualitative Failure Case Examples

### O. Candidate ranking/pruning failure (3,630 cases)
- **S1:** `Payne Enterprises` | `3315 Fremont Street, Peoria, IL`
  **Target:** `Payne Enterpires` | `3315 FREMONT ST, PEORIA, IL`
- **S1:** `Payne Enterprises` | `3315 Fremont Street, Peoria, IL`
  **Target:** `PAYNE-ENRTPRMISES` | `3315 FREMONT SAINT, PEORIA, IL`
- **S1:** `Payne Enterprises` | `3315 Fremont Street, Peoria, IL`
  **Target:** `Payne Etrepndiels` | `3315 Fremont St, Peoria, Illinois`

### F. Transliteration / non-ASCII (2,384 cases)
- **S1:** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY`
  **Target:** `Dréxkor` | `85 Wanye Avenue, Ticonderoga Townshiip, New York`
- **S1:** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu`
  **Target:** `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu`
- **S1:** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu`
  **Target:** `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு`

### G. Domain-name variation (1,437 cases)
- **S1:** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY`
  **Target:** `maurewilliamscolombier.com` | `Wayne Ave, Ticonderoga Townshiip, New York`
- **S1:** `Summit Health LLC` | `95 Forest Edge Drive, Eads, TN`
  **Target:** `summithealth.com` | `95 Forest Edge Drive, Eads, Tennessee`
- **S1:** `Zander Blue Co` | `9236 Meadowmont View Drive, Charlotte, NC`
  **Target:** `zanderblue.com` | `Charlotte, 9236 Meadowmont View Dr, North Carolina`

### I. Address variation (591 cases)
- **S1:** `Uptown Pub` | `6114 10th Avenue, Spokane Valley, WA`
  **Target:** `Solkeloquo` | `6114 Tenth Ave, Spokane Valley, Washington`
- **S1:** `Zander Blue Co` | `9236 Meadowmont View Drive, Charlotte, NC`
  **Target:** `ZB` | `9236 Meadowmont View Drive, Charlotte, NC`
- **S1:** `Seabird (India) Projects-Lucknow` | `Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh`
  **Target:** `Xylonexbrix` | `3/7, Lucknow, UP`

### E. Character typo (365 cases)
- **S1:** `Vadyne Inc` | `Greensboro, 1604 Washington Street, NC`
  **Target:** `Vagdyae Inc` | `01604 Washington Street, PMB 841, Greensboro, North Carolina`
- **S1:** `Johnson Holdings Group LLC` | `Saint Paul, MN, 5300 East Street`
  **Target:** `Jhsnno Holdings Group LLC` | `SAINT PAUL, EAST ST, MN`
- **S1:** `Green Staffing Group Group` | `Springfield, IL, 41 Groton Drive`
  **Target:** `greenstaffinggroupcom` | `41- Groton Drive, Springfield, Illinois`

### C. Tokenization failure (154 cases)
- **S1:** `First Seven Exports Pvt Ltd` | `Maharashtra, At Shahagad Tq. Ambad, Jalna`
  **Target:** `@Firstseven` | `At Shahgad Tq. Ambad, Jalna, MH`
- **S1:** `KBM Research Private Limited` | `C/O Arvind Kumar, S/O Ram Sujan Prasad, Asochak, Police Station-Ramkrishan Nagar, Patna, Bihar`
  **Target:** `kbmresearch.c0m` | `C/O ARVIND KUMAR, S/O RAM SUJAN PRASAD, ASOCHAK, POLICE STATION-RAMKRISHAN NAGAR, PATNA, Bihar`
- **S1:** `New India Private Limited` | `Fl 401, Telangana, Sri Niketan Abhinandana Aprt. Ayyappa Society, Nr.Cgr Int. Pl.250 251, Madhapur, Hyderabad, Rangareddy`
  **Target:** `@newindia` | `Rangareddy, #240 Fl 401, Hyderabad, Sri Niketan Abhinandana Aprt. Ayyappa Society, Nr.cgr Int. Pl.250 251, Madhapur, Andhra Pradesh`

### N. Common-token collision (32 cases)
- **S1:** `Srk Traders Private Limited` | `Ff Kh No. 42, 43, 44, Vipin Garden Extn. Near Gali No. 34, Delhi, Plot No. G-1, West Delhi`
  **Target:** `5RK PRIVATE  LIMITED SERVICES` | `NO 753 PLOT NO. G-1, FF KH NO. 42, 43, 44, VIPIN GARDEN EXTN. NEAR GALI NO. 34, DELHI, WEST DELHI, Delhi`
- **S1:** `Consultancy Research Ltd` | `No. 158, 8Th Main, 1St Cross, Btm Layout 1St Stage, Bangalore, Karnataka`
  **Target:** `Consultancy Ltd Center` | `No. 158/1, 8Th Main, 1St Cross, Btm Layout 1St Stage, Bangalore, KA`
- **S1:** `Corvus & Co` | `X - 54, Loha Mandi Naraina, New Delhi, Delhi`
  **Target:** `Chmmsu And Co` | `X - 54, Loha Mandi Naraina, New Delhi, DL`

### M. House-number mismatch (8 cases)
- **S1:** `VWQ Brookfield LLC` | `Unit Apartment 1, 52 Losson Road, NY, Cheektowaga`
  **Target:** `VB` | `BUFFALO CITY, NY, 52 LOSSON ROAD`
- **S1:** `Commercial Financial Global Center` | `6532 Sperryville Road, Watson, NY`
  **Target:** `Rizaecto` | `6533 SPERRYVILLE RD, PMB 6380, GLENFIELD, NY`
- **S1:** `Green College` | `Door No.26/281, Sadhoo Company Road, No.55 Kra, Ondenparamb, Kannur, Kerala`
  **Target:** `Lyraonyx` | `26/281, Kannur, Keralam`

### J. Missing address (7 cases)
- **S1:** `Pediatric Continental Associates LLC` | `1208 Stafford Road, Darlington, MD`
  **Target:** `Pediatric Associates LLC Center` | ``
- **S1:** `J/U Earths LLC` | `11304 Templeton Drive, Cincinnati, OH`
  **Target:** `J/U LLC Center` | ``
- **S1:** `R+ Vernal Inc` | `158 Trailblazer Drive, Bastrop, TX`
  **Target:** `R+ Vrethal Inc` | ``

### H. Phone-number contamination (5 cases)
- **S1:** `Medio European` | `37 Carter Street, Unit 1, Leominster, MA`
  **Target:** `@medioeuropean - 6455232733` | `Massachusetts, # 1, Leominster, Carter Street`
- **S1:** `Tejtech Tradelinks Pvt Ltd` | `Sr.No.35/4/1/1, Flat No.-4, Sita Residency Ganara, Pune, Maharashtra`
  **Target:** `Evovio - 5537801104` | `महाराष्ट्र, null, Pune, Sr.no.#35/4/1/1, Pune`
- **S1:** `Heritage Biomedical Systems L.L.C.` | `619 Chucky Pike, Jefferson City, TN`
  **Target:** `VIOXYLO - 4443557697` | `619 CHUCKY PIKE, JEFFERSON CITY, TN`

### P. Other (4 cases)
- **S1:** `Aarvita Biosciences Private Limited` | `No.41, Ponnambalam Salai, K.K.Nagar, Chennai, Tamil Nadu`
  **Target:** `Jaxrizagild` | `41, CHENNAI CITY REGION, தமிழ்நாடு`
- **S1:** `XHY Leasing` | `Dombivli (East), Thane, Shop No.2, Maharashtra, Ground Floor, Lord House Co-Op-Soc.Ltd, Rajaji Path Road`
  **Target:** `Vantageriza` | `THANE, NO D/2, महाराष्ट्र, DOMBIVLI (EAST)`
- **S1:** `Big Deli` | `406 Buchert Road, New Hanover Township, PA`
  **Target:** `Fluxnylayuma` | `PA, 406 BUCHERT RD, POTTSTOWN`

### B. Legal suffix variation (1 cases)
- **S1:** `K/U Group Inc` | `3815 Saint Victor Street, Baltimore, MD`
  **Target:** `K/U Group Inc` | `Maryland, 3815 St Victor Street, Broooklyn`

### D. Word-order variation (1 cases)
- **S1:** `K/U Group Inc` | `3815 Saint Victor Street, Baltimore, MD`
  **Target:** `K/U Group [Incorporated]` | `3815 SAINT VICTOR ST, BALTIMORE, MD`

