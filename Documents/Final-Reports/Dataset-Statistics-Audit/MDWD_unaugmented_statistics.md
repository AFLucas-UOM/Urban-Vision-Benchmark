# MDWD unaugmented dataset statistics audit

Source included: `unaugmented COCO split directories only`.

| Split | Images | Objects | Class total check |
|---|---:|---:|---|
| Train | 2,956 | 9,262 | PASS |
| Valid | 369 | 1,074 | PASS |
| Test | 369 | 1,106 | PASS |
| **Total** | **3,694** | **11,442** | **PASS** |

## Train class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Mixed Waste | 2,566 | 27.7046% |
| Orange CMD | 667 | 7.2015% |
| Organic Waste | 1,677 | 18.1062% |
| Other Waste | 2,150 | 23.2131% |
| Recyclable Material | 2,202 | 23.7746% |
| **Total** | **9,262** | **100.0000%** |

## Valid class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Mixed Waste | 275 | 25.6052% |
| Orange CMD | 78 | 7.2626% |
| Organic Waste | 196 | 18.2495% |
| Other Waste | 261 | 24.3017% |
| Recyclable Material | 264 | 24.5810% |
| **Total** | **1,074** | **100.0000%** |

## Test class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Mixed Waste | 334 | 30.1989% |
| Orange CMD | 78 | 7.0524% |
| Organic Waste | 178 | 16.0940% |
| Other Waste | 278 | 25.1356% |
| Recyclable Material | 238 | 21.5190% |
| **Total** | **1,106** | **100.0000%** |

## Integrity checks

- Exact-image duplicate groups: 0
- Normalized source-filename duplicate groups: 99
- Cross-split source-filename collision groups: 36
- Probable source-image leakage groups (dHash distance ≤5): 15
- Findings requiring attention: 36
