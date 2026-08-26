# MTSD unaugmented dataset statistics audit

Source included: `unaugmented COCO split directories only`.

| Split | Images | Objects | Class total check |
|---|---:|---:|---|
| Train | 5,986 | 16,535 | PASS |
| Valid | 749 | 1,996 | PASS |
| Test | 747 | 1,978 | PASS |
| **Total** | **7,482** | **20,509** | **PASS** |

## Train class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Pedestrian Crossing | 1,174 | 7.1001% |
| Stop Sign | 1,071 | 6.4772% |
| No Entry (One Way) | 1,789 | 10.8195% |
| Roundabout Ahead | 648 | 3.9190% |
| No Through Road (T-Junction) | 407 | 2.4614% |
| Blind-Spot Mirror (Convex Mirror) | 1,328 | 8.0314% |
| Street Sign | 813 | 4.9168% |
| Directional Sign | 906 | 5.4793% |
| Tourist Sign | 27 | 0.1633% |
| Auxiliary Sign | 1,069 | 6.4651% |
| Back-Unknown | 5,416 | 32.7548% |
| Other-Unknown | 1,887 | 11.4122% |
| **Total** | **16,535** | **100.0000%** |

## Valid class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Pedestrian Crossing | 139 | 6.9639% |
| Stop Sign | 135 | 6.7635% |
| No Entry (One Way) | 212 | 10.6212% |
| Roundabout Ahead | 86 | 4.3086% |
| No Through Road (T-Junction) | 69 | 3.4569% |
| Blind-Spot Mirror (Convex Mirror) | 159 | 7.9659% |
| Street Sign | 97 | 4.8597% |
| Directional Sign | 112 | 5.6112% |
| Tourist Sign | 3 | 0.1503% |
| Auxiliary Sign | 111 | 5.5611% |
| Back-Unknown | 642 | 32.1643% |
| Other-Unknown | 231 | 11.5731% |
| **Total** | **1,996** | **100.0000%** |

## Test class distribution

| Class | Objects | % of split objects |
|---|---:|---:|
| Pedestrian Crossing | 143 | 7.2295% |
| Stop Sign | 130 | 6.5723% |
| No Entry (One Way) | 211 | 10.6673% |
| Roundabout Ahead | 67 | 3.3873% |
| No Through Road (T-Junction) | 61 | 3.0839% |
| Blind-Spot Mirror (Convex Mirror) | 160 | 8.0890% |
| Street Sign | 94 | 4.7523% |
| Directional Sign | 108 | 5.4601% |
| Tourist Sign | 3 | 0.1517% |
| Auxiliary Sign | 120 | 6.0667% |
| Back-Unknown | 661 | 33.4176% |
| Other-Unknown | 220 | 11.1223% |
| **Total** | **1,978** | **100.0000%** |

## Integrity checks

- Exact-image duplicate groups: 1
- Normalized source-filename duplicate groups: 0
- Cross-split source-filename collision groups: 0
- Probable source-image leakage groups (dHash distance ≤5): 0
- Source-image hash duplicate groups in manifest: 1
- Source-image leakage groups across splits: 0
- COCO images absent from manifest: 0
- Manifest images absent from COCO: 0
- Final-QA source files checked: 11
- Final-QA image total matches prepared COCO: True
- Final-QA annotation total matches prepared COCO: True
- Final-QA per-class totals match prepared COCO: True
- Findings requiring attention: 3
