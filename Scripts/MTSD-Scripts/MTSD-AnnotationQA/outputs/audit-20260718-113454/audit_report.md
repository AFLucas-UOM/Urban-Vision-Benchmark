# MTSD annotation QA audit

Generated: 2026-07-18T11:34:55  
QA files audited: 11  
Thresholds: duplicate IoU >= 0.95, overlap IoU >= 0.75

| Finding | Count |
| --- | --- |
| missing attribute findings | 0 |
| invalid attribute findings | 0 |
| duplicate pair findings | 14 |
| reference problem findings | 0 |

## Per-group counts

| group | annotations | dup_conflicting_duplicate | dup_high_overlap | images |
|---|---|---|---|---|
| GRP-1 | 1960 | 0 | 0 | 724 |
| GRP-10 | 2334 | 0 | 3 | 779 |
| GRP-11 | 1506 | 1 | 2 | 602 |
| GRP-2 | 1621 | 0 | 3 | 630 |
| GRP-3 | 1844 | 0 | 0 | 617 |
| GRP-4 | 2314 | 0 | 0 | 875 |
| GRP-5 | 1839 | 0 | 0 | 657 |
| GRP-6 | 1106 | 0 | 0 | 470 |
| GRP-7 | 1891 | 0 | 0 | 710 |
| GRP-8 | 1869 | 0 | 0 | 703 |
| GRP-9 | 2225 | 0 | 5 | 715 |

Next steps: review findings visually with `python review_app.py`, then apply confirmed decisions with `python apply_fixes.py --decisions audit-20260718-113454/reviewed_decisions.json --apply`.
