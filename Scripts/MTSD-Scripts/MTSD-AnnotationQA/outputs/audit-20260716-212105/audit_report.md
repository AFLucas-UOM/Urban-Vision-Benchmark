# MTSD annotation QA audit

Generated: 2026-07-16T21:21:05  
QA files audited: 10  
Thresholds: duplicate IoU >= 0.95, overlap IoU >= 0.75

| Finding | Count |
| --- | --- |
| missing attribute findings | 0 |
| invalid attribute findings | 30 |
| duplicate pair findings | 18 |
| reference problem findings | 0 |

## Per-group counts

| group | annotations | attr_known_drop_value | dup_conflicting_duplicate | dup_exact_duplicate | dup_high_overlap | images |
|---|---|---|---|---|---|---|
| GRP-1 | 1960 | 0 | 0 | 0 | 0 | 724 |
| GRP-10 | 2358 | 2 | 0 | 0 | 4 | 789 |
| GRP-11 | 1506 | 4 | 1 | 0 | 2 | 602 |
| GRP-2 | 1621 | 1 | 0 | 0 | 3 | 630 |
| GRP-3 | 1844 | 1 | 0 | 0 | 0 | 617 |
| GRP-5 | 1841 | 3 | 0 | 1 | 1 | 657 |
| GRP-6 | 1106 | 3 | 0 | 0 | 0 | 470 |
| GRP-7 | 1891 | 7 | 0 | 0 | 0 | 710 |
| GRP-8 | 1869 | 2 | 0 | 0 | 0 | 703 |
| GRP-9 | 2226 | 7 | 0 | 0 | 6 | 715 |

Next steps: review findings visually with `python review_app.py`, then apply confirmed decisions with `python apply_fixes.py --decisions audit-20260716-212105/reviewed_decisions.json --apply`.
