# MTSD annotation QA audit

Generated: 2026-07-09T20:13:40  
QA files audited: 4  
Thresholds: duplicate IoU >= 0.95, overlap IoU >= 0.75

> Current audit snapshot for GRP-1/2/3/5. It remains a pre-fix record until
> reviewed decisions are applied and a new audit is generated.

| Finding | Count |
| --- | --- |
| missing attribute findings | 0 |
| invalid attribute findings | 5 |
| duplicate pair findings | 5 |
| reference problem findings | 0 |

## Per-group counts

| group | annotations | attr_known_drop_value | dup_exact_duplicate | dup_high_overlap | images |
|---|---|---|---|---|---|
| GRP-1 | 1960 | 0 | 0 | 0 | 724 |
| GRP-2 | 1621 | 1 | 0 | 3 | 630 |
| GRP-3 | 1844 | 1 | 0 | 0 | 617 |
| GRP-5 | 1841 | 3 | 1 | 1 | 657 |

Next steps: review findings visually with `python review_app.py`, then apply confirmed decisions with `python apply_fixes.py --decisions audit-20260709-201340/reviewed_decisions.json --apply`.
