# MTSD annotation QA audit

Generated: 2026-07-07T16:30:57  
QA files audited: 3  
Thresholds: duplicate IoU >= 0.95, overlap IoU >= 0.75

| Finding | Count |
| --- | --- |
| missing attribute findings | 0 |
| invalid attribute findings | 2 |
| duplicate pair findings | 3 |
| reference problem findings | 0 |

## Per-group counts

| group | annotations | attr_known_drop_value | dup_high_overlap | images |
|---|---|---|---|---|
| GRP-1 | 1960 | 0 | 0 | 724 |
| GRP-2 | 1621 | 1 | 3 | 630 |
| GRP-3 | 1844 | 1 | 0 | 617 |

Next steps: review findings visually with `python review_app.py`, then apply confirmed decisions with `python apply_fixes.py --decisions audit-20260707-163057/reviewed_decisions.json --apply`.
