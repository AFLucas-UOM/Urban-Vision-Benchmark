# MTSD annotation QA audit

Generated: 2026-07-07T16:17:27  
QA files audited: 3  
Thresholds: duplicate IoU >= 0.95, overlap IoU >= 0.75

> Historical audit retained for provenance. It predates the later cleanup and
> GRP-5 QA export; use `../audit-20260709-201340/audit_report.md` for the latest
> four-group audit snapshot.

| Finding | Count |
| --- | --- |
| missing attribute findings | 7 |
| invalid attribute findings | 4 |
| duplicate pair findings | 42 |
| reference problem findings | 0 |

## Per-group counts

| group | annotations | attr_known_drop_value | attr_missing | dup_conflicting_duplicate | dup_exact_duplicate | dup_high_overlap | images |
|---|---|---|---|---|---|---|---|
| GRP-1 | 1979 | 0 | 7 | 6 | 20 | 0 | 724 |
| GRP-2 | 1621 | 2 | 0 | 0 | 0 | 3 | 630 |
| GRP-3 | 1857 | 2 | 0 | 4 | 9 | 0 | 617 |

Next steps: review findings visually with `python review_app.py`, then apply confirmed decisions with `python apply_fixes.py --decisions audit-20260707-161727/reviewed_decisions.json --apply`.
