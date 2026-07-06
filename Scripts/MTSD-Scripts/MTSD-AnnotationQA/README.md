# MTSD-AnnotationQA

Audit → visual review → guarded apply workflow for the MTSD QA annotations
(`Datasets/MTSD/Annotations/*/Final-QA/QA-*.json`).

Groups are **discovered dynamically** — any QA-style JSON under the
annotations root is audited, so GRP-4+ join automatically once their
Final-QA export exists. The attribute schema (required attributes +
controlled vocabulary) is read from
[AttributeClassification/config/default.yaml](../AttributeClassification/config/default.yaml)
so the audit can never disagree with the training pipeline.

## The three-step workflow

```
scan_annotations.py  ->  review_app.py  ->  apply_fixes.py
     (read-only)        (writes decisions      (the ONLY step that
                         only, never edits      modifies annotations;
                         annotations)           dry-run + backups)
```

### 1. Audit

```bash
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py --dry-run   # findings summary only
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py            # writes timestamped reports
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py --verify-images  # + open every image header
```

Checks performed:

| Category | Findings |
| --- | --- |
| Missing attributes | absent `attributes` dict, absent keys, null/empty/`None` values for `view_angle`, `mounting`, `condition`, `sign_shape` |
| Invalid attributes | unknown vocabulary values, case mismatches and close misspellings (with fix suggestions), unknown `category_id`, known drop values (`Damaged-Unknown`) |
| Duplicates | same-image pairs with IoU ≥ 0.95 (exact vs conflicting-attribute duplicates) and 0.75–0.95 high-overlap warnings; thresholds configurable via `--iou-dup` / `--iou-overlap` |
| References | missing/old-convention `source_image` paths, missing or unreadable image files, malformed JSON, orphan annotations, invalid bboxes |

Outputs go to a **timestamped folder** `outputs/audit-<stamp>/` (previous
audits are never overwritten): `missing_attributes.csv`,
`invalid_attributes.csv`, `duplicate_candidates.csv`,
`reference_problems.csv`, `per_group_stats.csv`, `audit_summary.json`,
`audit_report.md`.

### 2. Review (Gradio)

```bash
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/review_app.py            # latest audit
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/review_app.py --audit outputs/audit-<stamp>
```

Two tabs:

* **Attribute fixes** — image + zoomed crop with the annotation box drawn, the
  problematic field with its current value and any suggestion; fill the correct
  value from the vocabulary dropdown or type it, then *Save decision* (or *Skip*).
* **Duplicates** — both boxes overlaid (A red, B blue) with a field-by-field
  attribute comparison; *Keep A (delete B)*, *Keep B (delete A)*, *Keep both*,
  or *Skip*.

The app **never edits annotation files**. Every decision is written to
`reviewed_decisions.json` inside the audit folder (re-deciding the same item
replaces the earlier decision; a delete supersedes attribute fixes on the
same annotation).

### 3. Apply (guarded)

```bash
# preview exactly what would change (default is dry-run):
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py --decisions outputs/audit-<stamp>/reviewed_decisions.json --dry-run

# actually apply (asks for a typed YES; add --yes to skip the prompt):
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py --decisions outputs/audit-<stamp>/reviewed_decisions.json --apply
```

Safety guarantees of the apply step:

- dry-run is the default; `--apply` + typed `YES` confirmation required to write;
- every modified file first gets a `*.pre-qa-fix-<stamp>.bak` backup;
- only targeted fields change (one attribute value, or one deleted annotation
  object — image entries are asserted untouched);
- decisions whose `old_value` no longer matches the file are skipped and
  reported (the file changed since review — re-run the scan);
- the result is re-parsed and count-checked before the original is replaced;
- an `applied_changes-<stamp>.json` log records everything done.

**Note on formatting:** the QA JSONs were exported with non-standard
whitespace; the first apply re-serialises them with `indent=2`. Content is
preserved exactly (backups + git history hold the pre-edit files).

**After applying fixes:** regenerate the attribute-classification manifest
(`python -m mtsd_attr.data_manifest` from
`Scripts/MTSD-Scripts/AttributeClassification/`), since it pins QA SHA-256
hashes.

## Dependencies

`scan_annotations.py` and `apply_fixes.py` are stdlib-only (PyYAML optional,
for reading the schema from the training config; Pillow optional, for
`--verify-images`). `review_app.py` needs `gradio` and `Pillow` — the
`mtsd-base` conda env has both.
