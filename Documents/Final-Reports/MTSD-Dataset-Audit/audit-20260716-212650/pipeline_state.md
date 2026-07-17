# MTSD dataset-pipeline state audit
Generated 2026-07-16T19:26:40+00:00 (read-only).

**NOT READY for final preparation.**

## Blockers
- Final-QA groups outside approved scope (need audit + gate refresh, never auto-approval): ['GRP-7', 'GRP-8', 'GRP-9', 'GRP-10', 'GRP-11']
- QA gate is unresolved; outstanding findings in newest audit: {'invalid_attribute_values': 30, 'duplicate_candidates': 18}; unaudited groups: none

## Warnings
- Attribute manifest auto-discovers groups: the next manifest refresh/training run will ingest ['GRP-10', 'GRP-11'], changing the dataset under the completed size-ablation runs (trained on ['GRP-1', 'GRP-2', 'GRP-3', 'GRP-5', 'GRP-6', 'GRP-7', 'GRP-8', 'GRP-9']). Decide the final attribute scope explicitly before any further attribute training.
- No prepared MTSD dataset exists yet (Datasets/MTSD/Prepared). Prompt-based MTSD evaluation and supervised training wait on final preparation.

## Scope
- approved: GRP-1, GRP-2, GRP-3, GRP-5, GRP-6
- discovered valid: GRP-1, GRP-2, GRP-3, GRP-5, GRP-6, GRP-7, GRP-8, GRP-9, GRP-10, GRP-11
- pending (unapproved): GRP-7, GRP-8, GRP-9, GRP-10, GRP-11

## QA gate
- status: unresolved
- newest audit: E:\2. UM-Student\MSC Dissertation\Urban-Vision-Benchmark\Scripts\MTSD-Scripts\MTSD-AnnotationQA\outputs\audit-20260716-212105 (2026-07-16T21:21:05)
- outstanding findings: {'invalid_attribute_values': 30, 'duplicate_candidates': 18}
- unaudited discovered groups: none

## Attribute crop manifest
- ingested groups: GRP-1, GRP-2, GRP-3, GRP-5, GRP-6, GRP-7, GRP-8, GRP-9
- split sizes: {'train': 11025, 'val': 1397, 'test': 1440}
- pending auto-ingest: GRP-10, GRP-11

## Prepared datasets
- not built yet

## Vocabulary
- consistent across groups: True
