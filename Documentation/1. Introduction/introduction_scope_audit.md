# Introduction — scope audit

- **Repository commit inspected:** `cee7dc6a4928a721396baf4f58f07ffab5a83c72` ("Add dataset integrity reports and issues for MDWD and MTSD datasets", 2026-07-11 21:14 +0200; working tree clean).
- **Inspection date:** 11 July 2026.
- **Chapter file:** `Documents/Dissertation-Drafts/Introduction/introduction.tex`.

## Files inspected

- `Summary.md`, `README.md` (read in full earlier this session; **counts partially stale** — see conflicts)
- `Scripts/FinalEvaluation/config/research_questions.yaml` (RQ1–RQ5)
- `Documents/Final-Reports/dataset_integrity_report.md` + `.json` (generated 2026-07-11 20:59 — newest primary evidence)
- `Datasets/MTSD/Annotations/GRP-*/Final-QA/QA-GRP*.json` — **counted directly** (primary artefact)
- `Documents/MDWD-EDA/eda_summary.json` (MDWD counts)
- `Documents/MTSD-EDA/GeneratedCSVs/` (class/attribute distributions — pre-GRP-5 snapshot, used only qualitatively)
- `Documents/mtsd_config.xml` (12 classes, 4 attributes, annotator instructions)
- `Scripts/MTSD-Scripts/AttributeClassification/README.md`, `config/default.yaml` (6 variants; heads; `Damaged-Unknown` = drop value)
- `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/README.md` + `config/default.yaml` (13-model matrix; QA gate)
- `Scripts/Other-Scripts/PromptDetect/backend.py`, `Documents/ModelArchitectures.md`, `batch_evaluation/prompt_protocols/dissertation_protocol.yaml`
- `Documents/Final-Reports/Dissertation-Dashboard/latest/` (existence checked; values not transcribed)
- Existing drafts: `Documentation/Background-LiteratureReview/{background.tex, literature_review.tex, background_literature_references.bib}` (labels `ch:background`, `ch:literature-review` confirmed)
- **Not available in workspace:** `MDWD_EUVIP26.pdf`, MSc progress report, `MScAISupervisionDeclarationForm(Full-time).pdf`, any existing dissertation abstract/introduction (searched repo + parent directory; only unrelated BSc FYP files found).

## Canonical dataset facts used in the chapter

| Fact | Value used | Primary source |
|---|---|---|
| MDWD unique source photographs | **3,598** | `dataset_integrity_report.md` (2026-07-11) + `eda_summary.json` |
| MDWD classes | 5: Mixed Waste, Organic Waste, Recyclable Material, Orange CMD, Other Waste | integrity report per-class table; `data.yaml` |
| MDWD augmented exports | mentioned only as "larger, described in Methodology" — no counts given in the chapter | eda_summary.json (30,225 / 211,512 export-level) |
| MTSD raw images | **7,492** across **11 groups** | Summary.md + group listing (unchanged by GRP-7 annotation) |
| MTSD QA-annotated images | **3,808** across **6 groups** (GRP-1/2/3/5/6/7) | direct count of `QA-GRP*.json` (724+630+617+657+470+710) |
| MTSD annotated instances | **10,263** | direct count (1960+1621+1844+1841+1106+1891) |
| MTSD object classes | 12 | every QA JSON has 12 categories; mtsd_config.xml |
| MTSD attributes | view angle {Front, Side, Back}; mounting {Pole-, Wall-Mounted}; condition {Good, Weathered, Heavily Damaged}; shape {Circular, Quadrangle, Triangular, Octagonal, Pentagon} | AttributeClassification config (Damaged-Unknown = drop value → **excluded**) |
| Attribute crops | not stated in chapter (avoids confusing crops with annotations); historical manifest = 5,273 crops (GRP-1–3) | AttributeClassification README |
| Waste statistic | Malta 621 kg/resident 2024 (574 kg incl. tourists); EU average 517 kg | NSO Malta "Municipal Waste: 2024"; Eurostat news 2026-03-30 — **verified 11 July 2026**, matches EUVIP paper values; newest comparable figures |

## Evaluated task families / evidence status (per repo at this commit)

| Track | Status | Treatment in Introduction |
|---|---|---|
| MDWD dataset + EDA + integrity + leakage | executed (final) | in scope; Obj. 1 |
| MTSD dataset + QA + EDA | executed (6 groups QA'd; EDA CSVs pre-date GRP-5/7) | in scope; Obj. 1 |
| MDWD supervised benchmark (YOLO11/12/26 n–l; RF-DETR nano) | executed; RF-DETR small/medium **not** executed, nano not normalised | Obj. 2 phrased "across the supported model capacities" without claiming full RF-DETR ablation |
| MTSD supervised detection | **not executed** (prepared dataset PENDING per integrity report) | **excluded from objectives**; not claimed anywhere |
| PromptDetect fixed-protocol evaluation | protocol implemented + dry-run verified; only a 5-image pilot executed | Obj. 3 (planned core scope; **pending execution** — flagged in writing notes) |
| Attribute classification (6 variants) | executed (historical GRP-1–3 snapshot; final-scope rerun deferred) | Obj. 4 |
| Inference-speed benchmark | implemented, **not executed** | Obj. 4 trade-off wording (**pending** — flagged) |
| Annotation-effort report | executed (manual hours "not recorded") | evaluation-dimension mention only |
| Bootstrap uncertainty / robustness slices | executed over available evidence | "where applicable" wording |

## Research questions → objectives mapping

RQ1 → Objective 2 (MDWD supervised comparison). RQ2 → Objective 3 (prompt-based). RQ3 → Objective 4 (adaptation strategies). RQ4 → folded into Objective 4 (deployment trade-offs). RQ5 → folded into Objective 1 (QA/integrity/leakage). Five RQs consolidated into four objectives as instructed.

## Contributions stated (result-free)

1. Two Malta-specific datasets with operational taxonomies + MTSD per-sign attribute schema.
2. Reproducible supervised benchmark (3 YOLO generations + RF-DETR) on MDWD.
3. Fixed-protocol integration of promptable/grounded/reasoning models against municipal ground truth.
4. Shared-backbone multi-head attribute framework comparing probe/LoRA/full fine-tune across 3 representation families.
5. Deployment-oriented evaluation framework + audited datasets/protocols/tooling.

## Outdated proposal elements (excluded)

Road-surface degradation, temporal object persistence, saliency prioritisation and video-based modelling appear nowhere in the current repository scope and were **not** mentioned in the chapter (original proposal PDF unavailable in workspace; exclusion based on repository absence, as instructed).

## Scope conflicts identified

1. **Stale documentation counts:** `Summary.md`/`README.md` state 3,098 QA images / 8,372 annotations over 5 groups; primary artefacts now show **3,808 / 10,263 over 6 groups** (QA-GRP7.json added in commit `1b2c918a`). The chapter uses the primary-artefact counts with "at the time of writing".
2. **QA gate lag:** `MTSD-SupervisedDetection/config/default.yaml` approved-groups lock still lists GRP-1/2/3/5/6; GRP-7 not yet audit-approved (`update_qa_gate.py` refresh pending). Does not affect the Introduction's dataset facts, but affects any future "QA-approved for training" claim.
3. **MDWD paper vs repository counts:** task brief cites the EUVIP paper's 3,697 source images / 11,461 instances. The repository v20 export contains **3,598** unique sources, and no repository artefact reproduces the 11,461 figure (export-level box counts are augmentation-inflated). The chapter therefore uses 3,598 and gives **no** source-level instance count. Resolve against the paper before submission (paper PDF unavailable in workspace).
4. **MTSD EDA CSVs** still reflect the pre-GRP-5 snapshot (1,971 images / 5,457 boxes); not used for counts.
5. **Integrity report status FAIL (MDWD out_of_range_box=15):** a 2026-07-11 audit note in `Summary.md` re-interprets these as polygon-format label lines misread as boxes; the Introduction does not mention integrity findings either way (results-adjacent), but Methodology must reconcile the FAIL status with that note.
6. **Dissertation title** differs between the task brief ("A Comparative Study of Vision-Based Perception Paradigms…") and `research_questions.yaml` ("Computer Vision for Urban Waste and Infrastructure Monitoring in Malta"). The chapter text does not name the title; front matter must pick one.

## Result-related content deliberately excluded

All experimental outcomes: MDWD benchmark values and rankings; attribute-classification macro-F1 values and variant ordering; leakage-sensitivity delta values; PromptDetect pilot outcomes; robustness/uncertainty findings; annotation-effort findings; any best-model statement; any success claim for objectives. The leakage analysis is referenced only as an activity (Obj. 1), never with its finding.

## Cross-references requiring confirmation

- `ch:background`, `ch:literature-review` — exist in current drafts (`Documentation/Background-LiteratureReview/`).
- `ch:methodology`, `ch:evaluation`, `ch:conclusion` — **chapters not yet written**; labels are the assumed convention and must be confirmed when those chapters are created.
- Bib keys resolve in `Documentation/Background-LiteratureReview/background_literature_references.bib` (extended 11 July 2026 with `nso2026municipalwaste`, `eurostat2026municipal`).
