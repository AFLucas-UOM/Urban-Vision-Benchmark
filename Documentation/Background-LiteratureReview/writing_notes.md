# Writing notes — Background & Literature Review drafts

## Provenance

- **Repository commit inspected:** `1b2c918a547af3f5f126c2ac21b1ec805e7458fd`
  ("Update .gitignore to exclude LabelStudio shared files; …", 2026-07-11 17:13 +0200; working tree clean at inspection time).
- **Inspection and literature-search date:** 11 July 2026.
- **Output location:** `Documentation/Background-LiteratureReview/` at the repository root, per the instruction that all LaTeX files go under the root `Documentation` folder (the folder existed and was empty). The originally suggested `Documents/Dissertation-Drafts/Background-LiteratureReview/` path was **not** used; move the folder if the other convention is preferred — nothing else references these files yet.

## Repository files inspected

- `Summary.md` (full), `README.md` (full)
- `Scripts/FinalEvaluation/config/research_questions.yaml` (RQ1–RQ5 wording used for the gap synthesis)
- `Documents/mtsd_config.xml` (12 detection classes, 4 required attributes, annotator instructions, `Damaged-Unknown` present only in the labelling tool's shape list)
- `Scripts/MTSD-Scripts/AttributeClassification/README.md` + `config/default.yaml` (six variants; LoRA r=8 α=16 on q_proj/v_proj vs fused qkv; `sign_shape.drop_values: [Damaged-Unknown]` with the 2026-07-03 "unlearnable" decision note; valid shape vocabulary = Circular, Quadrangle, Triangular, Octagonal, Pentagon)
- `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/README.md` + `config/default.yaml` (13-model dissertation matrix: yolo11/12/26 n-s-m + rfdetr n-s-m + yolo26l; QA gate)
- `Scripts/Other-Scripts/PromptDetect/backend.py` + `Documents/ModelArchitectures.md` (model registry: facebook/sam3, facebook/sam3.1, nvidia/Cosmos-Reason2-2B/8B/32B on Qwen3-VL, nvidia/LocateAnything-3B on MoonViT+Qwen2.5 with Parallel Box Decoding; SAM confidence available, Cosmos/LocateAnything boxes-only with no per-box confidence; 0–1000 grid conventions)
- `Scripts/Other-Scripts/PromptDetect/batch_evaluation/prompt_protocols/dissertation_protocol.yaml` (class vocabularies; prompts confirm bag-colour conventions: black=Mixed, grey=Recyclable, white=Organic, orange=Orange CMD)
- `Documents/MDWD-EDA/eda_summary.json` (5 classes; 3,598 unique sources; 30,225 images; 211,512 boxes)
- `Documents/MTSD-EDA/GeneratedCSVs/class_frequencies.csv`, `attribute_summary.csv` (class imbalance incl. Tourist Sign 14 boxes; condition imbalance Good 4,126 / Weathered 995 / Heavily Damaged 332 — pre-GRP-5 snapshot)
- Searched for existing dissertation/literature drafts: none found in the repository.

## Length calibration

- `background.tex` ≈ 3,300 words; `literature_review.tex` ≈ 4,700 words; total ≈ 8,000 words — inside the 6,500–8,500 target, expected ≈ 13–15 pages in a one-column dissertation template at ~600 words/page. **No LaTeX template exists in the repo** (the `Documentation` folder was empty), so page count could not be calibrated against the real class file; re-check after first compile and trim LR §4/§17 first if over length.

## Template and citation assumptions (verify on integration)

- Chapters use `\chapter` + `\section`/`\subsection`/`\paragraph` and natbib `\citet`/`\citep` author–year commands, per the task brief. If the UM template uses numeric citations (`biblatex`/IEEE style), the commands still compile but render numerically; no text changes needed.
- Cross-references assume the labels `ch:background` and `ch:literature-review` and that Background precedes the Literature Review; both files cross-reference each other (`\ref{ch:...}`) and Background §-labels are prefixed `sec:bg-`, LR labels `sec:lr-`.
- The `.bib` is self-contained (`background_literature_references.bib`); add to the main document's bibliography resource list.

## Model names verified against primary sources (11 Jul 2026)

| Model | Verified fact | Source |
|---|---|---|
| YOLO11 | Ultralytics, Sept 2024, no first-party paper | Ultralytics docs; arXiv 2410.17725 (third-party) |
| YOLO12 | Tian, Ye & Doermann; arXiv 2502.12524; area attention; 40.6 mAP N-scale | arXiv |
| YOLO26 | Released 14 Jan 2026; end-to-end NMS-free; DFL removed; ProgLoss+STAL; MuSGD; 5 scales; arXiv 2606.03748 (vendor-authored) | arXiv + Ultralytics docs + BusinessWire |
| RF-DETR | Robinson et al., ICLR 2026, arXiv 2511.09554; DINOv2-with-registers backbone; weight-sharing NAS; 60+ COCO mAP first for real-time; sizes ~29M–128M | arXiv/OpenReview/GitHub |
| SAM 3 | arXiv 2511.16719, Nov 2025; 848M; presence head; SA-Co; ~2× PCS gain | arXiv + ai.meta.com |
| SAM 3.1 | 27 Mar 2026; same architecture, checkpoint update; object multiplexing (16 obj/pass); 16→32 fps H100 video | Meta blog + model card |
| LocateAnything-3B | arXiv 2605.27365 (NVIDIA tech report, ~June 2026); MoonViT + Qwen2.5-3B; Parallel Box Decoding; ≤2.5× throughput; boxes/points only, no confidence | arXiv + research.nvidia.com + HF card |
| Cosmos Reason2 | Released Dec 2025 (CES 2026 announce); Qwen3-VL-based; 2B/8B/32B; no standalone tech report as of search date — cited via model cards/GitHub; lineage via Cosmos platform (arXiv 2501.03575) and Cosmos-Reason1 (arXiv 2503.15558) | HF cards + GitHub + NVIDIA pages |
| DINOv3 | arXiv 2508.10104, Aug 2025; 7B ViT / 1.7B images; Gram anchoring; distilled ViT-B/16 variant = repo's backbone | arXiv |
| V-JEPA 2.1 | arXiv 2603.14482, Mar 2026; Mur-Labadia et al.; dense predictive loss + deep supervision + multimodal tokenisers | arXiv |
| V-JEPA 2 | arXiv 2506.09985, Jun 2025 | arXiv |

## Institutional-responsibility findings (Malta)

- Official body name is **Cleansing and Maintenance Division** (not "Cleaning"); remit = public cleansing (arterial roads, illegal dumping removal, beaches, public spaces), providing services to central government **and** local councils. Source: publiccleanliness.gov.mt.
- Household waste: separation mandatory; **white = organic, grey/green = recyclables, black = mixed**; glass separate days; kerbside collection organised through local/regional councils on a nationally harmonised schedule (wastecollection.mt). WasteServ operates treatment facilities.
- Roads/signs: **Transport Malta** = regulator + technical guidelines for signs; **Infrastructure Malta** = arterial/distributor road building & maintenance + sign specifications (IM/1200); **local councils** = residential/local roads (Local Councils Act Cap. 363). The chapters state this as a shared, road-classification-dependent structure and avoid attributing everything to the CMD.
- **Orange CMD caution:** no public authoritative page describing the CMD's orange bags was found. The Background grounds the class in (a) the repository's own class definition and prompt protocol ("orange garbage bag") and (b) the CMD's verified cleansing remit, and explicitly frames it as an operational dataset category identified by colour + context, not a claim about contents. If the annotator instructions or a CMD source later specify the exact bag usage, tighten BG §3 "Orange CMD" accordingly.

## Damaged-Unknown decision (internal note)

`Damaged-Unknown` appears in the Label Studio config as a shape fallback, but the latest QA state (config `drop_values`, decision of 2026-07-03; audit-20260709-201340 found 5 instances flagged as invalid drop values pending review) treats it as **not a valid final class**. The chapters therefore present the shape vocabulary as exactly five classes and never mention `Damaged-Unknown`; it belongs in the Methodology QA discussion if anywhere.

## Boundary decisions (what was deliberately kept out)

- No repository results anywhere in either chapter: no macro-F1 values, no mAP values, no leakage numbers or QA counts, no pilot outcomes, no W&B details, no training hyperparameters, no split procedure, no augmentation recipe. The MDWD/MTSD sections describe taxonomy and purpose only.
- Grounding DINO, GLIP, OWL-ViT, Grounded SAM are context only, and LR §11 says so explicitly.
- The multi-head design choice is described conceptually in BG §8; its justification is deferred to Methodology.
- V-JEPA pseudo-clip construction is not described; LR §15 only notes the general clip-input cost of applying video encoders to stills.
- The Background mentions dataset selection rationale ("selected as recurrent, driver-relevant… within the study's collection scope") in line with the EDA class frequencies, without claiming island-wide statistical representativeness.

## Open items before submission

1. Expand corporate-author bib entries (Meta AI, NVIDIA, DINOv3 "others") to full author lists from arXiv.
2. Re-verify the eight **M**-flagged entries in `source_audit.md` (temel2020traffic, balali2015detection, lu2022computer, cheng2023towards, chen2024lwdetr, wang2022pedestrian, han2024parameter, rad2017computer pages).
3. Check final venues: YOLOv12 (possible NeurIPS 2025), SAM 3 (possible ICLR 2026), SAM 2 (ICLR 2025) and upgrade entries from preprint to conference where applicable.
4. Decide whether to keep `transport2025masterplan` (currently uncited) and confirm the exact WasteServ URL.
5. Compile in the UM template; check page count (target 12–15 pp) and hyphenation of long model names; confirm `\paragraph` styling suits the template (BG §3 class descriptions).
6. If the MTSD annotation scope changes (GRP-4/7–11 added), the Background needs no numeric edits (it cites no counts), but re-check the "recurrent… within the study's collection scope" phrasing against the refreshed EDA.
