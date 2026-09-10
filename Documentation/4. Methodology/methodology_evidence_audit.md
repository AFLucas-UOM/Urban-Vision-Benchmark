# Methodology — evidence audit

- **Repository commit inspected:** `cee7dc6a4928a721396baf4f58f07ffab5a83c72` (2026-07-11 21:14 +0200; tree clean).
- **Inspection date:** 11 July 2026.
- **Drafting mode:** final-state assumption (all 11 MTSD groups annotated and all final runs executed) — see `methodology_writing_notes.md` for the placeholder ledger.

Every claim in `methodology.tex` traces to one of the artefacts below, ranked by the mandated evidence hierarchy (run config > manifest > checkpoint metadata > source code > result metadata > central YAML > docs > Summary.md > archives).

## Verified claims and their sources

| Claim in chapter | Evidence (tier) | Notes |
|---|---|---|
| Hardware: RTX 4090 24,564 MiB; Ryzen 9 7900X3D; 128 GB RAM; Windows 11 Pro | live `nvidia-smi` + `Win32_Processor`/`Win32_ComputerSystem` (tier 3-equivalent, direct) | driver 610.62 not quoted in chapter |
| Env versions (4 conda envs) | direct interpreter introspection of installed packages (tier 3-equiv.) | MDWD env torch is a **cu130 dev build** (2.10.0.dev20251013) — chapter states "PyTorch 2.10.0 (CUDA 13.0)"; exact dev tag in notes |
| YOLO protocol (epochs 100, batch 32, imgsz 640, AdamW, lr0 0.001, lrf 0.01, momentum 0.937, wd 0.0005, warmup 3, patience 10, seed 42, deterministic, amp, workers 8, val IoU 0.7, max_det 300, pretrained) | `Results/MDWD-Runs/YOLO26-EUVIP/E001_*/args.yaml` (tier 1) | identical pattern across run folders |
| All Ultralytics online augmentations disabled | same `args.yaml`: hsv_h/s/v, degrees, translate, scale, shear, perspective, flipud, fliplr, bgr, mosaic, mixup, cutmix, copy_paste, erasing all 0.0 (tier 1) | strong methodological point; verified, not assumed |
| RF-DETR config (epochs 100, batch 32, grad_accum 1, lr 0.001, lr_encoder 0.001, res 640, multi_scale False, expanded_scales False, wd 0.0005, warmup 3, EMA, ckpt interval 10, early stop 10/0.001, seed 42, COCO/MMDetection export, category normalisation) | `RF-DETR.ipynb` shared config + `Results/MDWD-Results/RF-DETR/Model-Size-Comparison/run_configs/E00{1,2,3}*.json` (tier 1) | `do_random_resize_via_padding: false` also verified |
| Gradient checkpointing for RF-DETR | **NOT FOUND** in run configs | task brief's template mentioned it; **omitted from chapter** |
| YOLO parameter counts (11 values) | `Results/MDWD-Results/*/Model-Size-Comparison/*summary*.csv` `parameter_count` column (tier 5, runtime-introspected) | only params/size read; metric columns not used |
| RF-DETR parameter counts | not recorded in run configs | placeholders `\TBD{rfdetr-*-params}` |
| MDWD: 3,598 unique sources; 29,487/369/369; 209,334/1,072/1,106 boxes; per-class table | `dataset_integrity_report.md` (2026-07-11) + `eda_summary.json` (tier 5) | Roboflow README says "30224 images" vs 30,225 counted on disk — off-by-one conflict recorded below |
| MDWD preprocessing (EXIF auto-orient + strip; 640×640 stretch) and augmentation (H-flip 50%, V-flip 50%, brightness ±17%, exposure ±10%, Gaussian blur 0–1 px; ~10 versions/source) | `Datasets/MDWD/MDWD-YOLO26/README.roboflow.txt` (tier 2, export metadata) | **CONFLICT with task brief**: hue ±19°, saturation ±29%, motion blur ≤40 px and 2×2 mosaic are NOT in the v20 export metadata → excluded from the chapter |
| MDWD Roboflow project (um-vbsez / maltese-garbage-bag-detection-xgodx v20, CC BY 4.0) | `data.yaml` (tier 2) | |
| MTSD raw per-group counts (724/630/617/875/657/470/710/703/715/789/602 = 7,492) | Summary.md §2 corroborated by on-disk group listing (tier 8 + direct) | raw counts stable |
| MTSD annotated counts GRP-1/2/3/5/6/7 (3,808 img / 10,263 inst) | direct count of `QA-GRP*.json` (tier 2) | used as fixed values in group table; GRP-4/8–11 are placeholders |
| MTSD supervised config (split 80/10/10 v1 seed 42; photometric-v1 with 7 ops, 2 copies, seed 42; training block; 13-model matrix; ablation trio + optional 26l; W&B project/group) | `MTSD-SupervisedDetection/config/default.yaml` (tier 6) | no final MTSD run configs exist yet (final-state assumption); flagged in notes |
| QA duplicate thresholds 0.95 / 0.75–0.95 | `MTSD-AnnotationQA/config.py` `DUPLICATE_IOU`, `OVERLAP_IOU` (tier 4) | |
| QA three-stage workflow, guarded apply (dry-run default, backups, old-value verification, re-parse, change log) | AnnotationQA scripts + Summary §4 (tier 4/8) | |
| PromptDetect protocol (dissertation-v1; conf 0.30, IoU 0.50, max_det 100; 6 MDWD + 13 MTSD prompts with exact targets; MTSD source = prepared unaugmented test, manifest required, qa_only) | `prompt_protocols/dissertation_protocol.yaml` (tier 1-equivalent, versioned protocol) | "grey recycling bag" spelling verified; mtsd-p12 target list has 11 classes (no Blind-Spot Mirror) — chapter states this |
| Targeted evaluation semantics (target-class GT; FPs on negatives; non-target overlap recording; prompt-class confusion; greedy score-descending one-to-one matching; accuracy = TP/(TP+FP+FN)) | `batch_evaluation/targeted_metrics.py` + `metrics.py` (tier 4) | accuracy formula at metrics.py:65–67 |
| Atomic per-combination persistence and resume | `run_dissertation_protocol.py` combinations/ handling + Summary §5 (tier 4/8) | |
| SAM3/3.1 loading via native `sam3` pkg (shared path), Cosmos = Qwen3-VL boxes-only constant confidence, LocateAnything worker in mtsd-la (transformers 4.57) | `backend.py` + `ModelArchitectures.md` (tier 4/7) | 32B ≈64 GB fp16 requires offload → opt-in heavy |
| Attribute config (crop pad 0.25, min side 16 px, JPEG q95; split hash `mtsd-attr-split-v1` 80/10/10; drop Damaged-Unknown; mask unknown; training block incl. lr trio, cosine+warmup 5, clip 1.0, class-weighted CE mean-1, head weights 1.0, early stop 10/0.001 on val mean macro-F1; augs jitter/rotation 10°/RRC 0.8–1.0/no flip; probe linear; model IDs incl. dinov3 HF id, vjepa2_1_vit_large_384 + dist_vitG_384 ckpt, num_frames 2; LoRA r8 α16 d0.05 q_proj/v_proj vs qkv; lr overrides 2e-4 / heads 4e-4) | `AttributeClassification/config/default.yaml` (tier 6) | matches historical run metadata (Summary §4) |
| Multi-head architecture (pooled feature → one nn.Linear per head; frozen backbones pinned to eval; MaskedMultiTaskLoss with −1 masking, per-head CE, weighted sum) | `mtsd_attr/multihead_model.py` (tier 4) | loss equations written to match code |
| Pooling (DINOv3 pooler/CLS; ConvNeXt avgpool 768-d; V-JEPA mean-pool) | `mtsd_attr/backbones/*.py` (tier 4) | V-JEPA mean-pool inferred from backbone wrapper docstring pattern — re-verify exact pooling line before submission |
| Attribute trainable params (9,997 / 304,909 / 13,325 / 799,757 / 9,997 / 27,828,589) | Summary §4 table (tier 8) reproducing run introspection; 9,997 = 13×769 arithmetic check passes | final-scope rerun expected to match (architecture unchanged) |
| Robustness (min_support 15, working-conf 0.25, seed 42, dataset-derived terciles, COCO sizes at 640, stored-predictions-only) | `robustness_slice_analysis.py` CLI defaults + docstring (tier 4) | |
| Bootstrap (2,000 resamples, 95%, seed 42, paired identical indices, fraction favouring A) | `bootstrap_uncertainty.py` defaults + docstring (tier 4) | current attribute route reconstructs from confusion matrices; chapter states final route uses stored per-crop records per final-state instruction — flagged in notes |
| Inference benchmark (test split, 50 images, seeded 42, warmup 3, batch 1 primary + optional 4/8 fallback-to-1, imgsz 640, conf 0.30, CUDA sync, memory reset after warmup, saved image list, timestamped outputs) | `inference_speed_benchmark.py` CLI defaults + `timed_loop` (tier 4) | |
| Annotation-effort report (recorded hours only, no composite score) | `annotation_effort_report.py` + FinalEvaluation README (tier 4/7) | |
| GDPR workflow (faces/plates/QR; SAM 3.1 or classic backends; preview→verify→apply; guarded stale previews) | `GDPR-Compliance/README.md` (tier 7) | chapter avoids claiming full anonymisation |
| W&B projects (3) | configs + Summary (tier 6/8) | |

## Material conflicts found

1. **MDWD augmentation: task brief vs export metadata.** The brief (quoting the EUVIP paper) lists hue ±19°, saturation ±29%, motion blur ≤40 px and 2×2 mosaic. The v20 export README lists only H/V flip 50%, brightness ±17%, exposure ±10%, Gaussian blur 0–1 px. Per the hierarchy (export metadata > prose), the chapter includes **only the export-verified operations**. If the paper's pipeline described an earlier dataset version, the dissertation must consistently describe v20.
2. **MDWD export image count.** Roboflow README: "30224 images"; on-disk EDA count: 30,225 (29,487+369+369). Chapter uses the on-disk split counts and never quotes the README total.
3. **DGX-hosted runs vs hardware restriction.** `Results/MDWD-Runs/` contains both `YOLO26-EUVIP` and a second YOLO26 suite executed on other hardware; consolidated CSVs exist for both. Under the task's hardware restriction the chapter describes the single-workstation protocol only and takes YOLO26 settings from the EUVIP `args.yaml` (identical protocol fields). The final dissertation must choose the workstation-run suite for headline evidence or amend the environment section — recorded here as the audit's most significant unresolved tension.
4. **QA gate currently unresolved / five-group scope.** `config/default.yaml` (`approved_groups` GRP-1/2/3/5/6) and `qa_gate.yaml` (`resolution_status: unresolved`, audit predates GRP-6/7) contradict the final-state assumption. The chapter describes the gate as resolved for eleven groups per the drafting instruction; reality lags. `update_qa_gate.py --apply` + a fresh audit must precede the final runs.
5. **`Datasets/MTSD/Prepared/` does not exist yet** (integrity report: PENDING). All prepared-dataset counts are placeholders; the preparation *procedure* is fully implemented and was described from code and config.
6. **Attribute bootstrap unit.** Current implementation reconstructs attribute samples from stored confusion matrices; the chapter (final-state) states per-crop stored predictions. If the final round does not emit per-sample records, the chapter sentence must be revised to the reconstruction wording.
7. **RF-DETR "momentum 0.937 / warmup_momentum / warmup_bias_lr"** appear in the notebook config dict but are Ultralytics-style fields not obviously consumed by `rfdetr` 1.3.0; the chapter therefore lists momentum only "where applicable" and omits these for RF-DETR.
8. **MDWD annotation platform/protocol detail** (reviewer counts, the "~20% visible-object rule"). No repository artefact documents the 20% rule; EUVIP PDF unavailable in workspace. **Excluded from the chapter**; placeholders cover collection period/devices.
9. **YOLO11l/YOLO26l on MDWD** exist in the archives, but the task brief's final matrix for MDWD lists n/s/m (+26l). The chapter's matrix table follows the brief's final matrix (11 YOLO rows incl. 26l + 3 RF-DETR); the extra archived scale is simply not claimed.
10. **Summary.md staleness** (five-group counts, 3,098/8,372) superseded by direct QA JSON counts (six groups on disk, 3,808/10,263) — Summary not used for these numbers.

## Revision 2026-07-16/17 (commit 2d70be27) — new/updated claims

| Claim in chapter | Evidence (tier) | Notes |
|---|---|---|
| LingBot-Vision identifiers, loader, pooling, RoPE/register/512 px facts, frozen+LoRA-only support | `mtsd_attr/backbones/lingbot_backbone.py` + installed pinned `lingbot_vision` package configs (tier 4) + official HF model card (primary source, accessed 16 July 2026) | embed dims verified by loader probe; pretraining objective (masked boundary modelling) from the model card |
| 16-variant matrix completed 2026-07-13; per-variant total/trainable params; backends | `outputs/reports/size_ablation.csv` + `size_ablation.md` + `outputs/experiment_log.jsonl` (tier 5, run artefacts) | real runs, not smoke; every variant has best.pt/last.pt + test metrics |
| Snapshot scope 8 groups / 13,862 crops (11,025/1,397/1,440) | `outputs/manifests/manifest.json` updated_at 2026-07-13T20:45:29Z (tier 2) | GRP-10/11 QA arrived after training; pending auto-ingest flagged as a research decision |
| Gradient accumulation 16×2 for ViT-L LoRA + ConvNeXt-Large FT; effective batch 32 | `config/default.yaml` training_overrides (tier 6) + `train_common.py` accumulation loop (tier 4) | |
| V-JEPA 2.1 version-strict loading (no 2.0 fallback for vjepa21_*) | `vjepa_backbone.py` allow_backend_fallback (tier 4) | |

### Conflict status updates

- Conflict 4 (QA gate): still open; fresh audit 2026-07-16 over all 10 Final-QA groups: 30 invalid values (all Damaged-Unknown shapes) + 18 duplicate candidates. Gate refresh (`update_qa_gate.py --apply`) must follow findings resolution — never precede it.
- Conflict 5 (Prepared/ missing): still open.
- NEW: attribute crop manifest auto-ingests newly QA'd groups; completed 16-variant matrix is an 8-group snapshot. Chapter now states the snapshot explicitly; final-scope re-run is a pending research decision.
