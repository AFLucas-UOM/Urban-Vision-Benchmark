# Coverage matrix — Background & Literature Review

Maps every dissertation topic to its key concepts, principal sources, chapter locations, the research question it supports, and a coverage judgement.

Section labels: **BG** = `background.tex` (`\chapter{Background}`), **LR** = `literature_review.tex` (`\chapter{Literature Review}`). Section numbers refer to order of `\section` within each chapter.

RQ ids are those in `Scripts/FinalEvaluation/config/research_questions.yaml` (commit 1b2c918a).

| Dissertation topic | Key concepts | Principal sources | BG section | LR section | RQ | Coverage |
|---|---|---|---|---|---|---|
| Municipal monitoring rationale | inspection frequency, coverage, maintenance prioritisation, visual evidence | chen2019deep, merolla2025improving | BG §1 | LR §1 | RQ1–RQ4 | Strong |
| Maltese institutional context | CMD, WasteServ, Transport Malta, Infrastructure Malta, local councils, bag colours | cmd2026web, wastecollection2026web, era2023web, wasteserv2026web, transportmalta2021guidelines, infrastructuremalta2020spec, localcouncilsact1993 | BG §1.1 | LR §1 (gap: no Malta CV evidence) | all | Strong (official sources; multi-agency structure stated, not simplified) |
| CV task definitions | classification, detection, segmentation, localisation, OVD, prompting, VQA, attribute classification; multi-class vs multi-label vs multi-attribute | krizhevsky2012imagenet, everingham2010pascal, lin2014microsoft, he2017mask, zareian2021open, kirillov2023segment | BG §2 | — (definitions belong to BG) | all | Strong |
| MDWD taxonomy | 5 operational classes; presentation-based annotation; confusion sources | wastecollection2026web, era2023web, cmd2026web + repo dataset definitions | BG §3 | LR §2 (dataset gap) | RQ1, RQ2 | Strong (Orange CMD deliberately conservative) |
| MTSD object taxonomy | 12 classes incl. Back-/Other-Unknown; selection rationale | transportmalta2021guidelines + repo mtsd_config.xml | BG §4 | LR §3 (dataset gap) | RQ2, RQ3 | Strong |
| MTSD attribute taxonomy | view angle, mounting, condition (Good/Weathered/Heavily Damaged boundary), shape (Quadrangle umbrella) | repo config/default.yaml + annotation instructions; merolla2025improving (condition literature) | BG §4.1 | LR §3, §18 | RQ3 | Strong (Damaged-Unknown excluded per QA decision — see writing_notes) |
| Detector families | one/two-stage, anchors, FPN, set prediction, matching, NMS vs end-to-end | ren2015faster, redmon2016you, liu2016ssd, lin2017focal, lin2017feature, carion2020end, neubeck2006efficient | BG §5 | LR §4, §6 | RQ1 | Strong |
| YOLO11 | C3k2 lineage, no first-party paper | ultralytics2024yolo11, khanam2024yolov11 | BG §5 (family level) | LR §5 | RQ1 | Moderate (no peer-reviewed primary source exists) |
| YOLO12 | area attention, R-ELAN, speed claims | tian2025yolov12 | BG §5 | LR §5 | RQ1 | Strong |
| YOLO26 | end-to-end NMS-free, DFL removal, ProgLoss/STAL, MuSGD, edge claims | jocher2026yolo26, ultralytics2026yolo26docs, sapkota2025yolo26, wang2024yolov10 | BG §5 | LR §5, §6, §9 | RQ1, RQ4 | Strong on architecture; vendor-only benchmarks explicitly flagged |
| NMS and end-to-end detection | duplicate suppression, crowded scenes, one-to-one assignment; grouped bags / clustered signs | neubeck2006efficient, bodla2017soft, hosang2017learning, carion2020end, wang2024yolov10, jocher2026yolo26 | BG §5 | LR §6 | RQ1, RQ2 | Strong |
| RF-DETR | LW-DETR + Deformable-DETR lineage, DINOv2 backbone, NAS, scale caveats | robinson2025rfdetr, chen2024lwdetr, zhu2021deformable, oquab2024dinov2 | BG §5 | LR §7 | RQ1 | Strong |
| Transfer learning vs scratch | pretraining benefit conditions, domain shift | yosinski2014transferable, kornblith2019better, he2019rethinking, torralba2011unbiased, devries2019does | BG §7 | LR §8 | RQ1, RQ3 | Strong |
| Edge deployment | edge/cloud trade-offs, quantisation/pruning/distillation, "real-time" caveats | shi2016edge, satyanarayanan2017emergence, chen2019deep, murshed2021machine, han2016deep, jacob2018quantization, howard2017mobilenets, hinton2015distilling | BG §9 | LR §9 | RQ4 | Strong |
| SAM line | promptable segmentation → concept segmentation; SAM 3 presence head; SAM 3.1 multiplexing; confidence available | kirillov2023segment, ravi2024sam2, meta2025sam3, meta2026sam31 | BG §6 | LR §10 | RQ2 | Strong |
| Grounded / OVD context | GLIP, OWL-ViT/v2, Grounding DINO, Grounded SAM; terminology separation | li2022grounded, minderer2022simple, minderer2023scaling, liu2024grounding, ren2024grounded, zareian2021open | BG §6 | LR §11 | RQ2 | Strong (explicitly context-only, not evaluated models) |
| LocateAnything-3B | MoonViT+Qwen2.5, Parallel Box Decoding, no confidence, tech-report status | nvidia2026locateanything, bai2025qwen25vl | BG §6 | LR §12 | RQ2 | Strong (vendor-report status flagged) |
| Cosmos Reason2 / WFMs | world models, WFM ecosystem, physical-AI reasoning VLM, 2B/8B/32B, JSON grounding, no per-box confidence, comparison methodology | ha2018world, nvidia2025cosmosplatform, nvidia2025cosmosreason1, nvidia2025cosmosreason2, guo2017calibration | BG §6 | LR §13 | RQ2, RQ4 | Strong (terminological care WFM ≠ VLM ≠ OVD) |
| SSL and DINOv3 | contrastive → self-distillation → DINOv2 → DINOv3 Gram anchoring; dense-feature quality; DINO name clash | chen2020simple, he2020momentum, grill2020bootstrap, caron2021emerging, he2022masked, oquab2024dinov2, simeoni2025dinov3, zhang2023dino | BG §7 | LR §14 | RQ3 | Strong |
| JEPA and V-JEPA 2.1 | latent prediction, video SSL, dense loss in 2.1; static-crop relevance question | lecun2022path, assran2023self, bardes2024revisiting, assran2025vjepa2, murlabadia2026vjepa21 | BG §7 | LR §15 | RQ3 | Strong (static-transfer gap explicitly stated) |
| ConvNeXt | modernised pure ConvNet; not a hybrid; edge behaviour | liu2022convnet, woo2023convnextv2, he2016deep | BG §7 | LR §16 | RQ3, RQ4 | Strong |
| Adaptation strategies | linear probe, LoRA, full fine-tune; capacity confound; vision-LoRA evidence gap | hu2022lora, houlsby2019parameter, han2024parameter, kornblith2019better, he2019rethinking, chen2020simple | BG §7 | LR §17 | RQ3 | Strong |
| Multi-head vs single-head | hard sharing, negative transfer, loss balancing, deployment | caruana1997multitask, ruder2017overview, kendall2018multi, chen2018gradnorm, yu2020gradient, standley2020tasks, vandenhende2021multi, misra2016cross, wang2022pedestrian | BG §8 | LR §18 | RQ3, RQ4 | Strong |
| Evaluation terminology | IoU, P/R/F1, AP, mAP variants, macro/weighted F1, confusion matrices, imbalance; deployment metrics | everingham2010pascal, lin2014microsoft, padilla2021comparative, sokolova2009systematic, guo2017calibration | BG §10 | LR §13 (calibration problem), §19 | RQ1–RQ5 | Strong |
| Operational paradigm comparison | annotation effort, latency, memory, maintenance, prompt sensitivity, reproducibility | murshed2021machine, meta2025sam3, nvidia2025cosmosreason2, jocher2026yolo26, hu2022lora | BG §9 | LR §19 | RQ4 | Strong |
| Dataset integrity / QA / leakage motivation | label errors, datasheets, bias | northcutt2021pervasive, gebru2021datasheets, torralba2011unbiased | — | LR §19, §20 | RQ5 | Moderate (deliberately brief; methodology-chapter territory) |
| Waste datasets | setting taxonomy: clean-background → facility → street; material vs stream labels; policy dependence | yang2016trashnet, proenca2020taco, bashkirova2022zerowaste, kraft2021uavvaste, mittal2016spotgarbage, rad2017computer, abdallah2020artificial, lu2022computer | BG §3 (context) | LR §2 | RQ1, RQ2 | Strong |
| Traffic-sign datasets | GTSRB/GTSDB/LISA/TT100K/Mapillary/CURE-TSD; condition-annotation scarcity | stallkamp2012man, houben2013detection, mogelmose2012vision, zhu2016traffic, ertler2020mapillary, temel2020traffic, tabernik2020deep, balali2015detection, merolla2025improving | BG §4 (context) | LR §3 | RQ2, RQ3 | Strong |
| LingBot-Vision | boundary-centric SSL (masked boundary modelling), RoPE + register tokens, 512 px pretraining, frozen patch-token readout; directed shape/condition hypothesis; vendor-report status | fu2026lingbot | BG §7 (family list) | LR §17 (new LingBot section) | RQ3 | Strong on architecture; vendor-only evidence explicitly flagged |
| Research-gap synthesis | six named gaps → RQ1–RQ5 mapping | (synthesis of all above) | — | LR §20 | all | Strong |

## Known weak spots (deliberate)

| Topic | Why coverage is limited |
|---|---|
| Traffic-sign **attribute** recognition literature | Near-absent in the published literature (this is itself the reported gap); nearest proxy is pedestrian-attribute recognition. |
| Cosmos Reason2 scale-vs-localisation evidence | No published study isolates localisation benefit of scale; stated as unresolved. |
| LocateAnything independent evaluation | Model released mid-2026; no third-party studies exist yet. |
| YOLO11 primary literature | No first-party peer-reviewed paper exists; only vendor docs + third-party overview. |
| Malta-specific CV literature | None found — reported as "no identified study", not "none exists". |
| LingBot-Vision independent evaluation | Model released 2026 with a forthcoming technical report; only the vendor model card/release exists; no third-party studies yet. |

## Revision 2026-07-16 (commit 2d70be27)

- Added LingBot-Vision row (new LR section between ConvNeXt and adaptation strategies; BG representation-family list now names four families).
- Section numbering beyond LR §16 shifted by +1 relative to the 11 July snapshot.
