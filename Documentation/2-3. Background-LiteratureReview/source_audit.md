# Source audit — Background & Literature Review chapters

- **Repository commit inspected:** `1b2c918a547af3f5f126c2ac21b1ec805e7458fd` (2026-07-11)
- **Literature search date:** 11 July 2026
- **Verification legend:**
  - **W** = verified online during this writing task (11 July 2026) against a primary source (arXiv page, publisher page, official site).
  - **K** = canonical, stable publication cited from established knowledge; bibliographic details are standard and widely reproduced. Spot-check page numbers/DOIs against the publisher before final submission.
  - **M** = moderate confidence; details believed correct but **must be re-verified before submission** (flagged explicitly).

Status column: PR = peer-reviewed; PP = preprint; OD = official documentation / model card; GOV = government/legislative; NPR = non-peer-reviewed report.

## 1. Maltese institutional and legislative sources

| Key | Title / description | Year | Type | Ver. | Used in | Supports |
|---|---|---|---|---|---|---|
| `cmd2026web` | Cleansing and Maintenance Division — publiccleanliness.gov.mt | 2026 | GOV | W | BG 1.1, 1.3 (Orange CMD) | CMD remit: arterial-road cleaning, illegal-dumping removal, beach cleaning, services to central govt + local councils. Note: official name is **Cleansing** and Maintenance Division. |
| `wastecollection2026web` | wastecollection.mt national schedule portal | 2026 | GOV | W | BG 1.1, 1.3 | Bag colours: white=organic, grey/green=recyclables, black=mixed; glass on designated days; nationally harmonised schedule via councils. |
| `era2023web` | ERA press release — waste-separation enforcement | 2023 | GOV | W | BG 1.1, 1.3 | Mandatory household waste separation in Malta. |
| `wasteserv2026web` | WasteServ Malta website | 2026 | GOV | W (site exists; URL to re-check) | BG 1.1 | WasteServ operates treatment/disposal facilities. **Check exact URL** (wasteservmalta.com vs wasteserv.mt). |
| `transportmalta2021guidelines` | TM Technical Guidelines of Traffic Signs | 2021 | GOV | W | BG 1.1 | TM as regulator; issues sign technical guidelines. Year on page not explicit — listed as accessed 2026. |
| `infrastructuremalta2020spec` | IM Specification for Road Works IM/1200 Traffic Signs | 2020 | GOV | W | BG 1.1 | IM builds/maintains arterial roads; publishes sign specs (Cap. 588). Year approximate — **verify document date**. |
| `localcouncilsact1993` | Local Councils Act, Cap. 363 | 1993 | GOV | W | BG 1.1 | Local councils responsible for residential/local road maintenance. |
| `gdpr2016` | Regulation (EU) 2016/679 | 2016 | GOV | K | BG 1.10; LR 6.1, 6.9 | GDPR obligations for street imagery. |
| `transport2025masterplan` | National Transport Master Plan 2030 (consultation) | 2025 | GOV | W | (context; cited optionally) | TM/IM/local-council responsibility split. Currently **not cited in the chapters** — keep or remove from .bib at final pass. |

## 2. Waste recognition and datasets

| Key | Title | Authors / Venue | Year | Type | Ver. | Used in | Supports / limitations |
|---|---|---|---|---|---|---|---|
| `mittal2016spotgarbage` | SpotGarbage | Mittal et al., UbiComp | 2016 | PR | K | LR 6.1 | Early smartphone garbage-flagging CNN. DOI included (verify). |
| `rad2017computer` | CV system to localize/classify street wastes | Rad et al., ICVS/LNCS 10528 | 2017 | PR | K | LR 6.1, 6.2 | Vehicle-mounted street-litter detection. Page range approximate — **verify**. |
| `proenca2020taco` | TACO litter dataset | Proença & Simões, arXiv 2003.06975 | 2020 | PP | K | LR 6.2 | Outdoor litter masks; material taxonomy; preprint status flagged in text? (cited as dataset; fine) |
| `yang2016trashnet` | TrashNet / CS229 report | Yang & Thung, Stanford | 2016 | NPR | K | LR 6.2 | Clean-background classification; explicitly flagged non-peer-reviewed in the chapter. |
| `kraft2021uavvaste` | UAVVaste | Kraft et al., Remote Sensing 13(5):965 | 2021 | PR | K | LR 6.2 | Aerial litter detection. DOI included (verify). |
| `bashkirova2022zerowaste` | ZeroWaste | Bashkirova et al., CVPR | 2022 | PR | K | LR 6.2 | Recycling-facility segmentation; used to argue conveyor imagery ≠ street imagery. |
| `abdallah2020artificial` | AI in solid waste management: systematic review | Abdallah et al., Waste Management 109 | 2020 | PR | K | LR 6.2 | Field skew toward classification/sorting; street-level rare. DOI included (verify). |
| `lu2022computer` | CV for solid waste sorting: critical review | Lu & Chen, Waste Management 142 | 2022 | PR | M | LR 6.2 | Cited only as pointer to sorting literature. **Verify volume/pages.** |
| `merolla2025improving` | Improving Road Safety with AI | Merolla, Latorre, Salis, Boanelli; Computers 14(3):91 | 2025 | PR | W | LR 6.1, 6.3 | Municipal YOLO-based sign+surface damage pilot (Molise, Italy). DOI verified. |

## 3. Traffic-sign datasets and recognition

| Key | Title | Venue | Year | Type | Ver. | Used in | Notes |
|---|---|---|---|---|---|---|---|
| `stallkamp2012man` | GTSRB (Man vs. computer) | Neural Networks 32 | 2012 | PR | K | LR 6.3 | DOI verified pattern; canonical. |
| `houben2013detection` | GTSDB | IJCNN | 2013 | PR | K | LR 6.3 | Canonical. |
| `mogelmose2012vision` | LISA + survey | IEEE T-ITS 13(4) | 2012 | PR | K | LR 6.3 | Dataset + survey in one; DOI included. |
| `zhu2016traffic` | TT100K | CVPR | 2016 | PR | K | LR 6.3 | Small signs in panoramas. |
| `ertler2020mapillary` | Mapillary TSD | ECCV | 2020 | PR | K | LR 6.3 | 300+ classes, global. |
| `temel2020traffic` | CURE-TSD journal paper | IEEE T-ITS 21(9) | 2020 | PR | **M** | LR 6.3 | Volume/issue/pages from memory — **verify before submission** (may be different issue). |
| `tabernik2020deep` | Large-scale TS detection/recognition | IEEE T-ITS 21(4) | 2020 | PR | W (arXiv 1904.00649 seen) | LR 6.1, 6.3 | 200-category Slovenian inventory; Mask R-CNN based. |
| `balali2015detection` | US sign mapping from Street View | Visualization in Engineering 3:15 | 2015 | PR | M | LR 6.1, 6.3 | **Verify volume/article number.** |

## 4. Supervised detection (canon + models under study)

| Key | Title | Venue | Year | Ver. | Used in | Notes |
|---|---|---|---|---|---|---|
| `girshick2014rich` | R-CNN | CVPR | 2014 | K | LR 6.4 | Canonical. |
| `girshick2015fast` | Fast R-CNN | ICCV | 2015 | K | LR 6.4 | Canonical. |
| `ren2015faster` | Faster R-CNN | NeurIPS | 2015 | K | BG 1.6; LR 6.4 | Two-stage canonical (never "dual detector"). |
| `he2017mask` | Mask R-CNN | ICCV | 2017 | K | BG 1.2; LR 6.3 | Instance segmentation. |
| `liu2016ssd` | SSD | ECCV | 2016 | K | LR 6.4 | Canonical. |
| `lin2017focal` | RetinaNet / focal loss | ICCV | 2017 | K | LR 6.4 | Canonical. |
| `lin2017feature` | FPN | CVPR | 2017 | K | BG 1.6; LR 6.4 | Canonical. |
| `redmon2016you` | YOLO v1 | CVPR | 2016 | K | LR 6.4 | Canonical. |
| `redmon2018yolov3` | YOLOv3 | arXiv | 2018 | K | LR 6.4 | Preprint (flagged). |
| `bochkovskiy2020yolov4` | YOLOv4 | arXiv | 2020 | K | LR 6.4 | Preprint (flagged). |
| `wang2023yolov7` | YOLOv7 | CVPR | 2023 | K | LR 6.4 | Canonical. |
| `wang2024yolov10` | YOLOv10 (NMS-free dual assignment) | NeurIPS | 2024 | K | LR 6.5, 6.6 | Basis of one-to-one YOLO training. |
| `khanam2024yolov11` | YOLOv11 overview | arXiv 2410.17725 | 2024 | K | LR 6.5 | Third-party overview; YOLO11 has no first-party paper. |
| `ultralytics2024yolo11` | YOLO11 docs | Ultralytics docs | 2024 | W | LR 6.5, 6.9 | Vendor documentation (release Sept 2024). |
| `tian2025yolov12` | YOLOv12 attention-centric | arXiv 2502.12524 | 2025 | W | LR 6.5 | Verified authors Tian/Ye/Doermann; numbers (40.6 mAP, 1.64 ms T4) from abstract. Possible NeurIPS 2025 acceptance — **check and upgrade entry if so**. |
| `jocher2026yolo26` | Ultralytics YOLO26 | arXiv 2606.03748 | 2026 | W | LR 6.5, 6.6, 6.9, 6.19 | Verified: authors, dual-head NMS-free, DFL removal, ProgLoss+STAL, MuSGD, 40.9–57.5 mAP / 1.7–11.8 ms T4, five scales. Vendor-authored preprint — treated as vendor claims in text. Release 14 Jan 2026. |
| `ultralytics2026yolo26docs` | YOLO26 end-to-end guide | Ultralytics docs | 2026 | W | LR 6.5 | Vendor documentation. |
| `sapkota2025yolo26` | YOLO26 review | arXiv 2509.25164 | 2025 | W | LR 6.5 | Verified independent authors (Sapkota, Cheppally, Sharda, Karkee); analyses architecture, does not re-measure benchmarks. |

## 5. Transformer detection

| Key | Title | Venue | Year | Ver. | Used in | Notes |
|---|---|---|---|---|---|---|
| `carion2020end` | DETR | ECCV | 2020 | K | BG 1.6; LR 6.4, 6.6 | Canonical. |
| `zhu2021deformable` | Deformable DETR | ICLR | 2021 | K | LR 6.4, 6.7 | Canonical. |
| `zhang2023dino` | DINO detector | ICLR | 2023 | K | LR 6.4, 6.14 | Name-clash warning included in both chapters. |
| `zhao2024detrs` | RT-DETR | CVPR | 2024 | K | LR 6.4 | Canonical. |
| `chen2024lwdetr` | LW-DETR | arXiv 2406.03459 | 2024 | M | LR 6.4, 6.7 | Author list long — **verify author order** on arXiv. |
| `robinson2025rfdetr` | RF-DETR NAS | ICLR 2026; arXiv 2511.09554 | 2026 | W | LR 6.7 | Verified: authors (Robinson, Robicheaux, Popov, Ramanan, Peri), ICLR 2026, DINOv2-with-registers backbone + Deformable-DETR decoder + weight-sharing NAS, first real-time 60+ COCO mAP, 29M–128M published sizes. |

## 6. NMS / small objects

| Key | Notes |
|---|---|
| `neubeck2006efficient` | ICPR 2006, canonical (K). LR 6.6, BG 1.6. |
| `bodla2017soft` | Soft-NMS ICCV 2017, canonical (K). LR 6.6. |
| `hosang2017learning` | Learning NMS CVPR 2017, canonical (K). LR 6.6. |
| `cheng2023towards` | Small-object survey, TPAMI 45(11) 2023 (M — **verify pages 13467–13488**). LR 6.1, 6.6. |

## 7. Backbones and SSL

| Key | Ver. | Notes |
|---|---|---|
| `krizhevsky2012imagenet`, `deng2009imagenet`, `he2016deep`, `dosovitskiy2021image` | K | Canonical. |
| `liu2022convnet` | K | ConvNeXt CVPR 2022 — pure ConvNet, explicitly not a hybrid (stated in LR 6.16). |
| `woo2023convnextv2` | K | CVPR 2023. |
| `chen2020simple`, `he2020momentum`, `grill2020bootstrap`, `caron2021emerging`, `he2022masked` | K | Canonical SSL line. |
| `oquab2024dinov2` | K | TMLR 2024 / arXiv 2304.07193. |
| `simeoni2025dinov3` | W | arXiv 2508.10104 verified; 7B ViT, 1.7B images, Gram anchoring; distilled family incl. ViT-B/16 (repo variant). Author list beyond Siméoni abbreviated with "others" — **complete from arXiv before submission**. |
| `lecun2022path` | K | OpenReview position paper (not peer-reviewed; framed as programme statement). |
| `assran2023self` | K | I-JEPA CVPR 2023. |
| `bardes2024revisiting` | K | V-JEPA arXiv 2404.08471. |
| `assran2025vjepa2` | W | arXiv 2506.09985 (June 2025). Author list abbreviated — complete before submission. |
| `murlabadia2026vjepa21` | W | arXiv 2603.14482 verified (title, full author list, March 2026, dense predictive loss + deep supervision + multimodal tokenisers). |

## 8. Adaptation strategies

| Key | Ver. | Notes |
|---|---|---|
| `hinton2015distilling` | K | Distillation preprint. |
| `hu2022lora` | K | LoRA ICLR 2022. |
| `houlsby2019parameter` | K | Adapters ICML 2019. |
| `han2024parameter` | M | PEFT survey arXiv 2403.14608 — **verify author list**. |
| `kornblith2019better`, `he2019rethinking`, `yosinski2014transferable` | K | Canonical transfer-learning evidence. |

## 9. VLM / open-vocabulary / promptable

| Key | Ver. | Notes |
|---|---|---|
| `radford2021learning` | K | CLIP. |
| `zareian2021open` | K | OVD-from-captions CVPR 2021 (origin of OVD framing). |
| `li2022grounded` | K | GLIP CVPR 2022. |
| `minderer2022simple` / `minderer2023scaling` | K | OWL-ViT ECCV 2022 / OWLv2 NeurIPS 2023. |
| `liu2024grounding` | K | Grounding DINO ECCV 2024 (arXiv 2303.05499). Explicitly framed as context, NOT an evaluated model. |
| `ren2024grounded` | K | Grounded SAM arXiv 2401.14159. Context only. |
| `kirillov2023segment` | K | SAM ICCV 2023. |
| `ravi2024sam2` | K | SAM 2 arXiv 2408.00714 (also ICLR 2025 — **upgrade entry if final venue preferred**). |
| `meta2025sam3` | W | arXiv 2511.16719 verified; 848M params; presence head; SA-Co benchmark; ~2× PCS gain; Nov 2025. Author list on arXiv is a large team — bib uses corporate author "Meta AI"; **replace with full author list before submission**. OpenReview listing suggests ICLR 2026 — check final venue. |
| `meta2026sam31` | W | Meta blog + facebook/sam3.1 model card, 27 Mar 2026: drop-in checkpoint, object multiplexing (16 objects/pass), 16→32 fps on H100. Matches repo's ModelArchitectures.md (same architecture, new weights). |
| `bommasani2021opportunities` | K | Foundation-model framing. |
| `brown2020language` | K | Few-shot framing. |
| `ha2018world` | K | World-model origin. NeurIPS 2018 title is "Recurrent World Models Facilitate Policy Evolution" (arXiv version titled "World Models") — bib uses NeurIPS title. |
| `nvidia2025cosmosplatform` | W | arXiv 2501.03575, corporate author NVIDIA. |
| `nvidia2025cosmosreason1` | W | arXiv 2503.15558, corporate author NVIDIA. |
| `nvidia2025cosmosreason2` | W | HF model cards + GitHub (nvidia-cosmos/cosmos-reason2); released Dec 2025, announced CES 2026; Qwen3-VL basis (8B post-trained from Qwen3-VL-8B-Instruct); 2B/8B/32B; no standalone tech report found as of 11 Jul 2026 — cited as official documentation (OD). |
| `nvidia2026locateanything` | W | arXiv 2605.27365 + research.nvidia.com page + HF card; MoonViT + Qwen2.5 decoder; Parallel Box Decoding; up to 2.5× throughput; NVIDIA tech report (PP/OD, not peer-reviewed). Author list not captured — bib uses corporate author; **complete from arXiv before submission**. |
| `bai2025qwen25vl` | K/W | arXiv 2502.13923; cited for Qwen-VL normalised-grid coordinate conventions. |

## 10. Multi-task / attributes

| Key | Ver. | Notes |
|---|---|---|
| `caruana1997multitask` | K | Canonical; DOI included. |
| `ruder2017overview` | K | Preprint overview. |
| `kendall2018multi`, `chen2018gradnorm`, `yu2020gradient`, `standley2020tasks`, `misra2016cross` | K | Canonical MTL line. |
| `vandenhende2021multi` | K | TPAMI survey; bib year 2022 (journal issue) with DOI — verify year consistency (online 2021, issue 2022). |
| `wang2022pedestrian` | M | Pattern Recognition 121:108220 — **verify author list and article number**. |

## 11. Evaluation, quality, bias, edge

| Key | Ver. | Notes |
|---|---|---|
| `everingham2010pascal`, `lin2014microsoft` | K | Canonical metrics sources. |
| `padilla2021comparative` | K | Electronics 10(3):279. |
| `guo2017calibration` | K | ICML 2017. |
| `sokolova2009systematic` | K | IPM 45(4). |
| `northcutt2021pervasive` | K | NeurIPS D&B 2021. |
| `gebru2021datasheets` | K | CACM 2021. |
| `torralba2011unbiased`, `devries2019does` | K | Dataset bias / geographic robustness. |
| `shi2016edge`, `satyanarayanan2017emergence`, `chen2019deep`, `murshed2021machine` | K | Edge-computing canon; DOIs included. |
| `howard2017mobilenets`, `sandler2018mobilenetv2`, `han2016deep`, `jacob2018quantization` | K | Efficiency canon. |

## Known conflicts and cautions

1. **Vendor vs independent evidence (YOLO26, RF-DETR, LocateAnything, SAM 3.x, Cosmos).** All headline numbers for 2025–2026 models originate from their creators. The chapters consistently attribute them ("the vendor reports…", "the authors report…") and avoid superiority language. RF-DETR is the only one with peer review (ICLR 2026); SAM 3 may also be — check final venue.
2. **DINO name clash** (detector vs SSL family) — explicitly disambiguated in both chapters.
3. **ConvNeXt** is described as a pure convolutional architecture informed by ViT-era design, never as a hybrid.
4. **Faster R-CNN** is described only as a two-stage detector.
5. **Orange CMD**: no public authoritative source describing orange CMD bags was found; the chapter grounds the class in the repository's dataset definition plus the CMD's official remit, and explicitly frames it as an operational dataset category. Do not strengthen this wording without a new source.
6. **Corporate-author entries** (`meta2025sam3`, NVIDIA entries, DINOv3 "others") must be expanded to full author lists before submission; arXiv pages provide them.
7. **Entries flagged M** (temel2020traffic, balali2015detection, lu2022computer, cheng2023towards, chen2024lwdetr, wang2022pedestrian, han2024parameter, rad2017computer page range) need a bibliographic re-check before the final bibliography is frozen.
8. `transport2025masterplan` is in the .bib but currently uncited — remove or cite at final pass.

## Revision 2026-07-16 (commit 2d70be27) — sources added

| Key | Title / description | Year | Type | Ver. | Used in | Supports |
|---|---|---|---|---|---|---|
| `fu2026lingbot` | LingBot-Vision: Vision Pretraining for Dense Spatial Perception (Robbyant/Ant Group tech report + HF model cards vit-base/vit-large + GitHub) | 2026 | NPR/OD | W (model card verified 16 July 2026; tech report listed as forthcoming) | LR §LingBot; BG §7; Methodology attribute track | Masked boundary modelling objective; teacher–student with semantic self-distillation + categorical boundary-field guidance; ViT-S/B/L/g at patch 16; recommended frozen normalised patch-token readout. Architecture details (RoPE, 4 register tokens, 512 px global crops, embed dims 768/1024) additionally verified from the installed pinned package configs. |
The `fu2026lingbot` key was added to `background_literature_references.bib`.
