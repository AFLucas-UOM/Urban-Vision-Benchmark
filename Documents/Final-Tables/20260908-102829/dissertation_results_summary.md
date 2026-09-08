# Dissertation results summary

Generated: 2026-09-08T10:28:29 - consolidated from existing result files only (no metrics recomputed or invented).


## MDWD supervised detection

| dataset | model_family | model_variant | parameters | model_size_mb | precision | recall | f1 | map50 | map50_95 |
|---|---|---|---|---|---|---|---|---|---|
| MDWD | YOLO11 | yolo11n | 2590815 | 5.22 | 0.9035638191227982 | 0.8036866029135148 | 0.8507037127455663 | 0.8636417638926028 | 0.6888369062440758 |
| MDWD | YOLO11 | yolo11s | 9429727 | 18.29 | 0.9345350765796423 | 0.8350317903928598 | 0.8819858833775887 | 0.8973637859791648 | 0.7346008341140627 |
| MDWD | YOLO11 | yolo11m | 20056863 | 38.64 | 0.9203475253528313 | 0.8200175360752494 | 0.8672905780509078 | 0.8866884997447217 | 0.7353568731041266 |
| MDWD | YOLO11 | yolo11l | 25314335 | 48.83 | 0.9430254248999848 | 0.8613766154054854 | 0.9003537244993949 | 0.8992977873215562 | 0.7614110989418542 |
| MDWD | YOLO12 | yolo12n | 2569023 | 5.26 | 0.9154546014676533 | 0.8216141219033026 | 0.8659996215550535 | 0.8834670451829236 | 0.7121870181049601 |
| MDWD | YOLO12 | yolo12s | 9255071 | 18.06 | 0.9268869459779859 | 0.8458354822242413 | 0.8845083183312759 | 0.8941019091806585 | 0.7413695245455391 |
| MDWD | YOLO12 | yolo12m | 20141343 | 38.88 | 0.9416166668810266 | 0.8759404715528101 | 0.9075919868142031 | 0.9146507932357523 | 0.7655754993119791 |
| MDWD | YOLO26 | yolo26n | 2505750 | 5.14 | 0.9222904546628795 | 0.803722135332503 | 0.8589337736179753 | 0.8697870789738549 | 0.6983701238316875 |
| MDWD | YOLO26 | yolo26s | 9951734 | 19.38 | 0.9469718641738665 | 0.8486864448127379 | 0.8951393265871109 | 0.8998867887087038 | 0.7559666179134011 |
| MDWD | YOLO26 | yolo26m | 21780598 | 42.0 | 0.9405646614668035 | 0.8614048960762952 | 0.8992460511580751 | 0.9073978148901288 | 0.762111243348453 |
| MDWD | YOLO26 | yolo26l | 26184054 | 50.54 | 0.9578149412414303 | 0.8732289459488067 | 0.9135681972513805 | 0.9152102000336587 | 0.7951067059184301 |
| MDWD | YOLO26-DGX | yolo26n | 2505750 | 5.14 | 0.9052559522558431 | 0.8332353816675917 | 0.8677538670065944 | 0.8819945517967597 | 0.7101449387196088 |
| MDWD | YOLO26-DGX | yolo26s | 9951734 | 19.38 | 0.9375666900010955 | 0.8561401190591977 | 0.8950051965560605 | 0.9091723737957915 | 0.7565106419877367 |
| MDWD | YOLO26-DGX | yolo26m | 21780598 | 41.99 | 0.9396810135352093 | 0.8592301594256876 | 0.8976566260803364 | 0.9087364468168133 | 0.7637484286319105 |
| MDWD | YOLO26-DGX | yolo26l | 26184054 | 50.54 | 0.9485779923671703 | 0.8795903108683845 | 0.9127824935072647 | 0.9099584651611433 | 0.7947859960634506 |

*Full table incl. source paths: `mdwd_detection_results.csv`*

## MTSD supervised detection

**PENDING** - MTSD detection: no summary CSVs found under /Volumes/Filis SSD/2. UM-Student/MSC Dissertation/Urban-Vision-Benchmark/Results/MTSD-Results

## MTSD attribute classification

| variant | backbone | adaptation | mean_macro_f1 | view_angle_f1 | mounting_f1 | condition_f1 | sign_shape_f1 | total_params | trainable_params |
|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitb_frozen | dinov3 | frozen | 0.8068 |  |  |  |  | 85670413 | 9997 |
| dinov3_vitl_frozen | dinov3 | frozen | 0.8162 |  |  |  |  | 303142925 | 13325 |
| dinov3_vitb_lora | dinov3 | lora | 0.8835 |  |  |  |  | 85965325 | 304909 |
| dinov3_vitl_lora | dinov3 | lora | 0.8959 |  |  |  |  | 303929357 | 799757 |
| vjepa21_vitb_frozen | vjepa21 | frozen | 0.7888 |  |  |  |  | 86843149 | 9997 |
| vjepa21_vitl_frozen | vjepa21 | frozen | 0.8085 |  |  |  |  | 304694285 | 13325 |
| vjepa21_vitb_lora | vjepa21 | lora | 0.8844 |  |  |  |  | 87138061 | 304909 |
| vjepa21_vitl_lora | vjepa21 | lora | 0.9006 |  |  |  |  | 305480717 | 799757 |
| convnext_base_frozen | convnext | frozen | 0.7816 |  |  |  |  | 87577741 | 13325 |
| convnext_large_frozen | convnext | frozen | 0.7873 |  |  |  |  | 196247245 | 19981 |
| convnext_base_finetune | convnext | finetune | 0.8684 |  |  |  |  | 87577741 | 87577741 |
| convnext_large_finetune | convnext | finetune | 0.8770 |  |  |  |  | 196247245 | 196247245 |
| convnext_base_lora | convnext | lora | 0.8797 |  |  |  |  | 89021581 | 1457165 |
| convnext_large_lora | convnext | lora | 0.8748 |  |  |  |  | 198413005 | 2185741 |
| lingbot_vitb_frozen | lingbot | frozen | 0.8015 |  |  |  |  | 85679629 | 9997 |
| lingbot_vitl_frozen | lingbot | frozen | 0.8118 |  |  |  |  | 303167501 | 13325 |
| lingbot_vitb_lora | lingbot | lora | 0.8854 |  |  |  |  | 85974541 | 304909 |
| lingbot_vitl_lora | lingbot | lora | 0.8921 |  |  |  |  | 303953933 | 799757 |

> Best attribute model: vjepa21_vitl_lora (mean macro-F1 0.9006); weakest head across all variants is 'condition'.

*Full table incl. source paths: `mtsd_attribute_results.csv`*

## PromptDetect (prompt-based detection)

| evaluation_protocol | dataset | split | model | prompt | precision | recall | f1 | ap50 | map50_95 |
|---|---|---|---|---|---|---|---|---|---|
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | black garbage bag | 0.2899 | 0.2665 | 0.2777 | 0.0934 | 0.0544 |
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | grey recycling bag | 0.1777 | 0.1807 | 0.1792 | 0.0517 | 0.0317 |
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | white organic-waste bag | 0.1538 | 0.1798 | 0.1658 | 0.0833 | 0.0418 |
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | orange garbage bag | 0.0795 | 0.2436 | 0.1199 | 0.0453 | 0.0325 |
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | domestic waste | 0.8116 | 0.292 | 0.4295 | 0.2486 | 0.1537 |
| targeted-v1 | MDWD | test | Cosmos Reason2 2B | garbage | 0.8083 | 0.3011 | 0.4387 | 0.2551 | 0.158 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | black garbage bag | 0.4073 | 0.6317 | 0.4953 | 0.3212 | 0.2042 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | grey recycling bag | 0.2485 | 0.5294 | 0.3383 | 0.1653 | 0.1055 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | white organic-waste bag | 0.2668 | 0.7135 | 0.3884 | 0.2267 | 0.1471 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | orange garbage bag | 0.1129 | 0.5385 | 0.1867 | 0.133 | 0.0784 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | domestic waste | 0.7993 | 0.3996 | 0.5329 | 0.3379 | 0.2249 |
| targeted-v1 | MDWD | test | Cosmos Reason2 8B | garbage | 0.7859 | 0.3716 | 0.5046 | 0.3084 | 0.2071 |
| targeted-v1 | MDWD | test | LocateAnything 3B | black garbage bag | 0.0162 | 0.7186 | 0.0317 | 0.0169 | 0.0134 |
| targeted-v1 | MDWD | test | LocateAnything 3B | grey recycling bag | 0.0055 | 0.8067 | 0.011 | 0.0187 | 0.0167 |
| targeted-v1 | MDWD | test | LocateAnything 3B | white organic-waste bag | 0.0051 | 0.8427 | 0.0101 | 0.0057 | 0.004 |
| targeted-v1 | MDWD | test | LocateAnything 3B | orange garbage bag | 0.005 | 0.7179 | 0.0099 | 0.0058 | 0.0044 |
| targeted-v1 | MDWD | test | LocateAnything 3B | domestic waste | 0.0218 | 0.6763 | 0.0423 | 0.0182 | 0.0122 |
| targeted-v1 | MDWD | test | LocateAnything 3B | garbage | 0.0263 | 0.7541 | 0.0508 | 0.0253 | 0.0182 |
| targeted-v1 | MDWD | test | SAM 3.1 | black garbage bag | 0.73 | 0.9311 | 0.8184 | 0.896 | 0.6929 |
| targeted-v1 | MDWD | test | SAM 3.1 | grey recycling bag | 0.3529 | 0.0504 | 0.0882 | 0.0196 | 0.0159 |
| targeted-v1 | MDWD | test | SAM 3.1 | white organic-waste bag | 0.3599 | 0.7865 | 0.4938 | 0.46 | 0.366 |
| targeted-v1 | MDWD | test | SAM 3.1 | orange garbage bag | 0.8 | 0.8205 | 0.8101 | 0.7933 | 0.6888 |
| targeted-v1 | MDWD | test | SAM 3.1 | domestic waste | 0.797 | 0.1917 | 0.309 | 0.1729 | 0.1234 |
| targeted-v1 | MDWD | test | SAM 3.1 | garbage | 0.426 | 0.9159 | 0.5815 | 0.8465 | 0.6505 |
| targeted-v1 | MDWD | test | SAM 3 | black garbage bag | 0.7494 | 0.9311 | 0.8304 | 0.8967 | 0.6899 |
| targeted-v1 | MDWD | test | SAM 3 | grey recycling bag | 0.3077 | 0.0336 | 0.0606 | 0.0117 | 0.0095 |
| targeted-v1 | MDWD | test | SAM 3 | white organic-waste bag | 0.3948 | 0.7697 | 0.5219 | 0.4675 | 0.3756 |
| targeted-v1 | MDWD | test | SAM 3 | orange garbage bag | 0.8205 | 0.8205 | 0.8205 | 0.7891 | 0.6855 |
| targeted-v1 | MDWD | test | SAM 3 | domestic waste | 0.8177 | 0.142 | 0.2419 | 0.1288 | 0.0937 |
| targeted-v1 | MDWD | test | SAM 3 | garbage | 0.4333 | 0.9105 | 0.5872 | 0.8404 | 0.6457 |
| targeted-v1 | MDWD | test | SAM 3 | black garbage bag | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| targeted-v1 | MDWD | test | LocateAnything 3B | black garbage bag | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | black garbage bag | 0.3251 | 0.2754 | 0.2982 | 0.0984 | 0.0571 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | grey recycling bag | 0.2251 | 0.1807 | 0.2005 | 0.0577 | 0.0351 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | white organic-waste bag | 0.1788 | 0.1798 | 0.1793 | 0.0838 | 0.0418 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | orange garbage bag | 0.0792 | 0.2436 | 0.1195 | 0.0453 | 0.0323 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | domestic waste | 0.8726 | 0.2911 | 0.4366 | 0.259 | 0.1611 |
| targeted-v2 | MDWD | test | Cosmos Reason2 2B | garbage | 0.8698 | 0.302 | 0.4483 | 0.2685 | 0.1686 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | black garbage bag | 0.4435 | 0.6347 | 0.5222 | 0.3277 | 0.2103 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | grey recycling bag | 0.2587 | 0.5294 | 0.3476 | 0.1752 | 0.1127 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | white organic-waste bag | 0.2909 | 0.7191 | 0.4142 | 0.2457 | 0.1559 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | orange garbage bag | 0.1217 | 0.5385 | 0.1986 | 0.1423 | 0.0845 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | domestic waste | 0.816 | 0.3969 | 0.5341 | 0.3358 | 0.2229 |
| targeted-v2 | MDWD | test | Cosmos Reason2 8B | garbage | 0.8123 | 0.3716 | 0.5099 | 0.309 | 0.2074 |
| targeted-v2 | MDWD | test | LocateAnything 3B | black garbage bag | 0.498 | 0.7305 | 0.5922 | 0.4311 | 0.3117 |
| targeted-v2 | MDWD | test | LocateAnything 3B | grey recycling bag | 0.1662 | 0.7983 | 0.2752 | 0.1875 | 0.1489 |
| targeted-v2 | MDWD | test | LocateAnything 3B | white organic-waste bag | 0.1626 | 0.8539 | 0.2731 | 0.1667 | 0.1132 |
| targeted-v2 | MDWD | test | LocateAnything 3B | orange garbage bag | 0.1275 | 0.7308 | 0.2171 | 0.1394 | 0.0987 |
| targeted-v2 | MDWD | test | LocateAnything 3B | domestic waste | 0.443 | 0.6682 | 0.5328 | 0.3106 | 0.2134 |
| targeted-v2 | MDWD | test | LocateAnything 3B | garbage | 0.3737 | 0.7586 | 0.5007 | 0.3201 | 0.2219 |
| targeted-v2 | MDWD | test | SAM 3.1 | black garbage bag | 0.7692 | 0.9281 | 0.8412 | 0.8944 | 0.6848 |
| targeted-v2 | MDWD | test | SAM 3.1 | grey recycling bag | 0.3636 | 0.0504 | 0.0886 | 0.0197 | 0.016 |
| targeted-v2 | MDWD | test | SAM 3.1 | white organic-waste bag | 0.3646 | 0.7865 | 0.4982 | 0.4607 | 0.3644 |
| targeted-v2 | MDWD | test | SAM 3.1 | orange garbage bag | 0.8421 | 0.8205 | 0.8312 | 0.7956 | 0.6906 |
| targeted-v2 | MDWD | test | SAM 3.1 | domestic waste | 0.8108 | 0.1899 | 0.3077 | 0.1728 | 0.1221 |
| targeted-v2 | MDWD | test | SAM 3.1 | garbage | 0.4691 | 0.8978 | 0.6162 | 0.8385 | 0.6413 |
| targeted-v2 | MDWD | test | SAM 3 | black garbage bag | 0.7868 | 0.9281 | 0.8516 | 0.895 | 0.6842 |
| targeted-v2 | MDWD | test | SAM 3 | grey recycling bag | 0.3077 | 0.0336 | 0.0606 | 0.0117 | 0.0095 |
| targeted-v2 | MDWD | test | SAM 3 | white organic-waste bag | 0.4006 | 0.7697 | 0.5269 | 0.4682 | 0.3739 |
| targeted-v2 | MDWD | test | SAM 3 | orange garbage bag | 0.8421 | 0.8205 | 0.8312 | 0.7913 | 0.6874 |
| targeted-v2 | MDWD | test | SAM 3 | domestic waste | 0.8333 | 0.1401 | 0.2399 | 0.129 | 0.0929 |
| targeted-v2 | MDWD | test | SAM 3 | garbage | 0.4757 | 0.8924 | 0.6206 | 0.8324 | 0.6369 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | pedestrian crossing sign | 0.1104 | 0.5105 | 0.1816 | 0.0743 | 0.0559 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | directional sign | 0.0283 | 0.1667 | 0.0483 | 0.0172 | 0.0098 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | back of a traffic sign | 0.2054 | 0.1725 | 0.1875 | 0.0433 | 0.0315 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | traffic sign | 0.6354 | 0.2943 | 0.4023 | 0.1986 | 0.1369 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | roadside traffic-control object | 0.468 | 0.1517 | 0.2291 | 0.0815 | 0.0518 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | stop sign | 0.13 | 0.6692 | 0.2178 | 0.1065 | 0.0893 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | no entry sign | 0.1639 | 0.5166 | 0.2489 | 0.1004 | 0.0742 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | one way sign | 0.1418 | 0.4502 | 0.2157 | 0.0724 | 0.0585 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | roundabout ahead sign | 0.0415 | 0.403 | 0.0753 | 0.0191 | 0.0161 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | no through road sign | 0.058 | 0.6393 | 0.1064 | 0.0439 | 0.0348 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | T-junction sign | 0.0623 | 0.623 | 0.1133 | 0.0482 | 0.0387 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | blind-spot convex mirror | 0.1537 | 0.6562 | 0.2491 | 0.1131 | 0.096 |
| targeted-v2 | MTSD | test | Cosmos Reason2 2B | street sign | 0.0295 | 0.2447 | 0.0526 | 0.0105 | 0.0091 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | pedestrian crossing sign | 0.1511 | 0.6224 | 0.2432 | 0.1156 | 0.082 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | directional sign | 0.0409 | 0.2963 | 0.0719 | 0.0249 | 0.0183 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | back of a traffic sign | 0.3324 | 0.3752 | 0.3525 | 0.1456 | 0.1148 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | traffic sign | 0.7412 | 0.4615 | 0.5688 | 0.3629 | 0.2748 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | roadside traffic-control object | 0.5231 | 0.2063 | 0.2959 | 0.1156 | 0.0808 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | stop sign | 0.168 | 0.8077 | 0.2781 | 0.16 | 0.1327 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | no entry sign | 0.2554 | 0.6682 | 0.3696 | 0.1918 | 0.1542 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | one way sign | 0.191 | 0.5213 | 0.2795 | 0.11 | 0.094 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | roundabout ahead sign | 0.0864 | 0.6716 | 0.1531 | 0.0678 | 0.0555 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | no through road sign | 0.0874 | 0.7377 | 0.1563 | 0.0777 | 0.0743 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | T-junction sign | 0.0733 | 0.7705 | 0.1339 | 0.0663 | 0.0612 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | blind-spot convex mirror | 0.1962 | 0.7188 | 0.3083 | 0.1596 | 0.1353 |
| targeted-v2 | MTSD | test | Cosmos Reason2 8B | street sign | 0.0267 | 0.3085 | 0.0492 | 0.013 | 0.0114 |
| targeted-v2 | MTSD | test | LocateAnything 3B | pedestrian crossing sign | 0.1062 | 0.5245 | 0.1767 | 0.0689 | 0.0637 |
| targeted-v2 | MTSD | test | LocateAnything 3B | directional sign | 0.0294 | 0.3704 | 0.0545 | 0.0216 | 0.0194 |
| targeted-v2 | MTSD | test | LocateAnything 3B | back of a traffic sign | 0.0936 | 0.4508 | 0.155 | 0.0493 | 0.0404 |
| targeted-v2 | MTSD | test | LocateAnything 3B | traffic sign | 0.2262 | 0.4483 | 0.3007 | 0.11 | 0.0876 |
| targeted-v2 | MTSD | test | LocateAnything 3B | roadside traffic-control object | 0.2055 | 0.4373 | 0.2796 | 0.102 | 0.0776 |
| targeted-v2 | MTSD | test | LocateAnything 3B | stop sign | 0.0673 | 0.7923 | 0.1241 | 0.0642 | 0.059 |
| targeted-v2 | MTSD | test | LocateAnything 3B | no entry sign | 0.079 | 0.6919 | 0.1419 | 0.0566 | 0.0499 |
| targeted-v2 | MTSD | test | LocateAnything 3B | one way sign | 0.0648 | 0.2986 | 0.1065 | 0.0206 | 0.0186 |
| targeted-v2 | MTSD | test | LocateAnything 3B | roundabout ahead sign | 0.0179 | 0.6418 | 0.0348 | 0.0135 | 0.0115 |
| targeted-v2 | MTSD | test | LocateAnything 3B | no through road sign | 0.0192 | 0.7705 | 0.0375 | 0.0169 | 0.0167 |
| targeted-v2 | MTSD | test | LocateAnything 3B | T-junction sign | 0.0305 | 0.8361 | 0.0589 | 0.0287 | 0.0275 |
| targeted-v2 | MTSD | test | LocateAnything 3B | blind-spot convex mirror | 0.0765 | 0.6312 | 0.1364 | 0.0538 | 0.0496 |
| targeted-v2 | MTSD | test | LocateAnything 3B | street sign | 0.0062 | 0.2447 | 0.0121 | 0.0018 | 0.0015 |
| targeted-v2 | MTSD | test | SAM 3.1 | pedestrian crossing sign | 0.6457 | 0.5734 | 0.6074 | 0.5442 | 0.507 |
| targeted-v2 | MTSD | test | SAM 3.1 | directional sign | 0.0369 | 0.7222 | 0.0702 | 0.0411 | 0.0341 |
| targeted-v2 | MTSD | test | SAM 3.1 | back of a traffic sign | 0.2686 | 0.5189 | 0.354 | 0.1813 | 0.16 |
| targeted-v2 | MTSD | test | SAM 3.1 | traffic sign | 0.5935 | 0.681 | 0.6342 | 0.5998 | 0.504 |
| targeted-v2 | MTSD | test | SAM 3.1 | roadside traffic-control object | 0.3519 | 0.1466 | 0.207 | 0.0524 | 0.0395 |
| targeted-v2 | MTSD | test | SAM 3.1 | stop sign | 0.2676 | 0.8462 | 0.4067 | 0.6464 | 0.6131 |
| targeted-v2 | MTSD | test | SAM 3.1 | no entry sign | 0.1725 | 0.8199 | 0.285 | 0.518 | 0.4695 |
| targeted-v2 | MTSD | test | SAM 3.1 | one way sign | 0.1776 | 0.4882 | 0.2604 | 0.0904 | 0.0829 |
| targeted-v2 | MTSD | test | SAM 3.1 | roundabout ahead sign | 0.0626 | 0.6866 | 0.1147 | 0.1683 | 0.1642 |
| targeted-v2 | MTSD | test | SAM 3.1 | no through road sign | 0.058 | 0.6557 | 0.1065 | 0.0459 | 0.0449 |
| targeted-v2 | MTSD | test | SAM 3.1 | T-junction sign | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| targeted-v2 | MTSD | test | SAM 3.1 | blind-spot convex mirror | 0.7347 | 0.45 | 0.5581 | 0.425 | 0.3992 |
| targeted-v2 | MTSD | test | SAM 3.1 | street sign | 0.0218 | 0.6383 | 0.0421 | 0.0151 | 0.0131 |
| targeted-v2 | MTSD | test | SAM 3 | pedestrian crossing sign | 0.6614 | 0.5874 | 0.6222 | 0.5549 | 0.5161 |
| targeted-v2 | MTSD | test | SAM 3 | directional sign | 0.0381 | 0.7037 | 0.0723 | 0.0396 | 0.0326 |
| targeted-v2 | MTSD | test | SAM 3 | back of a traffic sign | 0.2717 | 0.4871 | 0.3489 | 0.1733 | 0.1546 |
| targeted-v2 | MTSD | test | SAM 3 | traffic sign | 0.6085 | 0.6711 | 0.6382 | 0.5896 | 0.4973 |
| targeted-v2 | MTSD | test | SAM 3 | roadside traffic-control object | 0.3671 | 0.1502 | 0.2131 | 0.0559 | 0.0428 |
| targeted-v2 | MTSD | test | SAM 3 | stop sign | 0.2678 | 0.8385 | 0.406 | 0.6541 | 0.6202 |
| targeted-v2 | MTSD | test | SAM 3 | no entry sign | 0.1776 | 0.8199 | 0.292 | 0.5076 | 0.4603 |
| targeted-v2 | MTSD | test | SAM 3 | one way sign | 0.1813 | 0.4976 | 0.2658 | 0.0965 | 0.0884 |
| targeted-v2 | MTSD | test | SAM 3 | roundabout ahead sign | 0.0687 | 0.6866 | 0.1248 | 0.1665 | 0.1629 |
| targeted-v2 | MTSD | test | SAM 3 | no through road sign | 0.0593 | 0.7049 | 0.1094 | 0.0596 | 0.0584 |
| targeted-v2 | MTSD | test | SAM 3 | T-junction sign | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| targeted-v2 | MTSD | test | SAM 3 | blind-spot convex mirror | 0.75 | 0.4313 | 0.5476 | 0.411 | 0.3867 |
| targeted-v2 | MTSD | test | SAM 3 | street sign | 0.0221 | 0.6277 | 0.0427 | 0.0152 | 0.0131 |

> Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity is a prompt-sensitivity run; see prompt_sensitivity_results.csv

> Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity is a prompt-sensitivity run; see prompt_sensitivity_results.csv

*Full table incl. source paths: `promptdetect_results.csv`*

## Model efficiency (inference speed)

**PENDING** - Model efficiency: no inference-speed benchmark runs yet - pending (run Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py)

## Prompt sensitivity and prediction consistency

Controlled paraphrase families only (identical target GT per family); synonym and broad-prompt effects are reported in the PromptDetect section, never here. Prediction consistency measures agreement between prompt variants, not accuracy.

| dataset | model | n_families | macro_mean_f1 | mean_within_family_f1_std | mean_f1_range | mean_relative_f1_degradation | worst_family | macro_canonical_f1 | macro_instruction_f1 | protocol_version |
|---|---|---|---|---|---|---|---|---|---|---|
| MDWD | Cosmos Reason2 2B | 4 | 0.2111 | 0.0735 | 0.2037 | 0.5889 | mdwd-recyclable-material | 0.1994 | 0.3258 | prompt-sensitivity-v2 |
| MDWD | Cosmos Reason2 8B | 4 | 0.3633 | 0.0274 | 0.0713 | 0.1893 | mdwd-orange-cmd | 0.3707 | 0.3576 | prompt-sensitivity-v2 |
| MDWD | LocateAnything 3B | 4 | 0.303 | 0.051 | 0.1245 | 0.3432 | mdwd-orange-cmd | 0.3394 | 0.2469 | prompt-sensitivity-v2 |
| MDWD | SAM 3 | 4 | 0.5211 | 0.1235 | 0.2927 | 0.4954 | mdwd-recyclable-material | 0.5676 | 0.3652 | prompt-sensitivity-v2 |
| MDWD | SAM 3.1 | 4 | 0.5262 | 0.1204 | 0.2942 | 0.4903 | mdwd-recyclable-material | 0.5648 | 0.3811 | prompt-sensitivity-v2 |
| MTSD | Cosmos Reason2 2B | 5 | 0.1926 | 0.0296 | 0.0742 | 0.3472 | mtsd-roundabout-ahead | 0.1945 | 0.216 | prompt-sensitivity-v2 |
| MTSD | Cosmos Reason2 8B | 5 | 0.3054 | 0.0367 | 0.098 | 0.3059 | mtsd-roundabout-ahead | 0.2705 | 0.3049 | prompt-sensitivity-v2 |
| MTSD | LocateAnything 3B | 5 | 0.0865 | 0.0244 | 0.0628 | 0.4713 | mtsd-pedestrian-crossing | 0.1228 | 0.0828 | prompt-sensitivity-v2 |
| MTSD | SAM 3 | 5 | 0.2418 | 0.1045 | 0.2736 | 0.6161 | mtsd-blind-spot-mirror | 0.3985 | 0.2359 | prompt-sensitivity-v2 |
| MTSD | SAM 3.1 | 5 | 0.2367 | 0.1049 | 0.2732 | 0.6144 | mtsd-blind-spot-mirror | 0.3944 | 0.2274 | prompt-sensitivity-v2 |

*Full tables: `prompt_sensitivity_results.csv` (per family), `prompt_sensitivity_model_summary.csv`, `prompt_consistency_results.csv` (per prompt pair); source run paths and protocol hashes are recorded per row.*

> Prompt-sensitivity tables aggregated from 2 run(s); prediction consistency measures agreement between prompt variants, not accuracy against GT.

---
*Source files are recorded per row for traceability. Pending sections list the command that produces them.*
