# PromptDetect Dissertation Protocol

`run_dissertation_protocol.py` is the fixed, target-aware dissertation evaluator. The existing `run_batch_eval.py` remains available and is explicitly labelled `class-agnostic-exploratory`.

Targeted-v1 evaluates every image for each prompt. Only boxes in that prompt's declared target classes are positive GT; detections on target-negative images are false positives. False positives overlapping non-target annotated objects are separately recorded. This corrects the exploratory evaluator's tendency to reward a prompt for matching any annotated object.

The versioned YAML contains six MDWD prompts and thirteen primary MTSD prompts, including the all-12-class “roadside traffic-control object.” “No entry”/“one way” and “no through road”/“T-junction” remain separate synonym comparisons. The broad MTSD “traffic sign” target deliberately excludes the convex mirror. The primary model matrix is SAM 3, SAM 3.1, LocateAnything 3B, Cosmos Reason2 2B, and Cosmos Reason2 8B; 32B remains behind `--allow-heavy`.

```powershell
cd Scripts/Other-Scripts/PromptDetect/batch_evaluation
python run_dissertation_protocol.py --dataset both --split test --dry-run
python run_dissertation_protocol.py --dataset both --split test --smoke-test --models sam3
python run_dissertation_protocol.py --dataset MDWD --split test --final --wandb-mode online
python run_dissertation_protocol.py --dataset MTSD --split test --final --wandb-mode online
python run_dissertation_protocol.py --dataset MTSD --resume <run-dir> --skip-completed
python run_dissertation_protocol.py --reports-only <run-dir>
```

With `--dataset both` and no explicit `--protocol`, the runner executes two
separate stages: this optimized classic protocol (`dissertation-v2`, label `targeted-v2`)
followed by the controlled prompt-sensitivity protocol
(`prompt-sensitivity-v2`), each in its own run directory with its own outputs
and reports. Pass `--protocol` to run a single protocol; see
[PromptDetect-Prompt-Sensitivity-Protocol.md](PromptDetect-Prompt-Sensitivity-Protocol.md).

Use the `mtsd-base` environment; LocateAnything delegates to `mtsd-la`. SAM checkpoints require the documented Hugging Face access. Final MTSD mode requires the canonical prepared unaugmented test split, a valid prep/split manifest, the explicit QA-resolved GRP-1…GRP-11 scope, QA-only annotations and a resolved non-overridden QA gate. It rejects QA fallback, mixed/raw-XML datasets, auto scope and stale hashes. Each model×prompt combination is persisted atomically with hashes and thresholds so an interrupted run loses no completed work.

The final protocol derives its per-image detection cap from the busiest image
in each complete canonical test split: **28 boxes for MDWD** and **18 boxes for
MTSD**. It applies class-agnostic NMS at IoU 0.50 before the cap. Both values,
their source image, the NMS threshold and the protocol hash are stored in the
run configuration and combination fingerprints; results produced under the
old 100-box policy cannot be mixed into or resumed as this protocol.

Inference is process-isolated. SAM gets one fresh worker process per
model×prompt combination. LocateAnything and Cosmos additionally split each
combination into 64-image workers. Every worker loads one model, persists its
chunk and exits, making process termination—not Python garbage collection—the
RAM/VRAM cleanup boundary. Completed chunks and combinations remain resumable.

Pass `--save-visualizations 10` to retain ten deterministic, outcome-stratified
examples per model/prompt combination. Each compact sheet uses the exact target
taxonomy and IoU matching used for scoring: matched target GT is blue, FN is
orange, non-target GT is grey, TP predictions are green, and FP predictions are
red. Open `visualizations/index.html` in each result directory to browse them.

Headline comparison is the macro average over both `class-targeted` and
`synonym-comparison` prompts; synonym phrasings remain independent rows. Broad
and optional-broad prompts are separate and never enter the headline. Universal
ranking uses precision, recall, F1, matched IoU and runtime. Cosmos and
LocateAnything use constant confidence 1.0, so `ap_meaningful=false`; their AP
is never used to choose a winner. SAM AP may be reported, but it is not the sole
cross-model ranking metric.
