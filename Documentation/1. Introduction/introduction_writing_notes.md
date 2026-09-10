# Introduction — writing notes

Companion to `introduction.tex` and `introduction_scope_audit.md`. Commit `cee7dc6a`, 11 July 2026.

## Revision 2026-07-18 (commit 0c344db)

- **MTSD facts updated to the completed scope:** all eleven groups annotated + QA-resolved (gate resolved 18 July 2026); chapter now states **7,482 images / 20,509 instances** (the old "7,492 raw / six groups / 3,808 / 10,263" wording removed; GRP-10 is 779 images after the QA resolution pass).
- **Objective 2 widened to MDWD + MTSD** (prepared datasets `mtsd-qa-v1-aug/-noaug` built 18 July; supervised pipeline final; training imminent) and now names the augmentation ablation. Contribution 2 updated to match. This supersedes the earlier "MTSD supervised excluded from objectives" decision.
- **Objective 4 corrected** to four backbone families (adds LingBot-Vision) at two sizes; the assessment paragraph now says "sixteen-variant matrix" (was "six-variant" — a leftover from the pre-revision draft).
- **Tightening pass:** Problem Definition ¶1 and ¶4 compressed (the ¶4 dimension list overlapped the Proposed Solution evaluation paragraph); future-work list no longer promises "additional annotated MTSD collection groups" (annotation is complete).
- Mandated structure, objective enumeration and the verbatim lead-in sentence are unchanged.

## Unresolved repository conflicts

1. **MDWD source/instance counts vs the EUVIP paper.** Chapter uses the repository-verified 3,598 unique sources and omits a manual-instance count because no repository artefact substantiates the paper's 11,461 figure (and export-level box counts are augmentation-inflated). If the EUVIP paper's 3,697/11,461 refer to a pre-v20 annotation set, decide whether the dissertation reports the collection-level figures (paper) or the working-release figures (repo v20), and state which release the number describes. **Supervisor/author decision required.**
2. **Stale Summary.md/README MTSD counts** (3,098/8,372, five groups) vs primary artefacts (3,808/10,263, six groups incl. GRP-7). Update those documents, or the dissertation and repo docs will disagree.
3. **QA gate not refreshed for GRP-7** (`update_qa_gate.py --apply` pending) and the recorded audit predates GRP-6/7. Irrelevant to the Introduction's wording but blocks `--final` MTSD preparation.
4. **MDWD integrity report shows FAIL (15 out_of_range_box)** while Summary.md's 2026-07-11 note reclassifies them as polygon-format lines misread by the checker. Methodology must reconcile; Introduction stays silent on findings.

## Pending experiments affecting objective defensibility

The Introduction's objectives must all be evaluated by submission. At this commit:

- **Objective 3 (PromptDetect)**: protocol implemented and dry-run verified; only a 5-image pilot executed. The full fixed-protocol MDWD and MTSD evaluations are **pending**. If they are ultimately not run, Objective 3 must be removed/reworded and the prompt-based track moved to future work.
- **Objective 4 deployment measurements**: inference-speed benchmark implemented but **not executed**; annotation-hour values "not recorded" in the effort report. Latency/throughput wording in the objective assumes the benchmark will run.
- **Objective 4 attribute study**: executed as the GRP-1–3 historical snapshot; a final-scope rerun is deferred until the annotation scope freezes. The chapter does not specify the data snapshot; Methodology must.
- **MTSD supervised detection** is deliberately absent from the objectives (not executed). If it is completed later, Objective 2 could be widened to both datasets — currently it is MDWD-only by design.

**Consequence:** the Introduction is a defensible draft but cannot be declared final until the PromptDetect evaluations and the inference benchmark have been executed (or the affected wording is trimmed).

## Unavailable reference files

- `MDWD_EUVIP26.pdf` — not in repo or workspace parent; style guidance followed from the task brief's description; statistics re-verified independently (NSO + Eurostat).
- Latest MSc progress report and `MScAISupervisionDeclarationForm(Full-time).pdf` — not found (only BSc FYP documents exist under `e:\2. UM-Student\BSC FYP\`).
- No existing dissertation abstract, introduction draft, or main `.tex` sources exist in the repository (only the Background/Literature Review drafts from 11 July 2026).

## Taxonomy notes

- `Damaged-Unknown` excluded from the shape taxonomy (config drop value; QA decision 2026-07-03). Not mentioned in the chapter.
- Attribute vocabularies written exactly as in `config/default.yaml`; class names for MDWD exactly as in the integrity report.
- MTSD class list summarised functionally ("safety-relevant, regulatory, navigation-related and municipal signage… rear-facing and otherwise unidentifiable") rather than enumerating all 12 names — full taxonomy is in Background §4 to avoid duplication.

## Citations

- 24 unique keys used; all resolve in `Documentation/Background-LiteratureReview/background_literature_references.bib`, to which `nso2026municipalwaste` and `eurostat2026municipal` were added (both verified 11 July 2026).
- Statistic verification: NSO Malta "Municipal Waste: 2024" (621 kg resident basis, 574 kg incl. tourists) and Eurostat news release 30 Mar 2026 (EU average 517 kg). These remain the newest comparable official figures as of the search date; the EUVIP paper's values are therefore retained.
- No results-bearing citations; model papers cited only to identify the systems.
- Corporate-author entries (Meta AI, NVIDIA) still need full author lists (carried over from the Background/LR audit).

## LaTeX / template notes

- Exact required section structure used; one `enumerate` for the four objectives (supervisor guidance requests explicit objectives; remaining prose is connected paragraphs).
- The mandated sentence "This will be achieved through the following objectives:" appears verbatim.
- Labels `ch:methodology`, `ch:evaluation`, `ch:conclusion` are **assumed** (chapters not yet written); confirm when created. `ch:background` and `ch:literature-review` match the existing drafts.
- No dissertation template exists in the repository, so the chapter was test-compiled in a minimal standalone wrapper (see completion report for outcome); re-check page count (target 3–4 pp) in the real UM template.
- The dissertation title differs between the task brief and `research_questions.yaml`; the chapter never states the title, so no change is needed here — but front matter must standardise it.

## Supervisor decisions that may still be required

1. Which MDWD count set is canonical for the dissertation (paper collection-level vs repo v20 release-level).
2. Whether GRP-7 (and any later groups) enter the final MTSD experimental scope, which fixes the attribute-study final rerun and the counts quoted across chapters.
3. Whether the mandated result-free Introduction is acceptable against the generic supervisor guidance requesting summarised significant results (the explicit dissertation-specific restriction was followed; Chapter 5/6 will carry the results and the critical statement of success).
4. Final dissertation title.
