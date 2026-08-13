# Project Proposal: Ontology-Grounded Explainable Breed-Ancestry Estimation for Mixed & Indigenous Companion Animals

**Course:** Knowledge Engineering — Lab Final Project
**Context:** Reimagining the breed-recognition module of StrayCare (animal welfare platform)

---

## 1. Executive Summary

StrayCare's original plan included an instant breed-classification feature — a standard image-in, breed-label-out CNN classifier. This is a well-trodden benchmark task with diminishing research value. We are replacing it with a **knowledge-graph-grounded, explainable breed-ancestry estimation system** that reasons over an ontology of breed traits instead of forcing a single fixed-class label. This directly targets a real, largely unsolved problem — most animals StrayCare serves are strays, and strays are overwhelmingly mixed-breed or locally indigenous, not purebred. It also makes the ontology (not the model) the centerpiece deliverable, aligning the project directly with Knowledge Engineering course outcomes.

---

## 2. Problem Statement

Existing breed classifiers (the "cliché" version of this task) share one hidden assumption: the input animal is purebred and belongs to one of ~100–120 fixed classes drawn from datasets like Stanford Dogs. Under that assumption, a classifier will confidently output a single breed name for any input.

This assumption breaks for stray animals:

- Strays are predominantly mixed-breed or regional/indigenous phenotypes (e.g., South Asian street dogs, local street cats), which no major benchmark dataset represents.
- Forcing a single-breed label on a mixed animal doesn't just lower accuracy — it produces a **confidently wrong** answer, which is worse than an honestly uncertain one for a welfare platform making care/adoption recommendations.
- A literature scan confirms this is a recognized, largely open gap: dedicated mixed-breed identification work is scarce, and shelter studies comparing staff visual breed ID against DNA testing show frequent mismatches — visual single-breed judgment is unreliable even for trained humans.

**Core question:** Can we replace forced single-label classification with an explainable, trait-grounded ancestry estimate that degrades honestly on mixed/indigenous inputs instead of failing silently?

---

## 3. Research Gap & Positioning

| Existing approach | Limitation | Our approach |
|---|---|---|
| CNN/ensemble classifiers on Stanford Dogs | Chases marginal accuracy gains on a solved benchmark; assumes purebred input | Reframes the task around mixed/indigenous ancestry, not forced single-label output |
| Fixed breed taxonomies | No representation for local/indigenous phenotypes | Ontology includes explicit Local/Indigenous phenotype nodes for both dogs and cats |
| Black-box CNN outputs | No explanation for *why* a breed was predicted | Trait-level reasoning trace justifies every prediction |

This also places the work inside an active, recognized research direction: using OWL-based knowledge graphs as a symbolic layer that grounds and constrains neural predictions (neuro-symbolic AI) — an approach with a standing research community (e.g., AAAI's "Combining Machine Learning and Knowledge Engineering" workshop lineage), not a fringe idea.

---

## 4. Objectives

1. Build a breed-and-trait ontology in OWL, including an indigenous/local phenotype class absent from existing datasets. **Dogs first** for the working prototype; cats are a designed but deferred extension (same schema, added once the pipeline is proven).
2. Ground a vision-language model's predictions in that ontology instead of raw breed-name prompts.
3. Produce ranked, explainable ancestry estimates with a human-readable reasoning trace.
4. Empirically show the grounded approach degrades more gracefully than plain zero-shot classification on mixed/indigenous inputs.
5. Package the result as StrayCare's breed-recognition module and as a short conference-style paper.

---

## 5. System Architecture

```
                [Input Photo]
                      |
                      v
          [CLIP Vision Encoder] ---------- image embedding
                      |
                      v
          [Similarity Scoring]  <----  [Ontology-Generated Trait Prompts]
                      |                          ^
                      |                          |
                      |               [Breed & Trait Ontology — OWL/Protégé]
                      v
          [Trait-Level Match Scores]
                      |
                      v
     [Reasoning & Aggregation Layer]   (weighted trait matching + confidence calibration)
                      |
                      v
          [Explainable Ancestry Output]
   e.g. "62% Labrador-associated traits (ears, coat)
         28% Local/Indigenous phenotype (build, snout)
         10% uncertain"
         + reasoning trace listing matched traits
```

**Three layers, mapped to course concepts:**

- **Knowledge layer** — the ontology itself: classes, subclasses, object/data properties, a reasoner (Pellet/HermiT) for consistency checking, SPARQL queries for trait lookup. This is the assessable Knowledge Engineering artifact.
- **Perception layer** — CLIP (pretrained, no training required for the base version) scores images against trait descriptions pulled directly from the ontology.
- **Reasoning layer** — aggregates trait-level matches into a ranked, explainable ancestry estimate. This is symbolic reasoning over a knowledge base, not a black box.

---

## 6. Novelty / Contributions

1. **Reframing the task** — ancestry estimation instead of forced single-label classification, matching the actual population StrayCare serves.
2. **Explicit indigenous/local phenotype representation** — a class no existing breed dataset or ontology includes, directly relevant to a Bangladesh-based platform.
3. **Ontology-grounded prompting** — using structured trait knowledge to ground a vision-language model, rather than bare breed names; this is directly ablatable (grounded vs. ungrounded CLIP) as a clean experimental result.
4. **Explainability by construction** — every output carries a trait-level reasoning trace, not just a confidence score.
5. **A locally-collected evaluation set** — stray photos gathered through StrayCare itself, annotated for traits — a small but genuine dataset contribution, since no such set exists in the literature.

---

## 7. Dataset & Evaluation Plan

- **Base data:** Oxford-IIIT Pet dataset, dog subset (13 breeds selected for this phase), breed-labeled — used for sanity-check accuracy and prompt calibration.
- **Local evaluation set:** a small set of stray dog photos collected via StrayCare, manually trait-annotated by the team — used to test behavior on the actual target population (mixed/indigenous animals).
- **Baseline for comparison:** plain CLIP zero-shot against bare breed names (no ontology grounding) — a near-free byproduct of the grounded system, since it's the same pipeline with the grounding step removed.

**Three experiments — not one accuracy number:**

1. **Sanity-check accuracy** — top-k accuracy on the clean Oxford-IIIT test set, grounded system vs. baseline. Expected to be roughly comparable; this proves the system works, it is not the headline result.
2. **Confidence/honesty comparison (the headline result)** — run both systems on the local mixed/indigenous set and measure the rate of overconfident single-breed predictions (e.g., % of images where top-1 confidence exceeds a fixed threshold). The baseline is expected to be confidently wrong far more often; the grounded system should show calibrated, split confidence toward the Local/Indigenous node on genuinely mixed animals.
3. **Efficiency comparison** — (a) training wall-clock time: linear probe on frozen CLIP embeddings (minutes) vs. a typical full CNN fine-tune (hours); (b) marginal cost to add one new breed class: one new ontology entry for our system vs. new labeled images + full retraining for a traditional classifier.

Plus a small curated set (3-5 cases) of qualitative side-by-side outputs — baseline vs. grounded system on the same photo — for the presentation and paper.

---

## 8. Implementation Plan (4 weeks)

| Week | Deliverable |
|---|---|
| **1** (prototype sprint, supervisor checkpoint) | Ontology built in Protégé (13 dog breeds + trait vocabulary + Local/Indigenous node), with example SPARQL queries. CLIP embedding + linear-probe training pipeline. Reasoning/aggregation layer. Minimal end-to-end demo. |
| **2** | Baseline (plain CLIP, no grounding) built. Local eval set finalized. Full pipeline hardened. |
| **3** | Run all three experiments (accuracy, confidence/honesty, efficiency). Start drafting Method + Experiments sections while results are fresh. |
| **4** | Finalize comparison tables + qualitative examples. Polish demo (working endpoint, not live production). Finish paper draft (Intro/Related Work/Method/Experiments/Discussion/Future Work). Build slides, rehearse. |

---

## 9. Expected Outcomes

- A working, explainable breed-ancestry module, demoed end-to-end (functioning API/script), showing measurably more honest behavior than a baseline classifier on mixed/indigenous animals.
- A quantified efficiency advantage over traditional classification (training time and marginal cost of adding a new breed).
- An OWL ontology (dog breed/trait taxonomy with indigenous phenotype representation) as a standalone, reusable Knowledge Engineering artifact.
- A conference-paper-length write-up documenting the comparison results and the indigenous-phenotype gap, suitable for a workshop-tier venue (e.g., ML+KE intersection workshops) or as a strong course-final research report.
- A small, original, trait-annotated local stray dataset as a secondary contribution.

---

## 10. Limitations & Future Work

- The base system uses pretrained CLIP with a lightweight linear probe rather than a fully fine-tuned trait detector; a fully trained neuro-symbolic pipeline (trait detector + graph neural network reasoning over the ontology) is a natural next step beyond this course.
- The local evaluation set will be small — sufficient for qualitative/directional evidence, not a large-scale benchmark claim.
- Ontology coverage is necessarily incomplete (breed standards number in the hundreds) and is dog-only for this phase; cats are designed but not yet populated.
- **Live integration into the running StrayCare platform and a continuous retraining loop on real site data are planned future work**, not part of this phase's deliverable — the near-term goal is a working, evaluated prototype and demo endpoint, not a production deployment.
