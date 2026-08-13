# HYPER-ID — Project Context (agentic-IDE handoff)

## 1. What this is
Knowledge Engineering course lab project, reimagining StrayCare's (an animal-welfare platform) planned breed-classifier module. Instead of forcing one breed label onto a photo, the system estimates breed **ancestry** via traits, explainably — because most target animals (strays) are mixed/indigenous, not purebred, and forced single-label output is confidently wrong for them. Core novelty: an OWL ontology of breed traits (including a Local/Indigenous phenotype class no existing dataset has) grounds a CLIP-based vision pipeline, instead of chasing accuracy on the already-solved plain-breed-classification benchmark task.

## 2. Architecture (3 layers, pipeline order)
```
Photo -> CLIP (frozen) -> embedding -> trained trait model -> per-trait predictions
      -> SPARQL query against ontology -> ranked, explainable breed-ancestry output
```
- **Knowledge layer**: `breed_trait_ontology.ttl` (OWL/Turtle, opens in Protégé).
- **Perception layer**: CLIP (`openai/clip-vit-base-patch32`, via HuggingFace `transformers`), permanently frozen.
- **Reasoning layer**: trait predictions -> SPARQL trait-overlap query -> ranked candidate breeds + reasoning trace.

## 3. Key decisions already made — DO NOT relitigate these without new information
- **CLIP is never fine-tuned.** Frozen throughout. Reason: fine-tuning on just 13 breeds would destroy zero-shot generalization to phenotypes with little/no training data — specifically the Local/Indigenous node, which is the entire point of the project.
- **"Track B" (lighter) chosen over full trained neuro-symbolic pipeline.** CLIP zero-shot + ontology grounding, not a from-scratch trained vision backbone + GNN. Reason: feasible in the timeframe; full pipeline is explicitly scoped as future work.
- **Joint multi-task neural network chosen over 6 independent per-trait classifiers.** One shared trunk + 6 output heads (EarShape, CoatType, CoatPattern, SizeClass, SnoutLength, TailCarriage), trained together via backprop. Independent logistic-regression-per-trait was tried first and explicitly rejected as low-impact/gimmicky. The multi-task-vs-independent comparison is itself now a reported ablation.
- **Weak supervision**, not manual labeling: breed (known from dataset) -> ontology lookup -> trait labels, computed once before training. This is preprocessing, not part of the training loop.
- **Live integration into StrayCare and continuous retraining on site data are explicit future work**, not part of this deliverable. Do not attempt to build these now.
- **Scope: dogs-first.** Cats are designed into the ontology schema conceptually but not populated; explicit near-term descope, cats come after the dog pipeline is proven.
- **Baseline for all comparisons**: plain CLIP zero-shot against bare breed names (no ontology grounding).

## 4. Technical specifications
- **Pretrained model**: `openai/clip-vit-base-patch32` (CLIP ViT-B/32), HuggingFace `transformers`, frozen. Drop-in upgrade path if needed later: `clip-vit-large-patch14`.
- **13 dog breeds** (Oxford-IIIT Pet subset) + 1 Local/Indigenous phenotype = 14 breed individuals in the ontology: Beagle, Boxer, Chihuahua, Pug, Samoyed, ShibaInu, GreatPyrenees, GermanShorthairedPointer, StaffordshireBullTerrier, YorkshireTerrier, Pomeranian, BassetHound, SaintBernard, LocalIndigenousDog.
- **Dataset name -> ontology name mapping** (torchvision `.classes` returns Title Case with spaces — must `.lower().replace(" ", "_")` before lookup): beagle, boxer, chihuahua, pug, samoyed, shiba_inu, great_pyrenees, german_shorthaired, staffordshire_bull_terrier, yorkshire_terrier, pomeranian, basset_hound, saint_bernard.
- **6 trait dimensions**: EarShape (FloppyEars/ErectEars/SemiErectEars/ButtonEars), CoatType (ShortCoat/MediumCoat/LongCoat/WireCoat), CoatPattern (SolidColor/Brindle/Spotted/Tricolor/PatchedColor), SizeClass (ToySize/SmallSize/MediumSize/LargeSize/GiantSize), SnoutLength (ShortSnout/MediumSnout/LongSnout), TailCarriage (CurledTail/StraightTail/PlumeTail/DockedTail).
- **Dataset**: Oxford-IIIT Pet via `torchvision.datasets.OxfordIIITPet` (auto-downloads). Plus a small (15-20 photo) locally-collected stray eval set — not yet collected.
- **Ontology namespace**: `http://straycare.org/ontology/breed#`. Properties are `owl:FunctionalProperty` (one value per breed per dimension) and trait subclasses are declared `owl:AllDisjointClasses` — the reasoner (HermiT/Pellet in Protégé) will catch contradictions.
- **Hardware**: RTX 3060, CUDA — currently not being detected (running on CPU; see debug note above, likely a CPU-only torch wheel on Python 3.14).

## 5. File inventory (all previously delivered to the user)
| File | Status | Purpose |
|---|---|---|
| `breed_trait_ontology.ttl` | FINAL (may still get small trait-value corrections once local photos are reviewed) | The ontology — open in Protégé |
| `sparql_example_queries.md` | FINAL | 4 example SPARQL queries, incl. the trait-overlap ranking template the reasoning layer should use |
| `train_trait_model.py` | FINAL / canonical training script (just bug-fixed — breed-name normalization) | Joint multi-task NN training, GPU-accelerated, includes built-in ablation vs. independent classifiers, saves `trained_model/multitask_trait_model.pt`, `label_encoders.pkl`, `loss_curve.png` |
| `train_trait_probes.py` | SUPERSEDED — do not use | Old independent-logistic-regression version, kept only for history |
| `demo_toy_training_walkthrough.py` | LEARNING TOOL ONLY, not part of pipeline | Synthetic-data walkthrough of the training mechanism |
| `project-proposal-breed-ancestry-ke.md` | Reference doc | Full motivation/architecture/evaluation-plan/timeline write-up |
| `beginners_guide_full_pipeline.md` | Reference doc | Plain-language onboarding guide for the (non-technical) team lead |

## 6. Evaluation plan (3 experiments — none run yet)
1. **Sanity accuracy** — top-k accuracy on clean Oxford-IIIT test set, grounded system vs. baseline. Expected roughly comparable; not the headline result.
2. **Confidence/honesty comparison (headline result)** — % of local mixed/indigenous-set photos where each system gives an overconfident single-breed answer. Baseline expected high (overconfident); grounded system expected lower/calibrated.
3. **Efficiency** — (a) training wall-clock time vs. a typical full CNN fine-tune; (b) marginal cost to add one new breed (one ontology entry vs. full retrain for a traditional classifier).

## 7. What's left to build, precisely (not just ontology + dataset)
1. Fix/verify the GPU device-detection issue (optional, not blocking).
2. Build the **baseline script**: plain CLIP zero-shot, bare breed-name prompts, no ontology grounding.
3. Build the **reasoning/aggregation layer**: load `trained_model/multitask_trait_model.pt`, run inference on a new photo, get trait predictions, execute the SPARQL trait-overlap query (template in `sparql_example_queries.md`, query #3) against `breed_trait_ontology.ttl`, output a ranked, explainable result.
4. Collect + manually trait-label 15-20 real local stray photos.
5. Run all 3 evaluation experiments once #2-4 exist; compile results tables.
6. Build a minimal demo interface (CLI or small API endpoint) wrapping the reasoning layer.
7. Draft the paper (structure already defined in `project-proposal-breed-ancestry-ke.md`).

## 8. Context / constraints
- Supervisor gave 1 week (from initial pitch) for a working prototype; overall goal is full proof-of-concept results + comparisons within 2-3 weeks, aiming toward a conference-paper-length writeup and eventual (future-work) StrayCare integration + retraining loop.
- Team of 5, juggling 4 other concurrent projects — team lead is non-technical/new to ontology tooling, needs plain-language explanations; prefers a skills-based task framework over Claude assigning named roles.
- Project name: **HYPER-ID**.
