# The Complete Beginner's Guide to Our Project — Read This When Anything Feels Overwhelming

This document assumes you know nothing going in. Every term is explained the first time it shows up. Read it in order, once, slowly — it's designed to be your map, not something to memorize.

---

## Chapter 0: What Are We Actually Building? (No jargon, one paragraph)

We're building a program that looks at a photo of a dog and, instead of guessing one fixed breed name (which is wrong and misleading for the mixed-breed strays we actually care about), gives an honest answer like: *"this dog shows traits associated with Labradors, and also traits associated with local street dogs — here's why."* To do that, we need three things working together: (1) a structured knowledge base of what each breed looks like, (2) an AI model that can look at a photo and recognize physical traits, and (3) a piece of logic that connects the two. That's the whole project. Everything below is just the "how."

---

## Chapter 1: Meet the 5 Artifacts (the map, before you enter the territory)

Before diving into any one piece, here's everything we're building or using, in one list, so nothing feels like a surprise later:

1. **The ontology** (`breed_trait_ontology.ttl`) — our structured knowledge base of breeds and traits. Built by hand (already drafted for you).
2. **The photo dataset** — a public, free collection of labeled dog photos we download, plus a small set of real stray photos your team collects.
3. **CLIP** — an already-built, freely available AI model that turns photos into a numeric summary. We use it as-is; we don't build it.
4. **Our trained "trait model"** — a small program we train ourselves, whose only job is reading CLIP's numeric summary and guessing which traits (ear shape, coat, etc.) are present.
5. **The reasoning script** — the glue code that takes trait guesses, asks the ontology "which breeds match this?", and prints a human-readable, explained answer.

Keep this list nearby. Every section below is really just zooming into one of these five things.

---

## Chapter 2: The Ontology — What It Is, and How to Use Protégé

**What is an ontology, really?** It's a structured way of storing facts and their relationships so a computer program can query them precisely — think of it as a very disciplined, cross-referenced encyclopedia, rather than a paragraph of prose a computer can't reliably parse.

**Three words you'll see constantly:**
- **Class** = a category. Example: `Breed`, `Trait`, `EarShape`.
- **Individual** = one specific example belonging to a class. Example: `Beagle` is an individual of class `Breed`. `FloppyEars` is an individual of class `EarShape`.
- **Property** = a labeled connection between two individuals. Example: `Beagle —hasEarShape→ FloppyEars` is one fact, stored as an arrow.

**What is OWL? What is a `.ttl` file?** OWL (Web Ontology Language) is just the *standard* that ontologies are written in, so any ontology tool can read any other tool's file — the same reason everyone agrees on `.docx` for Word documents. `.ttl` (Turtle) is one specific, human-readable text format for writing OWL. You don't need to write raw `.ttl` by hand going forward — Protégé gives you a visual interface instead.

**What is Protégé?** A free desktop application for viewing and editing ontology files — it's to an ontology what Microsoft Word is to a `.docx` file. You already have it open.

**Your Protégé tour** (do this once, slowly):

| Tab name | What you'll see | What it means |
|---|---|---|
| **Classes** | A tree: `Trait` branching into `EarShape`, `CoatType`, etc.; `Breed`, `BreedGroup` as separate branches | The category system — your "table of contents" |
| **Object Properties** | A list: `hasEarShape`, `belongsToGroup`, etc. | The types of arrows/connections allowed |
| **Individuals by class** | Pick a class on the left (e.g. `Breed`), see every specific example on the right | The actual data — click any breed to see its facts |
| **SPARQL Query** | A text box | Where you type questions (see Chapter 5) |

**What is "the reasoner," and why does it matter?** It's a checking program built into Protégé (two common ones are named HermiT and Pellet — just names, nothing to memorize) that re-reads everything in the ontology and flags contradictions — for example, if a breed accidentally got assigned two different sizes, when we've told it every breed has exactly one. Run it via the menu: **Reasoner → Start Reasoner**. If it finishes with no red errors, your ontology is internally consistent. This is a genuinely good thing to demo live to your supervisor — it's proof the system does real logical checking, not just data storage.

**What would you actually do in Protégé, hands-on, this week?** Mainly: open an individual (like `Beagle`), check whether its trait values look right, and correct them if a teammate finds a mistake — same as editing a row in a spreadsheet, just via clicking instead of typing into cells.

---

## Chapter 3: The Photo Dataset

**Oxford-IIIT Pet dataset** is a free, publicly available collection of ~7,000 labeled cat/dog photos, built by a university (Oxford) specifically for AI research — meaning it's already organized, already labeled, and legal/free to use. We only use the 13 dog breeds that match our ontology.

**"Downloading" it** means running one line of code that fetches it automatically from the internet the first time you run our script — no manual clicking through a website needed.

**The local stray photo set** is different: a small number (15-20) of real photos of actual strays, taken by your team (phone photos are completely fine), which we manually look at and label using our ontology's trait vocabulary (e.g., "this one has semi-erect ears, short coat..."). This is the set that tests whether our system behaves sensibly on the animals StrayCare actually serves — the purebred dataset above only sanity-checks that the pipeline runs correctly.

---

## Chapter 4: CLIP and "Embeddings"

**CLIP** is an AI model, already trained by OpenAI on hundreds of millions of internet photo+caption pairs, that we download and use exactly as-is — like using a calculator instead of building one. Give it a photo and a sentence, and it tells you how well they match.

**"Embedding"** — CLIP doesn't compare a photo to a sentence the way we do visually. Internally, it converts both the photo and the sentence into a long list of numbers (a few hundred numbers) — a kind of numeric fingerprint of meaning — such that similar meanings produce similar-looking number lists. This is just CLIP's internal representation; nothing about it is mysterious, it's simply not human-readable directly.

**Why can't we just use CLIP's answer directly, then?** Because CLIP's output is a list of numbers, and our ontology can only understand named symbols (`FloppyEars`, not `[0.23, -0.87, ...]`). We need a translator in between — that's Chapter 5.

---

## Chapter 5: Training Our Own Small Model (the "translator")

**What does "training a model" actually mean, mechanically?** You show a program many examples of (input → correct answer) pairs, repeatedly, and after each attempt, it slightly adjusts its internal numbers to be a little more likely to get it right next time. Do this enough times, on enough examples, and it gets good at guessing correctly on *new* examples it's never seen. That repeated adjust-and-retry loop is called **training**, and the automated code that runs this loop end-to-end is called a **training pipeline**.

**What exactly are we training?** Not CLIP — CLIP stays completely untouched ("frozen"). We're training one small, simple additional program that takes CLIP's numeric fingerprint as input and learns to output trait guesses (ear shape, coat type, etc.) as its answer. This small program is what turns CLIP's numbers into ontology-readable symbols — the actual translator.

**Where do "correct answers" come from, without anyone hand-labeling thousands of photos?** A trick called **weak supervision**: our photo dataset already says "this photo = Beagle." Our ontology already says "Beagle = floppy ears." Combine the two existing facts, and you get "this photo = floppy ears," automatically, for free.

**What is a "loss curve"?** A simple line graph, produced automatically during training, showing how wrong the model's guesses were at each step. It should trend downward — that's visual proof training is actually working, and it's a great thing to show in your presentation.

**What is CUDA, and why does your GPU matter?** A GPU (Graphics Processing Unit — your RTX 3060) can do many small math calculations simultaneously, which is exactly what both "embedding lots of photos with CLIP" and "training our small model" need. CUDA is just the software layer that lets our Python code actually use the GPU instead of the slower general-purpose CPU. Practically: it means training that might take a while on a regular laptop takes minutes on your machine.

**Why do we deliberately NOT fine-tune CLIP itself?** If we let CLIP's own numbers change during training, it would get very good at our 13 specific breeds but *worse* at recognizing anything outside them — including the Local/Indigenous phenotype, which is the entire point of the project. Keeping CLIP frozen preserves its general ability to make sense of photos it's never specifically studied.

---

## Chapter 6: SPARQL — Asking the Ontology Questions

**SPARQL** is a question-asking language built specifically for ontologies — the same idea as SQL (a language for asking normal databases questions), just adapted for fact-networks instead of tables. "Which breeds have floppy ears?" becomes a precise, typed question the computer answers instantly by searching the arrows we defined in Chapter 2.

**How this fits together, end to end, in one sentence per step:**
Photo → CLIP turns it into numbers → our trained small model reads those numbers and guesses traits → we hand those trait guesses to a SPARQL question → the ontology returns which breeds match, and how well → we print an explained, ranked answer.

---

## Chapter 7: Tools You Need Installed — Checklist

| Tool | What it's for | How to get it |
|---|---|---|
| **Protégé** | Viewing/editing the ontology | Free download, protege.stanford.edu |
| **Python** | The programming language everything else runs in | python.org, or already on most machines |
| **PyTorch (with CUDA)** | Runs CLIP and trains our small model, using your GPU | `pip install torch` (GPU-enabled build for the person with the 3060) |
| **transformers** | The library that loads CLIP for us | `pip install transformers` |
| **rdflib** | Lets Python read the ontology file and run SPARQL queries | `pip install rdflib` |
| **scikit-learn** | Small helper tools for measuring accuracy | `pip install scikit-learn` |
| **torchvision** | Downloads the Oxford-IIIT Pet dataset automatically | `pip install torchvision` |
| **GitHub** (account + repo) | Shared place for all code and files, and for task tracking | github.com |

Nobody needs to install everything — only whoever is doing GPU/training work needs the CUDA-enabled PyTorch specifically; everyone else needs a normal install.

---

## Chapter 8: How to Split the Work — A Framework, Not a Roster

You know your teammates; I don't. Instead of assigning names, here's what *kind* of person naturally fits each piece — match your own people to these:

- **Ontology work** (Chapter 2): suits someone organized, detail-oriented, comfortable structuring information — genuinely does **not** require strong coding skills. Great for someone less confident in programming but strong on precision and logic.
- **Data collection & labeling** (Chapter 3): suits someone patient and methodical — mostly photography, organizing files, and careful manual labeling. Also a low-code-skill-required task.
- **Training/GPU work** (Chapter 5): suits whoever is most comfortable with Python and reading error messages calmly — this is the most technically dense piece.
- **Reasoning/integration code** (Chapter 6, gluing everything together): suits your strongest general-purpose programmer — this piece touches every other piece, so it benefits from someone who can read other people's code quickly.
- **Evaluation, writing, and tracking**: suits someone who's a clear writer and likes synthesizing results into a story — doesn't require deep technical background, but does require staying on top of what everyone else produced.

A person can hold more than one of these if your team of 5 doesn't map one-to-one — the pieces are separable, not rigid roles.

---

## Chapter 9: Keeping Track of Everything

Simplest possible setup: a **GitHub Project board** (free, lives next to your code) with four columns — *Backlog, In Progress, Blocked, Done* — and one card per task from the chapters above. A 15-minute nightly check-in where everyone moves their cards and says what's blocking them catches problems early, which matters more than any fancy tool.

---

## If You Get Stuck

Come back to whichever chapter covers the piece that's confusing, re-read just that section, and ask me about that one piece specifically — you don't need to hold the whole pipeline in your head at once to make progress on the one part in front of you right now.
