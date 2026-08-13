"""
train_trait_model.py
----------------------------------------------------------------------
This REPLACES the earlier train_trait_probes.py. Use this version.

PLAIN-LANGUAGE OVERVIEW — read this before the code below.

WHY THIS STEP EXISTS AT ALL:
CLIP turns a photo into a list of numbers (an "embedding"). Our ontology
can only reason using symbols it knows about (FloppyEars, ShortCoat...),
not raw numbers. This script trains a small neural network whose entire
job is translation: numbers in, trait symbols out. That translation is
what lets a knowledge graph reason about something a neural network saw.
Without it, there is no bridge between perception and reasoning.

WHAT CHANGED FROM THE FIRST VERSION:
Instead of six separate, independent classifiers (one per trait, each
learning in isolation), this trains ONE joint neural network with a
shared trunk and six output heads -- so the model can learn patterns
that connect traits (e.g. certain ear shapes and coat types tend to
co-occur) rather than treating every trait dimension as unrelated. It
also reports a real comparison: joint multi-task learning vs. six
independent classifiers, on the same held-out data. That comparison
number is the actual research result this training step produces.

CLIP ITSELF IS NEVER TRAINED. Its weights stay completely frozen. Only
the small head we add on top is trained. This is deliberate: fine-tuning
CLIP itself would overfit to our 13 breeds and destroy its ability to
generalize to phenotypes it has little or no training data for --
including the Local/Indigenous node, which is the whole point of the
project.

STEPS:
1. Read breed -> trait facts out of the ontology via SPARQL (same as before).
2. Load frozen CLIP on the GPU if available.
3. Embed every image (+ a horizontally-flipped copy of each, to stretch
   our small photo count) -- this is where the GPU actually saves time.
4. Train ONE joint multi-task neural network (shared trunk + 6 heads),
   with a real training loop, real loss curve, on the GPU.
5. ALSO train the old six-independent-classifiers version on the exact
   same data, purely as a comparison baseline.
6. Print + plot both results side by side, save the trained model and
   the loss curve figure.

HOW TO RUN:
    pip install torch torchvision transformers rdflib scikit-learn pillow matplotlib
    python train_trait_model.py

Place this file in the same folder as breed_trait_ontology.ttl before running.
----------------------------------------------------------------------
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import hflip
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from rdflib import Graph

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ONTOLOGY_PATH = "breed_trait_ontology.ttl"
DATA_ROOT = "./data"
OUTPUT_DIR = "./trained_model"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

BREED_NAME_MAP = {
    "beagle": "Beagle", "boxer": "Boxer", "chihuahua": "Chihuahua",
    "pug": "Pug", "samoyed": "Samoyed", "shiba_inu": "ShibaInu",
    "great_pyrenees": "GreatPyrenees", "german_shorthaired": "GermanShorthairedPointer",
    "staffordshire_bull_terrier": "StaffordshireBullTerrier",
    "yorkshire_terrier": "YorkshireTerrier", "pomeranian": "Pomeranian",
    "basset_hound": "BassetHound", "saint_bernard": "SaintBernard",
}

TRAIT_PROPERTIES = {
    "EarShape": "hasEarShape", "CoatType": "hasCoatType",
    "CoatPattern": "hasCoatPattern", "SizeClass": "hasSizeClass",
    "SnoutLength": "hasSnoutLength", "TailCarriage": "hasTailCarriage",
}
ONT_NS = "http://straycare.org/ontology/breed#"


# ----------------------------------------------------------------------
# STEP 1 — breed -> trait facts, straight from the ontology
# ----------------------------------------------------------------------
def load_breed_traits_from_ontology(path):
    g = Graph()
    g.parse(path, format="turtle")
    breed_traits = defaultdict(dict)
    for trait_dim, prop in TRAIT_PROPERTIES.items():
        query = f"""
        PREFIX : <{ONT_NS}>
        SELECT ?breed ?value WHERE {{ ?breed :{prop} ?value . }}
        """
        for row in g.query(query):
            breed_local = str(row.breed).split("#")[-1]
            value_local = str(row.value).split("#")[-1]
            breed_traits[breed_local][trait_dim] = value_local
    return breed_traits


# ----------------------------------------------------------------------
# STEP 2 — frozen CLIP, on GPU if available
# ----------------------------------------------------------------------
def load_clip():
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # explicit: CLIP's own weights never update
    return model, processor


@torch.no_grad()
def embed_image(model, processor, image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    output = model(**inputs)
    features = output.image_embeds  # stable, documented field for this model class
    return features.squeeze(0).cpu().numpy()


# ----------------------------------------------------------------------
# STEP 3 — embed images (+ flipped copies) and attach weak trait labels
# ----------------------------------------------------------------------
def build_training_data(breed_traits, model, processor):
    print("Loading Oxford-IIIT Pet (13 target dog breeds only)...")
    dataset = OxfordIIITPet(root=DATA_ROOT, download=True, target_types="category")
    class_names = dataset.classes

    embeddings, labels_per_trait = [], defaultdict(list)
    kept = 0
    for image, class_idx in dataset:
        # torchvision auto-formats class names as "Title Case With Spaces"
        # (e.g. "Yorkshire Terrier"), not the raw "yorkshire_terrier" form —
        # normalize before comparing against BREED_NAME_MAP's keys.
        raw_name = class_names[class_idx].lower().replace(" ", "_")
        if raw_name not in BREED_NAME_MAP:
            continue
        ontology_breed = BREED_NAME_MAP[raw_name]
        traits = breed_traits.get(ontology_breed)
        if not traits:
            continue

        for img_variant in (image, hflip(image)):  # original + flipped = 2x data, GPU makes this cheap
            emb = embed_image(model, processor, img_variant)
            embeddings.append(emb)
            for trait_dim in TRAIT_PROPERTIES:
                labels_per_trait[trait_dim].append(traits[trait_dim])

        kept += 1
        if kept % 50 == 0:
            print(f"  embedded {kept} source images ({kept * 2} total incl. flips)...")

    print(f"Done. {kept} source images -> {kept * 2} embeddings.\n")
    return np.array(embeddings), labels_per_trait


# ----------------------------------------------------------------------
# STEP 4 — the joint multi-task neural network
# ----------------------------------------------------------------------
class MultiTaskTraitNet(nn.Module):
    """One shared trunk, one small classification head per trait dimension."""
    def __init__(self, input_dim, num_classes_per_trait):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({
            trait: nn.Linear(256, n_classes)
            for trait, n_classes in num_classes_per_trait.items()
        })

    def forward(self, x):
        shared = self.trunk(x)
        return {trait: head(shared) for trait, head in self.heads.items()}


def train_multitask_model(embeddings, labels_per_trait):
    encoders = {trait: LabelEncoder().fit(labels) for trait, labels in labels_per_trait.items()}
    encoded_labels = {trait: encoders[trait].transform(labels) for trait, labels in labels_per_trait.items()}
    num_classes_per_trait = {trait: len(enc.classes_) for trait, enc in encoders.items()}

    idx = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    X_test = torch.tensor(embeddings[test_idx], dtype=torch.float32)
    y_train = {t: torch.tensor(encoded_labels[t][train_idx], dtype=torch.long) for t in TRAIT_PROPERTIES}
    y_test = {t: torch.tensor(encoded_labels[t][test_idx], dtype=torch.long) for t in TRAIT_PROPERTIES}

    loader = DataLoader(TensorDataset(X_train, *[y_train[t] for t in TRAIT_PROPERTIES]),
                         batch_size=BATCH_SIZE, shuffle=True)

    model = MultiTaskTraitNet(embeddings.shape[1], num_classes_per_trait).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    trait_names = list(TRAIT_PROPERTIES.keys())

    loss_history = []
    print("Training joint multi-task network...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            xb, *yb = batch
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = sum(criterion(outputs[t], yb[i].to(DEVICE)) for i, t in enumerate(trait_names))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS} — loss: {loss_history[-1]:.4f}")

    # evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(DEVICE))
        multitask_acc = {}
        for t in trait_names:
            preds = outputs[t].argmax(dim=1).cpu().numpy()
            multitask_acc[t] = accuracy_score(y_test[t].numpy(), preds)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "multitask_trait_model.pt"))
    with open(os.path.join(OUTPUT_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Multi-task trait model — training loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    print(f"Saved model, label encoders, and loss curve to {OUTPUT_DIR}\n")

    return multitask_acc, X_train.numpy(), X_test.numpy(), encoded_labels, train_idx, test_idx


# ----------------------------------------------------------------------
# STEP 5 — baseline: six INDEPENDENT classifiers, same data, for comparison
# ----------------------------------------------------------------------
def train_independent_baseline(X_train, X_test, encoded_labels, train_idx, test_idx):
    print("Training independent single-task baselines (comparison only)...")
    baseline_acc = {}
    for trait in TRAIT_PROPERTIES:
        y_train = encoded_labels[trait][train_idx]
        y_test = encoded_labels[trait][test_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        baseline_acc[trait] = accuracy_score(y_test, clf.predict(X_test))
    return baseline_acc


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    print("Step 1/4 — reading breed-trait facts from the ontology...")
    breed_traits = load_breed_traits_from_ontology(ONTOLOGY_PATH)
    print(f"  loaded trait profiles for {len(breed_traits)} breeds.\n")

    print("Step 2/4 — loading frozen CLIP...")
    clip_model, clip_processor = load_clip()
    print("  done.\n")

    print("Step 3/4 — embedding images (+ flips) and attaching weak labels...")
    embeddings, labels_per_trait = build_training_data(breed_traits, clip_model, clip_processor)

    print("Step 4/4 — training joint multi-task model + independent baseline...")
    multitask_acc, X_train, X_test, encoded_labels, train_idx, test_idx = train_multitask_model(
        embeddings, labels_per_trait
    )
    baseline_acc = train_independent_baseline(X_train, X_test, encoded_labels, train_idx, test_idx)

    print("\n=== RESULT: joint multi-task network vs. independent classifiers ===")
    print(f"{'Trait':<15} {'Multi-task':>12} {'Independent':>14}")
    for trait in TRAIT_PROPERTIES:
        print(f"{trait:<15} {multitask_acc[trait]:>11.2%} {baseline_acc[trait]:>13.2%}")
