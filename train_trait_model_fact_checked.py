import argparse
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rdflib import Graph, Namespace, RDF
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
ONTOLOGY_NS = "http://straycare.org/ontology/breed#"

TRAIT_PROPERTIES = {
    "EarShape": "hasEarShape",
    "CoatType": "hasCoatType",
    "CoatPattern": "hasCoatPattern",
    "SizeClass": "hasSizeClass",
    "SnoutLength": "hasSnoutLength",
    "TailCarriage": "hasTailCarriage",
}

# These are the Oxford-IIIT Pet classes used by the current project.
# The ontology contains the corresponding canonical breed individuals.
BREEDS = [
    "beagle",
    "boxer",
    "chihuahua",
    "pug",
    "samoyed",
    "shiba_inu",
    "great_pyrenees",
    "german_shorthaired_pointer",
    "staffordshire_bull_terrier",
    "yorkshire_terrier",
    "pomeranian",
    "basset_hound",
    "saint_bernard",
]

# torchvision's Oxford-IIIT Pet names -> ontology individuals.
BREED_ALIASES = {
    "german_shorthaired": "GermanShorthairedPointer",
    "staffordshire_bull_terrier": "StaffordshireBullTerrier",
    "yorkshire_terrier": "YorkshireTerrier",
    "great_pyrenees": "GreatPyrenees",
    "shiba_inu": "ShibaInu",
    "saint_bernard": "SaintBernard",
}


def norm_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def ontology_name_from_dataset(name):
    n = norm_name(name)
    return BREED_ALIASES.get(n, "".join(part.title() for part in n.split("_")))


def load_ontology(path):
    g = Graph()
    g.parse(path, format="turtle")
    ns = Namespace(ONTOLOGY_NS)

    # Extract the complete allowed vocabulary for each trait from the TTL.
    trait_classes = {
        "EarShape": ns.EarShape,
        "CoatType": ns.CoatType,
        "CoatPattern": ns.CoatPattern,
        "SizeClass": ns.SizeClass,
        "SnoutLength": ns.SnoutLength,
        "TailCarriage": ns.TailCarriage,
    }

    vocab = {}
    for trait, cls in trait_classes.items():
        values = sorted(str(x).split("#")[-1] for x in g.subjects(RDF.type, cls))
        if not values:
            raise RuntimeError(f"No trait individuals found for {trait} in {path}")
        vocab[trait] = values

    # Build breed -> six canonical trait labels directly from TTL.
    breed_profiles = {}
    for breed in BREEDS:
        ont_name = ontology_name_from_dataset(breed)
        subject = ns[ont_name]
        if (subject, RDF.type, ns.Breed) not in g:
            raise RuntimeError(
                f"Ontology is missing breed individual :{ont_name} required for '{breed}'."
            )

        profile = {}
        for trait, prop_name in TRAIT_PROPERTIES.items():
            prop = ns[prop_name]
            values = list(g.objects(subject, prop))
            if len(values) != 1:
                raise RuntimeError(
                    f":{ont_name} must have exactly one {prop_name}; found {len(values)}."
                )
            value = str(values[0]).split("#")[-1]
            if value not in vocab[trait]:
                raise RuntimeError(
                    f":{ont_name} uses unknown {trait} value :{value}."
                )
            profile[trait] = value
        breed_profiles[breed] = profile

    return g, vocab, breed_profiles


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], {k: v[idx] for k, v in self.labels.items()}


class MultiTaskTraitNet(nn.Module):
    # Must stay compatible with hyperid_reasoning.py / crosscheck scripts:
    # 512 -> 256 -> six heads.
    def __init__(self, input_dim, num_classes_per_trait):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({
            trait: nn.Linear(256, n_classes)
            for trait, n_classes in num_classes_per_trait.items()
        })

    def forward(self, x):
        shared = self.trunk(x)
        return {trait: head(shared) for trait, head in self.heads.items()}


@torch.no_grad()
def build_embeddings(dataset, clip, processor, device, breed_profiles):
    """Precompute frozen CLIP image embeddings once."""
    embeddings = []
    breed_names = []

    for i in range(len(dataset)):
        image, target = dataset[i]
        dataset_breed = dataset.classes[int(target)]
        key = norm_name(dataset_breed)
        if key not in breed_profiles:
            raise RuntimeError(f"Unexpected dataset breed: {dataset_breed}")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clip(**inputs).image_embeds[0].float().cpu()
        embeddings.append(emb)
        breed_names.append(key)

        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"Embedding {i + 1}/{len(dataset)}")

    return torch.stack(embeddings), breed_names


def make_labels(breed_names, breed_profiles, encoders):
    labels = {}
    for trait in TRAIT_PROPERTIES:
        values = [breed_profiles[b][trait] for b in breed_names]
        labels[trait] = torch.tensor(
            encoders[trait].transform(values), dtype=torch.long
        )
    return labels


def evaluate(model, loader, device, traits):
    model.eval()
    correct = {t: 0 for t in traits}
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = {t: y[t].to(device) for t in traits}
            out = model(x)
            loss = sum(criterion(out[t], y[t]) for t in traits) / len(traits)
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            for t in traits:
                pred = out[t].argmax(1)
                correct[t] += (pred == y[t]).sum().item()

    acc = {t: correct[t] / total for t in traits}
    return loss_sum / total, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", default="breed_trait_ontology_fact_checked_v02.ttl")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="trained_model_fact_checked")
    parser.add_argument("--limit-per-breed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.ontology):
        raise FileNotFoundError(f"Ontology not found: {args.ontology}")

    print(f"Loading ontology: {args.ontology}")
    _, vocab, breed_profiles = load_ontology(args.ontology)

    print("\n=== ONTOLOGY LABELS USED FOR TRAINING ===")
    for breed in BREEDS:
        print(f"{breed:32s} " + " | ".join(
            f"{t}={breed_profiles[breed][t]}" for t in TRAIT_PROPERTIES
        ))

    print("\nLoading Oxford-IIIT Pet trainval split...")
    ds = OxfordIIITPet(
        root=args.data,
        split="trainval",
        target_types="category",
        download=True,
    )

    # Keep exactly the 13 project breeds and up to N images per breed.
    selected_indices = []
    counts = defaultdict(int)
    wanted = set(BREEDS)

    for i in range(len(ds)):
        _, target = ds[i]
        raw_name = ds.classes[int(target)]
        key = norm_name(raw_name)
        if key in wanted and counts[key] < args.limit_per_breed:
            selected_indices.append(i)
            counts[key] += 1

    missing = [b for b in BREEDS if counts[b] == 0]
    if missing:
        raise RuntimeError(f"Missing Oxford breeds: {missing}")

    print(f"Selected {len(selected_indices)} images from {len(BREEDS)} breeds.")
    print("Per-breed counts:", dict(counts))

    # Subset while preserving the original Oxford dataset object/classes.
    from torch.utils.data import Subset
    subset = Subset(ds, selected_indices)
    subset.classes = ds.classes

    print("\nLoading frozen CLIP vision encoder...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    # Precompute embeddings. This makes the actual NN training extremely fast.
    print("\nPrecomputing CLIP embeddings...")
    embeddings = []
    breed_names = []
    with torch.no_grad():
        for j in range(len(subset)):
            image, target = subset[j]
            raw_name = ds.classes[int(target)]
            key = norm_name(raw_name)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = clip(**inputs).image_embeds[0].float().cpu()
            embeddings.append(emb)
            breed_names.append(key)
            if (j + 1) % 100 == 0 or j + 1 == len(subset):
                print(f"  {j + 1}/{len(subset)}")

    embeddings = torch.stack(embeddings)

    # Deterministic 80/20 split, stratified by breed.
    train_idx, val_idx = [], []
    by_breed = defaultdict(list)
    for i, b in enumerate(breed_names):
        by_breed[b].append(i)

    rng = random.Random(args.seed)
    for b in BREEDS:
        ids = by_breed[b][:]
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_idx.extend(ids[:cut])
        val_idx.extend(ids[cut:])

    train_breeds = [breed_names[i] for i in train_idx]
    val_breeds = [breed_names[i] for i in val_idx]

    # Full vocabulary comes from the ontology, not from observed labels.
    encoders = {}
    for trait, classes in vocab.items():
        enc = LabelEncoder()
        enc.fit(classes)
        encoders[trait] = enc

    all_labels = make_labels(breed_names, breed_profiles, encoders)
    train_labels = {t: v[train_idx] for t, v in all_labels.items()}
    val_labels = {t: v[val_idx] for t, v in all_labels.items()}

    train_ds = EmbeddingDataset(embeddings[train_idx], train_labels)
    val_ds = EmbeddingDataset(embeddings[val_idx], val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskTraitNet(
        512,
        {trait: len(encoders[trait].classes_) for trait in TRAIT_PROPERTIES},
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    patience = 6
    stale = 0

    print("\n=== TRAINING ===")
    print(f"Train images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = {t: y[t].to(device) for t in TRAIT_PROPERTIES}
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = sum(criterion(out[t], y[t]) for t in TRAIT_PROPERTIES) / len(TRAIT_PROPERTIES)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running / n
        val_loss, val_acc = evaluate(model, val_loader, device, TRAIT_PROPERTIES)
        mean_acc = sum(val_acc.values()) / len(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mean_trait_acc={mean_acc*100:.2f}%"
        )
        print("  " + "  ".join(f"{t}={val_acc[t]*100:.1f}%" for t in TRAIT_PROPERTIES))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")

    model.load_state_dict(best_state)
    model.to(device)
    final_loss, final_acc = evaluate(model, val_loader, device, TRAIT_PROPERTIES)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "multitask_trait_model.pt")
    encoder_path = os.path.join(args.out, "label_encoders.pkl")
    metadata_path = os.path.join(args.out, "training_metadata.pkl")

    torch.save(model.state_dict(), model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)

    metadata = {
        "model_name": MODEL_NAME,
        "ontology": os.path.abspath(args.ontology),
        "input_dim": 512,
        "hidden_dim": 256,
        "traits": list(TRAIT_PROPERTIES.keys()),
        "vocab": vocab,
        "breeds": BREEDS,
        "limit_per_breed": args.limit_per_breed,
        "train_count": len(train_ds),
        "validation_count": len(val_ds),
        "seed": args.seed,
        "best_val_loss": final_loss,
        "validation_trait_accuracy": final_acc,
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n=== FINAL VALIDATION ===")
    print(f"Images: {len(subset)}")
    print(f"Train:  {len(train_ds)}")
    print(f"Val:    {len(val_ds)}")
    print(f"Mean trait accuracy: {sum(final_acc.values())/len(final_acc)*100:.2f}%")
    for trait in TRAIT_PROPERTIES:
        print(f"{trait:15s}: {final_acc[trait]*100:.2f}%")

    print("\n=== SAVED ===")
    print(model_path)
    print(encoder_path)
    print(metadata_path)
    print("\nThis checkpoint is compatible with the current 512 -> 256 -> six-head reasoning architecture.")


if __name__ == "__main__":
    main()
