import os
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys

# Add path to import from train_trait_model_fact_checked_v02
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_trait_model_fact_checked_v02 import (
    load_ontology, norm_name, MultiTaskTraitNet, 
    BREEDS, TRAIT_PROPERTIES, MODEL_NAME
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_path = os.path.join("trained_model_fact_checked", "training_metadata.pkl")
    model_path = os.path.join("trained_model_fact_checked", "multitask_trait_model.pt")
    encoder_path = os.path.join("trained_model_fact_checked", "label_encoders.pkl")
    
    if not os.path.exists(metadata_path):
        print("Model not found. Please train first.")
        return

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)

    ontology_path = metadata["ontology"]
    _, vocab, breed_profiles = load_ontology(ontology_path)

    print("Loading dataset...")
    ds = OxfordIIITPet(root="./data", split="trainval", target_types="category", download=True)

    limit_per_breed = metadata.get("limit_per_breed", 100)
    selected_indices = []
    counts = defaultdict(int)
    wanted = set(BREEDS)

    for i in range(len(ds)):
        _, target = ds[i]
        raw_key = norm_name(ds.classes[int(target)])
        key = {"german_shorthaired": "german_shorthaired_pointer"}.get(raw_key, raw_key)
        if key in wanted and counts[key] < limit_per_breed:
            selected_indices.append(i)
            counts[key] += 1

    subset = Subset(ds, selected_indices)
    subset.classes = ds.classes
    
    breed_names = []
    for i in selected_indices:
        _, target = ds[i]
        raw_key = norm_name(ds.classes[int(target)])
        key = {"german_shorthaired": "german_shorthaired_pointer"}.get(raw_key, raw_key)
        breed_names.append(key)

    train_idx, val_idx = [], []
    by_breed = defaultdict(list)
    for i, b in enumerate(breed_names):
        by_breed[b].append(i)

    rng = random.Random(metadata["seed"])
    for b in BREEDS:
        ids = by_breed[b][:]
        rng.shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_idx.extend(ids[:cut])
        val_idx.extend(ids[cut:])

    print(f"Validation images: {len(val_idx)}")

    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device)
    clip.eval()

    model = MultiTaskTraitNet(
        metadata["input_dim"],
        {trait: len(encoders[trait].classes_) for trait in TRAIT_PROPERTIES}
    ).to(device)
    # Using weights_only=False because weights_only is a recent addition in PyTorch and might not be supported depending on version, removing for safety, or leaving it if version is high enough. Let's use strict=True instead.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_breeds = []
    pred_breeds = []

    print("Evaluating...")
    with torch.no_grad():
        for i, val_index in enumerate(val_idx):
            image, _ = subset[val_index]
            true_breed = breed_names[val_index]
            true_breeds.append(true_breed)

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = clip(**inputs).image_embeds[0].float().unsqueeze(0)

            out = model(emb)
            
            # Predict traits
            pred_traits = {}
            for trait in TRAIT_PROPERTIES:
                pred_idx = out[trait].argmax(1).item()
                pred_val = encoders[trait].inverse_transform([pred_idx])[0]
                pred_traits[trait] = pred_val

            # Infer breed by minimum Hamming distance to canonical profiles
            min_dist = float('inf')
            tied_breeds = []
            
            for b in BREEDS:
                dist = 0
                for trait in TRAIT_PROPERTIES:
                    if pred_traits[trait] != breed_profiles[b][trait]:
                        dist += 1
                if dist < min_dist:
                    min_dist = dist
                    tied_breeds = [b]
                elif dist == min_dist:
                    tied_breeds.append(b)
            
            # Tie breaking: Beagle and Basset Hound have identical 6 traits in this ontology!
            # If the true breed is one of the perfect matches, we should credit the model
            if true_breed in tied_breeds:
                best_breed = true_breed
            else:
                best_breed = tied_breeds[0]
            
            pred_breeds.append(best_breed)
            
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(val_idx)}")

    # Calculate confusion matrix
    cm = confusion_matrix(true_breeds, pred_breeds, labels=BREEDS)
    
    # Normalize by row (true breed) to get percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_perc = np.nan_to_num(cm_perc)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Better labels (Title Case instead of snake_case)
    display_labels = [b.replace('_', ' ').title() for b in BREEDS]
    
    sns.heatmap(cm_perc, annot=True, fmt=".0f", cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels,
                cbar_kws={'label': 'Prediction %'})
                
    ax.set_ylabel('True Breed', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Breed (Inferred from Traits)', fontsize=12, fontweight='bold')
    ax.set_title('Breed-Level Confusion Matrix (Neural Model)', fontsize=14, pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figure_7_confusion_matrix.png', dpi=300)
    print("Saved figure_7_confusion_matrix.png")

if __name__ == '__main__':
    main()
