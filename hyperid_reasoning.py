import argparse
import pickle
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from rdflib import Graph

MODEL_NAME   = "openai/clip-vit-base-patch32"
ONTOLOGY_PATH = "breed_trait_ontology_fact_checked_v02.ttl"
MODEL_PATH   = "trained_model_fact_checked/multitask_trait_model.pt"
ENCODER_PATH = "trained_model_fact_checked/label_encoders.pkl"
ONT_NS       = "http://straycare.org/ontology/breed#"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIT_PROPERTIES = {
    "EarShape":     "hasEarShape",
    "CoatType":     "hasCoatType",
    "CoatPattern":  "hasCoatPattern",
    "SizeClass":    "hasSizeClass",
    "SnoutLength":  "hasSnoutLength",
    "TailCarriage": "hasTailCarriage",
}

# Natural-language prompts used by CLIP for zero-shot trait scoring.
TRAIT_PROMPTS = {
    "EarShape": {
        "FloppyEars":    "a dog with floppy ears",
        "ErectEars":     "a dog with erect upright ears",
        "SemiErectEars": "a dog with semi-erect ears",
        "ButtonEars":    "a dog with small button ears",
    },
    "CoatType": {
        "ShortCoat":  "a dog with a short coat",
        "MediumCoat": "a dog with a medium-length coat",
        "LongCoat":   "a dog with a long coat",
        "WireCoat":   "a dog with a wiry coat",
    },
    "CoatPattern": {
        "SolidColor":   "a dog with a solid single-color coat",
        "Brindle":      "a dog with a brindle coat pattern",
        "Spotted":      "a dog with a spotted coat pattern",
        "Tricolor":     "a dog with a tricolor coat",
        "PatchedColor": "a dog with patches of different coat colors",
    },
    "SizeClass": {
        "ToySize":   "a toy-sized dog",
        "SmallSize": "a small-sized dog",
        "MediumSize":"a medium-sized dog",
        "LargeSize": "a large-sized dog",
        "GiantSize": "a giant-sized dog",
    },
    "SnoutLength": {
        "ShortSnout":  "a dog with a short snout",
        "MediumSnout": "a dog with a medium-length snout",
        "LongSnout":   "a dog with a long snout",
    },
    "TailCarriage": {
        "CurledTail":   "a dog with a curled tail",
        "StraightTail": "a dog with a straight tail",
        "PlumeTail":    "a dog with a long plume-like tail",
        "DockedTail":   "a dog with a docked tail",
    },
}


class MultiTaskTraitNet(nn.Module):
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


def load_system():
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    num_classes = {
        trait: len(encoders[trait].classes_)
        for trait in TRAIT_PROPERTIES
    }

    trait_model = MultiTaskTraitNet(512, num_classes).to(DEVICE)
    trait_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    trait_model.eval()

    # Full CLIPModel needed for both vision (neural embedding) and
    # text (zero-shot trait scoring).
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    graph = Graph()
    graph.parse(ONTOLOGY_PATH, format="turtle")

    return clip, processor, trait_model, encoders, graph


@torch.no_grad()
def neural_probs(image, clip, processor, trait_model, encoders):
    """Return per-trait probability dicts from the trained neural model."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    # Replicate the exact CLIP projection used during training.
    vision_out = clip.vision_model(pixel_values=inputs["pixel_values"])
    emb = clip.visual_projection(vision_out.pooler_output)
    outputs = trait_model(emb)

    result = {}
    for trait in TRAIT_PROPERTIES:
        probs = torch.softmax(outputs[trait][0], dim=0)
        labels = list(encoders[trait].classes_)
        result[trait] = {labels[i]: float(probs[i].cpu()) for i in range(len(labels))}
    return result


@torch.no_grad()
def clip_probs(image, clip, processor):
    """Return per-trait probability dicts from zero-shot CLIP text scoring."""
    result = {}
    for trait, prompts in TRAIT_PROMPTS.items():
        labels = list(prompts)
        texts  = [prompts[l] for l in labels]
        inputs = processor(
            text=texts, images=image,
            return_tensors="pt", padding=True
        ).to(DEVICE)
        logits = clip(**inputs).logits_per_image[0]
        probs  = torch.softmax(logits, dim=0)
        result[trait] = {labels[i]: float(probs[i].cpu()) for i in range(len(labels))}
    return result


def combine_probs(neural, clip_scores, clip_weight):
    """
    Log-space weighted combination of neural and CLIP probability dicts.
    clip_weight=0.65 means CLIP contributes 65 % and neural 35 %.
    """
    eps = 1e-8
    combined = {}
    neural_weight = 1.0 - clip_weight
    for trait in TRAIT_PROPERTIES:
        labels = list(neural[trait])
        scores = []
        for lab in labels:
            n_p = max(neural[trait].get(lab, eps), eps)
            c_p = max(clip_scores[trait].get(lab, eps), eps)
            s = neural_weight * torch.log(torch.tensor(n_p)) \
              + clip_weight  * torch.log(torch.tensor(c_p))
            scores.append(s)
        probs = torch.softmax(torch.stack(scores), dim=0)
        combined[trait] = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return combined


def rank_breeds_by_trait_overlap(combined, graph, top_k=5):
    """
    SPARQL trait-overlap query using the top-1 trait from the ensemble.
    """
    rows = []
    for trait, dist in combined.items():
        top_value = max(dist, key=dist.get)
        prop = TRAIT_PROPERTIES[trait]
        rows.append(f"(:{prop} :{top_value})")

    if not rows:
        return []

    query = f"""
    PREFIX : <{ONT_NS}>

    SELECT ?breed (COUNT(?matchedTrait) AS ?sharedTraits)
    WHERE {{
        VALUES (?property ?trait) {{
            {" ".join(rows)}
        }}
        ?breed ?property ?trait .
        BIND(?trait AS ?matchedTrait)
    }}
    GROUP BY ?breed
    ORDER BY DESC(?sharedTraits) ?breed
    LIMIT {int(top_k)}
    """
    return [
        (str(row.breed).split("#")[-1], int(row.sharedTraits))
        for row in graph.query(query)
    ]


def print_result(combined, ranked):
    print("\n=== DETECTED TRAIT PROFILE (CLIP+Neural Ensemble) ===")
    for trait, dist in combined.items():
        top_label = max(dist, key=dist.get)
        confidence = dist[top_label]
        print(f"{trait:<15} {top_label:<25} {confidence:.2%}")

    print("\n=== ONTOLOGY REASONING ===")
    print("Trait-overlap reasoning using the ontology's six properties.")

    print("\n=== RANKED CANDIDATES ===")
    if not ranked:
        print("No matching ontology candidates.")
        return

    for rank, (breed, score) in enumerate(ranked, 1):
        print(f"{rank}. {breed:<35} shared traits = {score}/6")


def main(image_path, clip_weight):
    print(f"Using device: {DEVICE}")
    print(f"CLIP weight: {clip_weight:.2f} / Neural weight: {1-clip_weight:.2f}")

    clip, processor, trait_model, encoders, graph = load_system()
    image = Image.open(image_path).convert("RGB")

    neural  = neural_probs(image, clip, processor, trait_model, encoders)
    clip_sc = clip_probs(image, clip, processor)
    combined = combine_probs(neural, clip_sc, clip_weight)

    ranked = rank_breeds_by_trait_overlap(combined, graph)
    print_result(combined, ranked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperID-KE: CLIP+Neural ensemble → ontology reasoning"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--clip-weight", type=float, default=0.65,
        help="Weight given to CLIP zero-shot scorer (0–1). Default: 0.65"
    )
    args = parser.parse_args()
    if not 0 <= args.clip_weight <= 1:
        raise ValueError("--clip-weight must be between 0 and 1")
    main(args.image, args.clip_weight)
