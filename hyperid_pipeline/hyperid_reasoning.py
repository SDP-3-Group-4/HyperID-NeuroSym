"""
HYPER-ID: frozen CLIP -> multi-task trait model -> ontology/SPARQL reasoning.

The reasoning query follows Query #3 from sparql_example_queries-1.md:
rank candidate breeds by the number of shared detected traits.
"""
import argparse, pickle
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from rdflib import Graph

MODEL_NAME = "openai/clip-vit-base-patch32"
ONTOLOGY_PATH = "breed_trait_ontology.ttl"
MODEL_PATH = "trained_model/multitask_trait_model.pt"
ENCODER_PATH = "trained_model/label_encoders.pkl"
ONT_NS = "http://straycare.org/ontology/breed#"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIT_PROPERTIES = {
    "EarShape": "hasEarShape", "CoatType": "hasCoatType",
    "CoatPattern": "hasCoatPattern", "SizeClass": "hasSizeClass",
    "SnoutLength": "hasSnoutLength", "TailCarriage": "hasTailCarriage",
}

class MultiTaskTraitNet(nn.Module):
    def __init__(self, input_dim, num_classes_per_trait):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleDict({t: nn.Linear(256, n) for t, n in num_classes_per_trait.items()})
    def forward(self, x):
        shared = self.trunk(x)
        return {t: head(shared) for t, head in self.heads.items()}

def load_system():
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    num_classes = {t: len(encoders[t].classes_) for t in TRAIT_PROPERTIES}
    trait_model = MultiTaskTraitNet(768, num_classes).to(DEVICE)
    trait_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    trait_model.eval()

    clip = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    for p in clip.parameters(): p.requires_grad = False

    graph = Graph()
    graph.parse(ONTOLOGY_PATH, format="turtle")
    return clip, processor, trait_model, encoders, graph

@torch.no_grad()
def predict_traits(image, clip, processor, trait_model, encoders, top_k=3):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    embedding = clip(**inputs).image_embeds
    outputs = trait_model(embedding)
    predictions = {}
    for trait in TRAIT_PROPERTIES:
        probs = torch.softmax(outputs[trait][0], dim=0)
        values, indices = probs.topk(min(top_k, len(encoders[trait].classes_)))
        predictions[trait] = [
            (encoders[trait].inverse_transform([int(i.cpu())])[0], float(p.cpu()))
            for i, p in zip(indices, values)
        ]
    return predictions

def rank_breeds_by_trait_overlap(predictions, graph, top_k=5):
    """Exact structure of supplied SPARQL Query #3, with VALUES generated from the image."""
    detected_values = [vals[0][0] for vals in predictions.values()]
    values_tokens = " ".join(f":{v}" for v in detected_values)
    query = f"""
    PREFIX : <{ONT_NS}>
    SELECT ?breed (COUNT(?trait) AS ?sharedTraits) WHERE {{
        VALUES ?trait {{ {values_tokens} }}
        ?breed :hasTrait ?trait .
    }}
    GROUP BY ?breed
    ORDER BY DESC(?sharedTraits)
    LIMIT {int(top_k)}
    """
    return [
        (str(row.breed).split("#")[-1], int(row.sharedTraits))
        for row in graph.query(query)
    ]

def main(image_path):
    print(f"Using device: {DEVICE}")
    clip, processor, trait_model, encoders, graph = load_system()
    image = Image.open(image_path).convert("RGB")
    predictions = predict_traits(image, clip, processor, trait_model, encoders)

    print("\n=== DETECTED TRAIT PROFILE ===")
    for trait, vals in predictions.items():
        print(f"{trait:<15} {vals[0][0]:<25} {vals[0][1]:.2%}")

    ranked = rank_breeds_by_trait_overlap(predictions, graph)
    print("\n=== ONTOLOGY REASONING ===")
    print("SPARQL Query #3: rank candidates by shared detected traits")
    print("\n=== RANKED CANDIDATES ===")
    if not ranked:
        print("No matching ontology candidates.")
    else:
        for i, (breed, score) in enumerate(ranked, 1):
            print(f"{i}. {breed:<35} shared traits = {score}/6")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    main(parser.parse_args().image)
