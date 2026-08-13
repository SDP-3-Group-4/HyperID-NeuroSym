import os
import pickle
import argparse

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPProcessor


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./trained_model_fact_checked/multitask_trait_model.pt"
ENCODER_PATH = "./trained_model_fact_checked/label_encoders.pkl"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAITS = [
    "EarShape",
    "CoatType",
    "CoatPattern",
    "SizeClass",
    "SnoutLength",
    "TailCarriage",
]


# ============================================================
# SAME NETWORK ARCHITECTURE USED DURING TRAINING
# ============================================================

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

        return {
            trait: head(shared)
            for trait, head in self.heads.items()
        }


# ============================================================
# LOAD SYSTEM
# ============================================================

def load_system():

    print(f"Using device: {DEVICE}")

    # Frozen CLIP
    clip = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_MODEL_NAME
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    clip.eval()

    for p in clip.parameters():
        p.requires_grad = False

    # Label encoders
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    # Automatically determine CLIP embedding size
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    input_dim = checkpoint["trunk.0.weight"].shape[1]

    num_classes = {
        trait: len(encoders[trait].classes_)
        for trait in TRAITS
    }

    print(f"CLIP embedding size: {input_dim}")
    print("Trait classes:")

    for trait in TRAITS:
        print(
            f"  {trait}: "
            f"{list(encoders[trait].classes_)}"
        )

    # Trait network
    model = MultiTaskTraitNet(
        input_dim,
        num_classes
    ).to(DEVICE)

    model.load_state_dict(checkpoint)
    model.eval()

    return clip, processor, model, encoders


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(image_path):

    clip, processor, model, encoders = load_system()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    # PHOTO -> CLIP EMBEDDING
    embedding = clip(**inputs).image_embeds

    print("\nCLIP embedding generated.")
    print(f"Embedding shape: {tuple(embedding.shape)}")

    # EMBEDDING -> SIX TRAIT HEADS
    outputs = model(embedding)

    print("\n" + "=" * 60)
    print("TRAIT MODEL CROSS-CHECK")
    print("=" * 60)

    predictions = {}

    for trait in TRAITS:

        probabilities = torch.softmax(
            outputs[trait],
            dim=1
        )[0]

        # Highest probability
        confidence, index = torch.max(
            probabilities,
            dim=0
        )

        label = encoders[trait].inverse_transform(
            [index.item()]
        )[0]

        predictions[trait] = label

        print(f"\n{trait}")
        print(
            f"  PREDICTION: {label} "
            f"({confidence.item() * 100:.2f}%)"
        )

        # Show all classes
        ranked = torch.argsort(
            probabilities,
            descending=True
        )

        print("  Alternatives:")

        for i in ranked[:3]:
            name = encoders[trait].inverse_transform(
                [i.item()]
            )[0]

            print(
                f"    {name:<25} "
                f"{probabilities[i].item() * 100:6.2f}%"
            )

    # ========================================================
    # OUR MANUAL EXPECTED PROFILE
    # ========================================================

    expected = {
        "EarShape": "ErectEars",
        "CoatType": "ShortCoat",
        "CoatPattern": "PatchedColor",
        "SizeClass": "LargeSize",
        "SnoutLength": "LongSnout",
        "TailCarriage": "StraightTail",
    }

    print("\n" + "=" * 60)
    print("MANUAL PROFILE vs MODEL")
    print("=" * 60)

    matches = 0

    for trait in TRAITS:

        predicted = predictions[trait]
        expected_value = expected[trait]

        match = predicted == expected_value

        if match:
            matches += 1

        symbol = "MATCH" if match else "DIFFER"

        print(
            f"{trait:<15} "
            f"Expected: {expected_value:<22} "
            f"Model: {predicted:<22} "
            f"[{symbol}]"
        )

    print("\nAgreement:")
    print(f"{matches}/6 traits")

    print("\nNOTE:")
    print(
        "This is only a cross-check against our manually proposed "
        "phenotype profile. It does NOT prove the breed identity."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Cross-check trained trait model on a new image."
    )

    parser.add_argument(
        "image",
        help="Path to test image"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    predict(args.image)