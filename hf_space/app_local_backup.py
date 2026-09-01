import pickle
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor
from rdflib import Graph

BASE_DIR = Path(__file__).parent
MODEL_NAME = "openai/clip-vit-base-patch32"
ONTOLOGY_PATH = BASE_DIR / "breed_trait_ontology_fact_checked_v02.ttl"
MODEL_PATH = BASE_DIR / "trained_model_fact_checked" / "multitask_trait_model.pt"
ENCODER_PATH = BASE_DIR / "trained_model_fact_checked" / "label_encoders.pkl"
ONT_NS = "http://straycare.org/ontology/breed#"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIT_PROPERTIES = {
    "EarShape": "hasEarShape",
    "CoatType": "hasCoatType",
    "CoatPattern": "hasCoatPattern",
    "SizeClass": "hasSizeClass",
    "SnoutLength": "hasSnoutLength",
    "TailCarriage": "hasTailCarriage",
}

TRAIT_PROMPTS = {
    "EarShape": {"FloppyEars": "a dog with floppy ears", "ErectEars": "a dog with erect upright ears", "SemiErectEars": "a dog with semi-erect ears", "ButtonEars": "a dog with small button ears"},
    "CoatType": {"ShortCoat": "a dog with a short coat", "MediumCoat": "a dog with a medium-length coat", "LongCoat": "a dog with a long coat", "WireCoat": "a dog with a wiry coat"},
    "CoatPattern": {"SolidColor": "a dog with a solid single-color coat", "Brindle": "a dog with a brindle coat pattern", "Spotted": "a dog with a spotted coat pattern", "Tricolor": "a dog with a tricolor coat", "PatchedColor": "a dog with patches of different coat colors"},
    "SizeClass": {"ToySize": "a toy-sized dog", "SmallSize": "a small-sized dog", "MediumSize": "a medium-sized dog", "LargeSize": "a large-sized dog", "GiantSize": "a giant-sized dog"},
    "SnoutLength": {"ShortSnout": "a dog with a short snout", "MediumSnout": "a dog with a medium-length snout", "LongSnout": "a dog with a long snout"},
    "TailCarriage": {"CurledTail": "a dog with a curled tail", "StraightTail": "a dog with a straight tail", "PlumeTail": "a dog with a long plume-like tail", "DockedTail": "a dog with a docked tail"},
}

class MultiTaskTraitNet(nn.Module):
    def __init__(self, input_dim, num_classes_per_trait):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleDict({t: nn.Linear(256, n) for t, n in num_classes_per_trait.items()})
    def forward(self, x):
        h = self.trunk(x)
        return {t: head(h) for t, head in self.heads.items()}

state = {}

def load_system():
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    nums = {t: len(encoders[t].classes_) for t in TRAIT_PROPERTIES}
    net = MultiTaskTraitNet(512, nums).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False
    g = Graph()
    g.parse(str(ONTOLOGY_PATH), format="turtle")
    return clip, processor, net, encoders, g

@torch.no_grad()
def neural_probs(image, clip, proc, net, enc):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    emb = clip.visual_projection(clip.vision_model(pixel_values=inputs["pixel_values"]).pooler_output)
    out = net(emb)
    return {t: {list(enc[t].classes_)[i]: float(torch.softmax(out[t][0], dim=0)[i].cpu()) for i in range(len(enc[t].classes_))} for t in TRAIT_PROPERTIES}

@torch.no_grad()
def clip_probs(image, clip, processor):
    res = {}
    for t, pm in TRAIT_PROMPTS.items():
        labels = list(pm)
        texts = [pm[x] for x in labels]
        x = processor(text=texts, images=image, return_tensors="pt", padding=True).to(DEVICE)
        p = torch.softmax(clip(**x).logits_per_image[0], dim=0)
        res[t] = {labels[i]: float(p[i].cpu()) for i in range(len(labels))}
    return res

def combine_probs(neural, clip_scores, w):
    out = {}
    for t in TRAIT_PROPERTIES:
        labels = list(neural[t])
        vals = [(1-w)*torch.log(torch.tensor(max(neural[t].get(l,1e-8),1e-8))) + w*torch.log(torch.tensor(max(clip_scores[t].get(l,1e-8),1e-8))) for l in labels]
        p = torch.softmax(torch.stack(vals), dim=0)
        out[t] = {labels[i]: float(p[i]) for i in range(len(labels))}
    return out

def rank_breeds(combined, graph, top_k=5):
    rows = [f"(:{TRAIT_PROPERTIES[t]} :{max(d, key=d.get)})" for t, d in combined.items() if d]
    if not rows:
        return []
    q = f"PREFIX : <{ONT_NS}> SELECT ?breed (COUNT(?matchedTrait) AS ?sharedTraits) WHERE {{ VALUES (?property ?trait) {{ {' '.join(rows)} }} ?breed ?property ?trait . BIND(?trait AS ?matchedTrait) }} GROUP BY ?breed ORDER BY DESC(?sharedTraits) ?breed LIMIT {top_k}"
    return [(str(r.breed).split("#")[-1], int(r.sharedTraits)) for r in graph.query(q)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    clip, proc, net, enc, g = load_system()
    state.update(dict(clip=clip, processor=proc, net=net, encoders=enc, graph=g))
    print(f"[HyperID] loaded on {DEVICE}")
    yield
    state.clear()

app = FastAPI(title="HyperID-KE Inference API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"name": "HyperID-KE Inference API", "device": str(DEVICE), "endpoints": ["/health", "/predict", "/docs"]}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "loaded": "clip" in state}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    clip_weight: float = Query(0.65, ge=0, le=1, description="CLIP weight 0-1"),
    top_k: int = Query(5, ge=1, le=20),
):
    if "clip" not in state:
        clip, proc, net, enc, g = load_system()
        state.update(dict(clip=clip, processor=proc, net=net, encoders=enc, graph=g))
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    clip, proc, net, enc, g = state["clip"], state["processor"], state["net"], state["encoders"], state["graph"]
    neural = neural_probs(image, clip, proc, net, enc)
    clip_sc = clip_probs(image, clip, proc)
    combined = combine_probs(neural, clip_sc, clip_weight)
    ranked = rank_breeds(combined, g, top_k)
    profile = {t: max(d, key=d.get) for t, d in combined.items()}
    confidence = {t: max(d.values()) for t, d in combined.items()}
    return {
        "trait_profile": profile,
        "confidence": confidence,
        "trait_distributions": combined,
        "ranked_breeds": [{"rank": i+1, "breed": b, "shared_traits": s, "score": f"{s}/6"} for i, (b, s) in enumerate(ranked)],
        "debug": {"neural": neural, "clip": clip_sc} if False else None,
        "meta": {"clip_weight": clip_weight, "neural_weight": 1-clip_weight, "device": str(DEVICE)},
    }

@app.post("/predict/json")
async def predict_json(payload: dict):
    import base64
    b64 = payload.get("image_base64", "")
    if not b64:
        return {"error": "provide image_base64"}
    img_bytes = base64.b64decode(b64.split(",")[-1])
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    w = float(payload.get("clip_weight", 0.65))
    top_k = int(payload.get("top_k", 5))
    if "clip" not in state:
        clip, proc, net, enc, g = load_system()
        state.update(dict(clip=clip, processor=proc, net=net, encoders=enc, graph=g))
    clip, proc, net, enc, g = state["clip"], state["processor"], state["net"], state["encoders"], state["graph"]
    neural = neural_probs(image, clip, proc, net, enc)
    clip_sc = clip_probs(image, clip, proc)
    combined = combine_probs(neural, clip_sc, w)
    ranked = rank_breeds(combined, g, top_k)
    profile = {t: max(d, key=d.get) for t, d in combined.items()}
    return {"trait_profile": profile, "trait_distributions": combined, "ranked_breeds": ranked}
