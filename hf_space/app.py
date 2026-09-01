import pickle, os, base64
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from rdflib import Graph

BASE_DIR = Path(__file__).parent
MODEL_NAME = "openai/clip-vit-base-patch32"
ONTOLOGY_PATH = BASE_DIR / "breed_trait_ontology_fact_checked_v02.ttl"
MODEL_PATH = BASE_DIR / "trained_model_fact_checked" / "multitask_trait_model.pt"
ENCODER_PATH = BASE_DIR / "trained_model_fact_checked" / "label_encoders.pkl"
ONT_NS = "http://straycare.org/ontology/breed#"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIT_PROPERTIES = {"EarShape":"hasEarShape","CoatType":"hasCoatType","CoatPattern":"hasCoatPattern","SizeClass":"hasSizeClass","SnoutLength":"hasSnoutLength","TailCarriage":"hasTailCarriage"}


class MultiTaskTraitNet(nn.Module):
    def __init__(self, input_dim, num_classes_per_trait):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim,256),nn.ReLU(),nn.Dropout(0.2))
        self.heads = nn.ModuleDict({t: nn.Linear(256,n) for t,n in num_classes_per_trait.items()})
    def forward(self,x):
        return {t: head(self.trunk(x)) for t,head in self.heads.items()}

state={}

def load_system():
    with open(ENCODER_PATH,"rb") as f: encoders=pickle.load(f)
    nums={t: len(encoders[t].classes_) for t in TRAIT_PROPERTIES}
    net=MultiTaskTraitNet(512,nums).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    net.eval()
    clip=CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    processor=CLIPImageProcessor.from_pretrained(MODEL_NAME)
    clip.eval()
    for p in clip.parameters(): p.requires_grad=False
    g=Graph(); g.parse(str(ONTOLOGY_PATH),format="turtle")
    return clip,processor,net,encoders,g

def ensure_loaded():
    if "clip" not in state:
        clip,proc,net,enc,g=load_system()
        state.update(clip=clip,processor=proc,net=net,encoders=enc,graph=g)

@torch.no_grad()
def neural_probs(image,clip,proc,net,enc):
    inputs=proc(images=image,return_tensors="pt").to(DEVICE)
    emb=clip(**inputs).image_embeds
    out=net(emb)
    return {t:{list(enc[t].classes_)[i]:float(torch.softmax(out[t][0],dim=0)[i].cpu()) for i in range(len(enc[t].classes_))} for t in TRAIT_PROPERTIES}

def combine_probs(neural,clip_scores,w):
    return neural

def rank_breeds(combined,graph,top_k=5):
    rows=[f"(:{TRAIT_PROPERTIES[t]} :{max(d,key=d.get)})" for t,d in combined.items() if d]
    if not rows: return []
    q=f"PREFIX : <{ONT_NS}> SELECT ?breed (COUNT(?matchedTrait) AS ?sharedTraits) WHERE {{ VALUES (?property ?trait) {{ {' '.join(rows)} }} ?breed ?property ?trait . BIND(?trait AS ?matchedTrait) }} GROUP BY ?breed ORDER BY DESC(?sharedTraits) ?breed LIMIT {int(top_k)}"
    return [(str(r.breed).split("#")[-1],int(r.sharedTraits)) for r in graph.query(q)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_loaded()
        print(f"[HyperID] loaded on {DEVICE}")
    except Exception as e:
        print(f"[HyperID] lazy load: {e}")
    yield

app=FastAPI(title="HyperID-KE Inference API",version="1.0.0",lifespan=lifespan, docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.get("/")
def root():
    return {"name":"HyperID-KE","status":"ok","device":str(DEVICE),"loaded":"clip" in state,"endpoints":{ "health":"GET /health","predict":"POST /predict (multipart file)","predict_json":"POST /predict_json (base64)","docs":"/docs"}}

@app.get("/health")
def health():
    return {"status":"ok","device":str(DEVICE),"loaded":"clip" in state}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ensure_loaded()
    try:
        image=Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400,"Invalid image file")
    clip,proc,net,enc,g=state["clip"],state["processor"],state["net"],state["encoders"],state["graph"]
    neural=neural_probs(image,clip,proc,net,enc)
    clip_sc = {} # removed clip zero-shot
    combined=combine_probs(neural,clip_sc,0)
    ranked=rank_breeds(combined,g,5)
    profile={t:max(d,key=d.get) for t,d in combined.items()}
    confidence={t:max(d.values()) for t,d in combined.items()}
    return {"trait_profile":profile,"confidence":confidence,"trait_distributions":combined,"ranked_breeds":[{"rank":i+1,"breed":b,"shared_traits":s,"score":f"{s}/6"} for i,(b,s) in enumerate(ranked)],"meta":{"clip_weight":0,"neural_weight":1}}

@app.post("/predict_json")
async def predict_json(payload: dict):
    ensure_loaded()
    b64=payload.get("image_base64","")
    if not b64: raise HTTPException(400,"provide image_base64")
    try:
        img_bytes=base64.b64decode(b64.split(",")[-1])
        image=Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400,"Invalid base64 image")
    w=0.0; top_k=int(payload.get("top_k",5))
    clip,proc,net,enc,g=state["clip"],state["processor"],state["net"],state["encoders"],state["graph"]
    neural=neural_probs(image,clip,proc,net,enc)
    clip_sc = {}
    combined=combine_probs(neural,clip_sc,w)
    ranked=rank_breeds(combined,g,top_k)
    profile={t:max(d,key=d.get) for t,d in combined.items()}
    return {"trait_profile":profile,"trait_distributions":combined,"ranked_breeds":[{"breed":b,"shared_traits":s} for b,s in ranked]}
