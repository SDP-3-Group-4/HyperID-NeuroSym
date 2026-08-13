import argparse
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from rdflib import Graph

MODEL_NAME='openai/clip-vit-base-patch32'
ONTOLOGY_PATH='breed_trait_ontology_fact_checked_v02.ttl'
MODEL_PATH='trained_model_fact_checked/multitask_trait_model.pt'
ENCODER_PATH='trained_model_fact_checked/label_encoders.pkl'
ONT_NS='http://straycare.org/ontology/breed#'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIT_PROPERTIES={'EarShape':'hasEarShape','CoatType':'hasCoatType','CoatPattern':'hasCoatPattern','SizeClass':'hasSizeClass','SnoutLength':'hasSnoutLength','TailCarriage':'hasTailCarriage'}
TRAIT_PROMPTS={
'EarShape':{'FloppyEars':'a dog with floppy ears','ErectEars':'a dog with erect upright ears','SemiErectEars':'a dog with semi-erect ears','ButtonEars':'a dog with small button ears'},
'CoatType':{'ShortCoat':'a dog with a short coat','MediumCoat':'a dog with a medium-length coat','LongCoat':'a dog with a long coat','WireCoat':'a dog with a wiry coat'},
'CoatPattern':{'SolidColor':'a dog with a solid single-color coat','Brindle':'a dog with a brindle coat pattern','Spotted':'a dog with a spotted coat pattern','Tricolor':'a dog with a tricolor coat','PatchedColor':'a dog with patches of different coat colors'},
'SizeClass':{'ToySize':'a toy-sized dog','SmallSize':'a small-sized dog','MediumSize':'a medium-sized dog','LargeSize':'a large-sized dog','GiantSize':'a giant-sized dog'},
'SnoutLength':{'ShortSnout':'a dog with a short snout','MediumSnout':'a dog with a medium-length snout','LongSnout':'a dog with a long snout'},
'TailCarriage':{'CurledTail':'a dog with a curled tail','StraightTail':'a dog with a straight tail','PlumeTail':'a dog with a long plume-like tail','DockedTail':'a dog with a docked tail'}}

class MultiTaskTraitNet(nn.Module):
    def __init__(self,input_dim,num_classes_per_trait):
        super().__init__(); self.trunk=nn.Sequential(nn.Linear(input_dim,256),nn.ReLU(),nn.Dropout(.2)); self.heads=nn.ModuleDict({t:nn.Linear(256,n) for t,n in num_classes_per_trait.items()})
    def forward(self,x):
        h=self.trunk(x); return {t:head(h) for t,head in self.heads.items()}

def load_system():
    with open(ENCODER_PATH,'rb') as f: encoders=pickle.load(f)
    nums={t:len(encoders[t].classes_) for t in TRAIT_PROPERTIES}
    net=MultiTaskTraitNet(512,nums).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE)); net.eval()
    clip=CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor=CLIPProcessor.from_pretrained(MODEL_NAME); clip.eval()
    for p in clip.parameters(): p.requires_grad=False
    g=Graph(); g.parse(ONTOLOGY_PATH,format='turtle')
    return clip,processor,net,encoders,g

@torch.no_grad()
def neural_probs(image, clip, proc, net, enc):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)

    vision_output = clip.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled = vision_output.pooler_output

    # CLIP visual projection: 768 -> 512
    emb = clip.visual_projection(pooled)

    out = net(emb)
    result = {}

    for trait in TRAIT_PROPERTIES:
        probs = torch.softmax(out[trait][0], dim=0)
        labels = list(enc[trait].classes_)

        result[trait] = {
            labels[i]: float(probs[i].cpu())
            for i in range(len(labels))
        }

    return result

@torch.no_grad()
def clip_probs(image,clip,processor):
    result={}
    for t,pm in TRAIT_PROMPTS.items():
        labels=list(pm); texts=[pm[x] for x in labels]
        x=processor(text=texts,images=image,return_tensors='pt',padding=True).to(DEVICE)
        logits=clip(**x).logits_per_image[0]; p=torch.softmax(logits,dim=0)
        result[t]={labels[i]:float(p[i].cpu()) for i in range(len(labels))}
    return result

def combine(a,b,w):
    out={}; eps=1e-8
    for t in TRAIT_PROPERTIES:
        labels=list(a[t]); vals=[]
        for lab in labels:
            s=(1-w)*torch.log(torch.tensor(max(a[t].get(lab,eps),eps)))+w*torch.log(torch.tensor(max(b[t].get(lab,eps),eps)))
            vals.append(s)
        p=torch.softmax(torch.stack(vals),dim=0); out[t]={labels[i]:float(p[i]) for i in range(len(labels))}
    return out

def rank(pred,g,topk=5):
    rows=[]
    for t,c in pred.items():
        if c:
            v=max(c,key=c.get); rows.append(f'(:{TRAIT_PROPERTIES[t]} :{v})')
    if not rows:return []
    q=f'''PREFIX : <{ONT_NS}> SELECT ?breed (COUNT(?matchedTrait) AS ?sharedTraits) WHERE {{ VALUES (?property ?trait) {{ {' '.join(rows)} }} ?breed ?property ?trait . BIND(?trait AS ?matchedTrait) }} GROUP BY ?breed ORDER BY DESC(?sharedTraits) ?breed LIMIT {topk}'''
    return [(str(r.breed).split('#')[-1],int(r.sharedTraits)) for r in g.query(q)]

def show(title,data):
    print('\n'+'='*70+'\n'+title+'\n'+'='*70)
    for t in TRAIT_PROPERTIES:
        print('\n'+t)
        for lab,p in sorted(data[t].items(),key=lambda z:z[1],reverse=True)[:3]: print(f'  {lab:<25} {p:6.2%}')

def main(path,w):
    print('Using device:',DEVICE)
    for p in (ONTOLOGY_PATH,MODEL_PATH,ENCODER_PATH):
        if not Path(p).exists(): raise FileNotFoundError('Required file not found: '+p)
    clip,proc,net,enc,g=load_system(); image=Image.open(path).convert('RGB')
    n=neural_probs(image,clip,proc,net,enc); c=clip_probs(image,clip,proc); e=combine(n,c,w); r=rank(e,g)
    show('NEURAL TRAIT MODEL',n); show('CLIP TEXT PHENOTYPE SCORER',c); show(f'COMBINED PHENOTYPE EVIDENCE (CLIP={w:.2f}, NEURAL={1-w:.2f})',e)
    print('\n'+'='*70+'\nONTOLOGY REASONING\n'+'='*70)
    for i,(breed,score) in enumerate(r,1): print(f'{i}. {breed:<35} shared traits = {score}/6')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--clip-weight',type=float,default=.65); a=p.parse_args()
    if not 0<=a.clip_weight<=1: raise ValueError('--clip-weight must be between 0 and 1')
    main(a.image,a.clip_weight)
