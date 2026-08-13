import argparse
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME='openai/clip-vit-base-patch32'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BREEDS=['Beagle','Boxer','Chihuahua','Pug','Samoyed','Shiba Inu','Great Pyrenees','German Shorthaired Pointer','Staffordshire Bull Terrier','Yorkshire Terrier','Pomeranian','Basset Hound','Saint Bernard']

def load_model():
    model=CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor=CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters(): p.requires_grad=False
    return model,processor

@torch.no_grad()
def predict(image,model,processor,top_k=5):
    inputs=processor(text=BREEDS,images=image,return_tensors='pt',padding=True).to(DEVICE)
    probs=model(**inputs).logits_per_image[0].softmax(dim=0)
    vals,idx=probs.topk(min(top_k,len(BREEDS)))
    return [(BREEDS[int(i)],float(v)) for v,i in zip(vals.cpu(),idx.cpu())]

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--top-k',type=int,default=5); a=p.parse_args()
    print('Using device:',DEVICE)
    m,pr=load_model(); image=Image.open(a.image).convert('RGB')
    print('\n=== Plain CLIP zero-shot baseline ===')
    for r,(b,s) in enumerate(predict(image,m,pr,a.top_k),1): print(f'{r}. {b:<32} {s:.2%}')
