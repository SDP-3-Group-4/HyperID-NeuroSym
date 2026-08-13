import argparse
from collections import defaultdict
from torchvision.datasets import OxfordIIITPet
from hyperid_clip_baseline import load_model,predict
from hyperid_reasoning import load_system,predict_traits, rank_breeds_by_trait_overlap

MAP={'beagle':'Beagle','boxer':'Boxer','chihuahua':'Chihuahua','pug':'Pug','samoyed':'Samoyed','shiba_inu':'ShibaInu','great_pyrenees':'GreatPyrenees','german_shorthaired':'GermanShorthairedPointer','staffordshire_bull_terrier':'StaffordshireBullTerrier','yorkshire_terrier':'YorkshireTerrier','pomeranian':'Pomeranian','basset_hound':'BassetHound','saint_bernard':'SaintBernard'}

def main(limit):
    ds=OxfordIIITPet(root='./data',download=True,target_types='category'); bm,bp=load_model(); clip,proc,net,enc,g=load_system(); counts=defaultdict(int); hit={k:0 for k in ['baseline_top1','baseline_top3','grounded_top1','grounded_top3']}
    for image,idx in ds:
        raw=ds.classes[idx].lower().replace(' ','_')
        if raw not in MAP or (limit and counts[MAP[raw]]>=limit): continue
        target=MAP[raw]; counts[target]+=1; image=image.convert('RGB')
        b=predict(image,bm,bp,3); gr=rank_breeds_by_trait_overlap(predict_traits(image,clip,proc,net,enc),g,3)
        bpicks=[x[0] for x in b]; gpicks=[x[0] for x in gr]
        for k,name,picks in [(1,'baseline_top1',bpicks),(3,'baseline_top3',bpicks),(1,'grounded_top1',gpicks),(3,'grounded_top3',gpicks)]: hit[name]+=target in picks[:k]
    total=sum(counts.values()); print('\n=== OXFORD SANITY RESULTS ==='); print('Images:',total)
    for k,v in hit.items(): print(f'{k:<20}: {v/total:.2%}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--limit-per-breed',type=int,default=10); a=p.parse_args(); main(a.limit_per_breed)
