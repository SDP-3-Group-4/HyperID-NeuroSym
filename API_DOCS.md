# HyperID-KE API — https://hyperid-ke.onrender.com

Base: `https://hyperid-ke.onrender.com` | Docs: `/docs` `/redoc`

## `GET /health`
```bash
curl https://hyperid-ke.onrender.com/health
# {"status":"ok","device":"cpu","loaded":true}
```

## `POST /predict` (multipart)
Query: `clip_weight` 0-1 (default 0.65), `top_k` 1-20 (default 5)
```bash
curl -X POST "https://hyperid-ke.onrender.com/predict?clip_weight=0.65&top_k=5" \
  -F "file=@dog.jpg"
```
Response:
```json
{
  "trait_profile": {"EarShape":"FloppyEars","CoatType":"ShortCoat","CoatPattern":"SolidColor","SizeClass":"MediumSize","SnoutLength":"MediumSnout","TailCarriage":"StraightTail"},
  "confidence": {"EarShape":0.92, ...},
  "trait_distributions": {"EarShape":{"FloppyEars":0.92,"ErectEars":0.03,...}, ...},
  "ranked_breeds": [{"rank":1,"breed":"LabradorRetriever","shared_traits":5,"score":"5/6"}, ...],
  "meta": {"clip_weight":0.65,"neural_weight":0.35}
}
```

## `POST /predict_json` (base64)
```bash
curl -X POST https://hyperid-ke.onrender.com/predict_json \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"data:image/jpeg;base64,/9j/...","clip_weight":0.65,"top_k":5}'
```

## Web (JS)
```js
const fd=new FormData(); fd.append("file", input.files[0]);
const r=await fetch("https://hyperid-ke.onrender.com/predict?top_k=5",{method:"POST",body:fd});
const {trait_profile, ranked_breeds, trait_distributions}=await r.json();
```

## React / Next.js
```tsx
const fd=new FormData(); fd.append("file", file);
const res=await fetch(`${process.env.NEXT_PUBLIC_HYPERID_URL}/predict`,{method:"POST",body:fd});
if(!res.ok) throw new Error(await res.text());
const data=await res.json();
```

## Notes
- CORS `*` enabled. Cold start ~40s on free tier (ping `/health` every 5m to keep warm).
- Traits: EarShape, CoatType, CoatPattern, SizeClass, SnoutLength, TailCarriage
- Errors: 400 Invalid image, 422 missing file
