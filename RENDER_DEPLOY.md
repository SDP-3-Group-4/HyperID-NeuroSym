# Deploy to Render (Free)

1. Push to GitHub
```powershell
cd hf_space
git init; git add .; git commit -m "hyperid render"
gh repo create hyperid-ke --public --source=. --push
# or manually create repo and: git remote add origin https://github.com/YOU/hyperid-ke.git; git push -u origin main
```

2. Render Dashboard
- https://dashboard.render.com → New + → Web Service
- Connect `hyperid-ke` repo → Root Directory: `hf_space` (or repo root if you pushed hf_space contents at root)
- Runtime: Docker → Free plan → Health check: `/health`
- Deploy → URL like `https://hyperid-ke.onrender.com`

3. Test
```bash
curl https://hyperid-ke.onrender.com/health
curl -X POST "https://hyperid-ke.onrender.com/predict?top_k=5" -F "file=@dog.jpg"
```

4. Web app
```js
const fd = new FormData(); fd.append("file", file);
const r = await fetch("https://hyperid-ke.onrender.com/predict", {method:"POST", body: fd});
const {trait_profile, ranked_breeds} = await r.json();
```

Note: Free instance spins down after 15m, first request cold-start ~30-45s (CLIP load).
Add UptimeRobot ping to /health every 5m to keep warm if needed.
