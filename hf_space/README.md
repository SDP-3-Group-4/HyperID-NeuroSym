---
title: HyperID-KE
emoji: 🐾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
license: mit
short_description: Neuro-symbolic breed-ancestry inference (CLIP + TraitNet + OWL)
---

# HyperID-KE Inference API

Neuro-symbolic pipeline: CLIP ViT-B/32 + MultiTask TraitNet → log-space ensemble → OWL/SPARQL ontology reasoning.

## Endpoints
- `GET /` `GET /health` `GET /docs`
- `POST /predict` multipart `file` + `?clip_weight=0.65&top_k=5`
- `POST /predict_json` JSON `{"image_base64":"...","clip_weight":0.65}`

## Local test
```bash
curl -X POST "http://localhost:7860/predict?clip_weight=0.65" -F "file=@dog.jpg"
```
