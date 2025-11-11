APAN 5560 – Module 6 & Module 8 Assignment

This project implements two generative models trained on CIFAR-10:

Diffusion Model (DDPM)

Energy-Based Model (EBM) with Langevin Dynamics Sampling

Both models are exposed through a FastAPI service, and can be run locally or via Docker.

Environment Setup (Local)
```bash
cd ass4-genai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Start API (Local)
```bash
uvicorn app:app --reload
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Generate Images (Local)
Diffusion:
```bash
curl -X POST http://127.0.0.1:8000/generate/diffusion \
  -H "Content-Type: application/json" \
  -d '{"num":9, "steps":200}'
```
EBM:
```bash
curl -X POST http://127.0.0.1:8000/generate/ebm \
  -H "Content-Type: application/json" \
  -d '{"num":9, "steps":60}'
```

Generated files are saved in:
outputs/

Run Using Docker (Instructor Reproduction)
Build:
```bash
docker build -t ass4-genai:latest .
```
Run container:
```bash
docker run --rm -p 8000:8000 ass4-genai:latest
```

Test API inside Docker:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate/diffusion -H "Content-Type: application/json" -d '{"num":9,"steps":100}'
curl -X POST http://127.0.0.1:8000/generate/ebm -H "Content-Type: application/json" -d '{"num":9,"steps":40}'
```
