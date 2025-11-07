# Module 6/8 — CIFAR-10 Diffusion & Energy-Based Models (API)

This repo adds **Diffusion** and **EBM** image generators to the Module-6 API.

## Train
```bash
python -m scripts.train_diffusion   # saves diffusion.pt (5 epochs, DDPM)
python -m scripts.train_ebm         # saves ebm.pt (3 epochs, Langevin-ready)
uvicorn app:app --reload
# Health: GET  /health
# Diffusion: POST /generate/diffusion {"num":9,"steps":200}
# EBM:       POST /generate/ebm        {"num":9,"steps":60}
