from fastapi import FastAPI
from pydantic import BaseModel
import torch
from helper_lib.model import build_diffusion_unet, build_ebm
from helper_lib.generator import sample_ddpm, sample_ebm
import os

class GenReq(BaseModel):
    num: int = 9
    steps: int = 200

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

app = FastAPI(title="Module 6/8 API — Diffusion + EBM")
device = get_device()

# 模型（惰性加载/训练后加载权重）
diffusion = build_diffusion_unet().to(device)
ebm = build_ebm().to(device)

if os.path.exists("diffusion.pt"):
    diffusion.load_state_dict(torch.load("diffusion.pt", map_location=device))
if os.path.exists("ebm.pt"):
    ebm.load_state_dict(torch.load("ebm.pt", map_location=device))

@app.get("/health")
def health():
    return {"ok": True, "device": device}

@app.post("/generate/diffusion")
def generate_diffusion(req: GenReq):
    try:
        os.makedirs("outputs", exist_ok=True)
        path = sample_ddpm(diffusion, device=device, num=req.num, steps=req.steps)
        return {"status": "ok", "output": path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate/ebm")
def generate_ebm(req: GenReq):
    try:
        os.makedirs("outputs", exist_ok=True)
        steps = max(10, min(req.steps, 200))
        path = sample_ebm(ebm, device=device, num=req.num, steps=steps)
        return {"status": "ok", "output": path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
