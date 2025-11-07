import os, time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- DDPM 采样（predict-ε 更稳）----
def _ddpm_params(T):
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1.0 - beta
    abar  = torch.cumprod(alpha, dim=0)
    return beta, alpha, abar

def sample_ddpm(model, device="cpu", num=9, steps=400):
    model.eval()
    beta, alpha, abar = _ddpm_params(steps)
    beta, alpha, abar = beta.to(device), alpha.to(device), abar.to(device)

    x = torch.randn(num, 3, 32, 32, device=device)
    with torch.no_grad():
        for t in reversed(range(steps)):
            tt = torch.full((num,), t, device=device, dtype=torch.long)
            eps_theta = model(x, tt)
            a_t  = alpha[t]; ab_t = abar[t]; b_t = beta[t]
            mean = (x - (b_t/torch.sqrt(1 - ab_t + 1e-8)) * eps_theta) / torch.sqrt(a_t + 1e-8)
            z = torch.randn_like(x) if t > 0 else 0.0
            sigma_t = torch.sqrt(b_t)
            x = mean + sigma_t * z
    x = x.clamp(-1,1).cpu()
    return _save_grid(x, prefix="diffusion")

# ---- EBM Langevin 动力学采样 ----
def sample_ebm(ebm, device="cpu", num=9, steps=60, step_size=0.05, noise=0.01):
    ebm.eval()
    x = torch.randn(num, 3, 32, 32, device=device)  # 从噪声出发
    x.requires_grad_(True)
    for _ in range(steps):
        e = ebm(x).sum()
        grad = torch.autograd.grad(e, x, retain_graph=False, create_graph=False)[0]
        x = x - step_size * grad + noise * torch.randn_like(x)
        x = x.detach().clamp(-1,1).requires_grad_(True)
    x = x.detach().clamp(-1,1).cpu()
    return _save_grid(x, prefix="ebm")

# ---- 保存九宫格 ----
def _save_grid(x, prefix):
    # x: [-1,1] -> [0,1]
    x = (x + 1) / 2
    n = int(np.ceil(np.sqrt(x.size(0))))
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    idx = 0
    for i in range(n):
        for j in range(n):
            axes[i,j].axis("off")
            if idx < x.size(0):
                img = x[idx].permute(1,2,0).numpy()
                axes[i,j].imshow(img)
            idx += 1
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
