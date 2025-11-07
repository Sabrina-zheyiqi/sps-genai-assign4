import torch, torch.nn as nn
from torch import optim

def ddpm_betas(T=400, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def train_diffusion(model, loader, device="cpu", epochs=5, T=400, lr=3e-4):
    betas = ddpm_betas(T).to(device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)  # \bar{α}_t
    opt = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    step = 0
    for ep in range(epochs):
        for x,_ in loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            a = abar[t].view(-1,1,1,1)
            eps = torch.randn_like(x)
            x_t = torch.sqrt(a)*x + torch.sqrt(1-a)*eps   # q(x_t|x_0)
            eps_pred = model(x_t, t)
            loss = mse(eps_pred, eps)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f"[diffusion] ep={ep+1} step={step} loss={loss.item():.4f}")
    return model

def train_ebm(model, loader, device="cpu", epochs=3, lr=1e-4, sigma=0.3, k_neg=1):
    # 对比散度风格：正样本 x+ 来自数据，负样本 x- 来自高斯噪声或轻度扰动
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        for x,_ in loader:
            x = x.to(device)
            with torch.no_grad():
                x_neg = x + sigma*torch.randn_like(x)      # 简单负样本
                x_neg = x_neg.clamp(-1,1)
            e_pos = model(x)
            e_neg = model(x_neg)
            # 让真样本能量低、负样本能量高
            loss = (e_pos.mean() - e_neg.mean())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"[ebm] ep={ep+1} loss={loss.item():.4f}")
    return model
