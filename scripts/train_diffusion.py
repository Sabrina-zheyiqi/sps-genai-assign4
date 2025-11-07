import torch
from data import get_cifar10
from helper_lib.model import build_diffusion_unet
from helper_lib.trainer import train_diffusion

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

if __name__ == "__main__":
    device = get_device()
    print("Device:", device)
    dl = get_cifar10(batch_size=256, num_workers=2)
    model = build_diffusion_unet().to(device)
    model = train_diffusion(model, dl, device=device, epochs=5, T=400, lr=3e-4)
    torch.save(model.state_dict(), "diffusion.pt")
    print("✅ Saved diffusion.pt")
