import torch
from data import get_cifar10
from helper_lib.model import build_ebm
from helper_lib.trainer import train_ebm

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

if __name__ == "__main__":
    device = get_device()
    print("Device:", device)
    dl = get_cifar10(batch_size=256, num_workers=2)
    model = build_ebm().to(device)
    model = train_ebm(model, dl, device=device, epochs=3)
    torch.save(model.state_dict(), "ebm.pt")
    print("✅ Saved ebm.pt")
