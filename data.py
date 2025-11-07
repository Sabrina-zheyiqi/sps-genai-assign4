from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(batch_size=128, num_workers=2):
    tf = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # -> [-1,1]
    ])
    train = datasets.CIFAR10(root="~/.torch/data", train=True, download=True, transform=tf)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
