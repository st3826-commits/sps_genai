from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir,
    batch_size=32,
    train=True,
    dataset_name="CIFAR10"
):
    name = dataset_name.strip().lower()

    if name == "cifar10":
        transform = transforms.ToTensor()

        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )

    elif name == "fashionmnist":
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor()
        ])

        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )

    else:
        raise ValueError(f"{dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

def get_gan_data_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

def get_diffusion_data_loader(
    data_dir,
    batch_size=32,
    train=True
):
    transform = transforms.Compose([
        transforms.CenterCrop(340),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.Flowers102(
        root=data_dir,
        split="train" if train else "val",
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

## Assignment 3:
def get_mnist_gan_data_loader(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )