from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def load_datasets():
    transform = get_transform()

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = load_datasets()

    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Classes:", train_dataset.classes)
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    assert images.shape == (64, 3, 64, 64)
    assert labels.shape == (64,)
    print("Data check passed!")