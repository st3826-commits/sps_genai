import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.cnn_model import CNNClassifier
from app.image_data import PROJECT_ROOT, load_datasets


def main():
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    train_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    history = []

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "cifar10_cnn.pt"

    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        average_loss = total_loss / total
        train_accuracy = correct / total

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "train_accuracy": train_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} finished | "
            f"Average loss: {average_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Time: {time.time() - start_time:.1f}s",
            flush=True,
        )

        torch.save(
            {
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "classes": train_dataset.classes,
                "epoch": epoch + 1,
                "image_size": 64,
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
            model_path,
        )

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total

    metrics_path = model_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": 64,
                "learning_rate": 0.001,
                "history": history,
                "test_accuracy": test_accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nTest accuracy: {test_accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()