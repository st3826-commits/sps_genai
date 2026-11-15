from pathlib import Path
import torch


def save_model(model, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to: {filepath}")