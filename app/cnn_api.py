from functools import lru_cache
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from app.cnn_model import CNNClassifier
from app.image_data import get_transform, PROJECT_ROOT

router = APIRouter(tags=["Image classification"])
transform = get_transform()

@lru_cache(maxsize=1)
def load_classifier():
    model_path = PROJECT_ROOT / "models" / "cifar10_cnn.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = CNNClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model,checkpoint["classes"]

@router.post("/classify")
def classify_image(file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image, such as a JPG or PNG.",
        )
    inputs = transform(image).unsqueeze(0)

    model, classes = load_classifier()

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    class_id = predicted.item()

    return {
        "class": classes[class_id],
        "class_id": class_id,
        "confidence": round(confidence.item(), 4),
    }
