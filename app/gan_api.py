from functools import lru_cache
from io import BytesIO
from pathlib import Path
import math

import torch
from fastapi import APIRouter, Query, Response
from PIL import Image

from helper_lib.assignment3_gan import MNISTGAN


router = APIRouter(tags=["GAN"])


@lru_cache(maxsize=1)
def load_gan():
    checkpoint_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "assignment3_mnist_gan.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = MNISTGAN(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@router.get(
    "/generate_with_gan",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def generate_with_gan(
    num_samples: int = Query(default=16, ge=1, le=64),
    seed: int = Query(default=42, ge=0, le=2147483647),
):
    model = load_gan()

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(seed)

    noise = torch.randn(
        num_samples,
        model.latent_dim,
        generator=random_generator,
    )

    with torch.inference_mode():
        images = model.generator(noise)
        images = ((images + 1) / 2).clamp(0, 1)
        pixels = (images[:, 0] * 255).round().to(torch.uint8)

    columns = min(4, num_samples)
    rows = math.ceil(num_samples / columns)
    tile_size = 112
    gap = 8

    canvas = Image.new(
        "L",
        (
            columns * tile_size + (columns + 1) * gap,
            rows * tile_size + (rows + 1) * gap,
        ),
        color=255,
    )

    for index in range(num_samples):
        tile = Image.fromarray(pixels[index].numpy())
        tile = tile.resize(
            (tile_size, tile_size),
            resample=Image.Resampling.NEAREST,
        )

        x = gap + (index % columns) * (tile_size + gap)
        y = gap + (index // columns) * (tile_size + gap)
        canvas.paste(tile, (x, y))

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )