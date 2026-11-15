from functools import lru_cache
from io import BytesIO
from pathlib import Path
from helper_lib.assignment4_ebm import EnergyModel, langevin_sample

import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import (
    UNet,
    DiffusionModel,
    offset_cosine_diffusion_schedule
)


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def images_to_png(images):
    images = images.detach().cpu().clamp(0, 1)

    grid = make_grid(
        images,
        nrow=min(4, len(images)),
        padding=2
    )

    pixels = (
        grid.mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )

    image = Image.fromarray(pixels)
    image = image.resize(
        (image.width * 4, image.height * 4),
        resample=Image.Resampling.NEAREST
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@lru_cache(maxsize=1)
def load_diffusion():
    checkpoint = torch.load(
        MODEL_DIR / "assignment4_cifar10_diffusion.pth",
        map_location="cpu",
        weights_only=True
    )

    model = DiffusionModel(
        model=UNet(image_size=32, num_channels=3),
        schedule_fn=offset_cosine_diffusion_schedule
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_normalizer(
        checkpoint["normalizer_mean"],
        checkpoint["normalizer_std"]
    )
    model.eval()
    return model


def generate_diffusion_png(num_samples, seed, diffusion_steps):
    model = load_diffusion()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = torch.randn(
        num_samples, 3, 32, 32,
        generator=generator
    )

    with torch.no_grad():
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=32,
            initial_noise=initial_noise
        )

    return images_to_png(samples)

@lru_cache(maxsize=1)
def load_ebm():
    checkpoint = torch.load(
        MODEL_DIR / "assignment4_cifar10_ebm.pth",
        map_location="cpu",
        weights_only=True
    )

    model = EnergyModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_ebm_png(num_samples, seed, sampling_steps):
    model = load_ebm()


    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        initial_images = torch.rand(
            num_samples, 3, 32, 32
        ) * 2 - 1

        samples = langevin_sample(
            model=model,
            initial_images=initial_images,
            steps=sampling_steps,
            step_size=10.0,
            noise_std=0.005
        )

    return images_to_png((samples + 1) / 2)