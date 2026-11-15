import math
import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    model = model.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            z = torch.randn(
                num_samples,
                model.latent_dim,
                device=device
            )

            if hasattr(model, "generator"):
                samples = model.generator(z)
                samples = (samples + 1.0) / 2.0
                title = "GAN Generated Samples"

            elif hasattr(model, "decoder"):
                samples = model.decoder(z)
                title = "VAE Generated Samples"

            else:
                raise ValueError("Expected a VAE or GAN model")

            samples = samples.cpu().clamp(0, 1)

    finally:
        model.train(was_training)

    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2, rows * 2),
        squeeze=False
    )

    for index, ax in enumerate(axes.flat):
        ax.axis("off")

        if index >= num_samples:
            continue

        sample = samples[index]

        if sample.shape[0] == 1:
            ax.imshow(
                sample[0].numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )
        else:
            ax.imshow(sample.permute(1, 2, 0).numpy())

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return samples

def generate_samples(
    model,
    device,
    num_samples=10,
    diffusion_steps=100
):
    import math
    import torch
    import matplotlib.pyplot as plt

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=model.network.image_size
        ).cpu()

    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2, rows * 2),
        squeeze=False
    )

    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < num_samples:
            ax.imshow(samples[index].permute(1, 2, 0).numpy())

    plt.tight_layout()
    plt.show()

    return samples