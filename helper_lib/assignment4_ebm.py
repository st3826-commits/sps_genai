import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.conv1(x))
        x = torch.nn.functional.silu(self.conv2(x))
        x = torch.nn.functional.silu(self.conv3(x))
        x = torch.nn.functional.silu(self.conv4(x))
        x = self.flatten(x)
        x = torch.nn.functional.silu(self.fc1(x))
        return self.fc2(x)

def langevin_sample(
    model,
    initial_images,
    steps=60,
    step_size=10.0,
    noise_std=0.005
):
    original_mode = model.training
    parameters = list(model.parameters())
    original_flags = [p.requires_grad for p in parameters]

    try:
        model.eval()

        for parameter in parameters:
            parameter.requires_grad_(False)

        images = initial_images.detach().clone()

        with torch.enable_grad():
            for _ in range(steps):
                with torch.no_grad():
                    images = (
                        images
                        + noise_std * torch.randn_like(images)
                    ).clamp(-1.0, 1.0)

                images = images.detach().requires_grad_(True)

                energies = model(images)
                gradients = torch.autograd.grad(
                    outputs=energies.sum(),
                    inputs=images
                )[0]

                with torch.no_grad():
                    gradients = gradients.clamp(-0.03, 0.03)
                    images = (
                        images - step_size * gradients
                    ).clamp(-1.0, 1.0)

        return images.detach()

    finally:
        for parameter, flag in zip(parameters, original_flags):
            parameter.requires_grad_(flag)

        model.train(original_mode)

def train_ebm(
    model,
    data_loader,
    optimizer,
    device,
    epochs=1,
    sampling_steps=60,
    buffer_size=8192,
    alpha=0.1
):
    model.to(device)

    replay_buffer = torch.rand(
        buffer_size, 3, 32, 32, device=device
    ) * 2 - 1

    for epoch in range(epochs):
        total_loss = 0.0
        total_real = 0.0
        total_fake = 0.0
        total_images = 0

        for batch_index, (images, _) in enumerate(data_loader):
            real_images = images.to(device) * 2 - 1
            real_images = (
                real_images
                + 0.005 * torch.randn_like(real_images)
            ).clamp(-1, 1)

            batch_size = real_images.size(0)

            indices = torch.randint(
                buffer_size, (batch_size,), device=device
            )
            initial_images = replay_buffer[indices].clone()

            restart = torch.rand(
                batch_size, 1, 1, 1, device=device
            ) < 0.05

            initial_images = torch.where(
                restart,
                torch.rand_like(initial_images) * 2 - 1,
                initial_images
            )

            fake_images = langevin_sample(
                model,
                initial_images,
                steps=sampling_steps,
                step_size=10.0,
                noise_std=0.005
            )

            with torch.no_grad():
                replay_buffer[indices] = fake_images

            model.train()
            optimizer.zero_grad(set_to_none=True)

            real_energy = model(real_images)
            fake_energy = model(fake_images.detach())

            contrastive_loss = (
                real_energy.mean() - fake_energy.mean()
            )
            regularization = alpha * (
                real_energy.square().mean()
                + fake_energy.square().mean()
            )
            loss = contrastive_loss + regularization

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.1
            )
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_real += real_energy.detach().sum().item()
            total_fake += fake_energy.detach().sum().item()
            total_images += batch_size

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_index + 1}/{len(data_loader)} | "
                    f"Loss: {total_loss / total_images:.4f} | "
                    f"Real E: {total_real / total_images:.4f} | "
                    f"Fake E: {total_fake / total_images:.4f}"
                )

        print(
            f"Epoch {epoch + 1} complete | "
            f"Loss: {total_loss / total_images:.4f} | "
            f"Real E: {total_real / total_images:.4f} | "
            f"Fake E: {total_fake / total_images:.4f}"
        )

    model.eval()
    return model