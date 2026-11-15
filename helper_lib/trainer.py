import torch
import torch.nn.functional as F

def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {avg_loss:.4f}"
        )

    return model

def vae_loss_function(reconstruction, inputs, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        inputs,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return 500.0 * reconstruction_loss + kl_loss


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, _ in data_loader:
            images = images.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar = model(images)
            loss = criterion(reconstruction, images, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss per image: {avg_loss:.4f}"
        )

    return model

def wasserstein_loss(scores, target):
    return -(scores * target).mean()


def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    n_critic=5,
    clip_value=0.01,
    max_batches=None
):
    model = model.to(device)
    generator = model.generator
    critic = model.critic

    opt_generator = optimizer["generator"]
    opt_critic = optimizer["critic"]

    for epoch in range(epochs):
        model.train()

        total_critic_loss = 0.0
        total_generator_loss = 0.0
        total_samples = 0

        for batch_index, (real_images, _) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            critic_loss_sum = 0.0

            for _ in range(n_critic):
                opt_critic.zero_grad(set_to_none=True)

                noise = torch.randn(
                    batch_size, model.latent_dim, device=device
                )

                with torch.no_grad():
                    fake_images = generator(noise)

                real_scores = critic(real_images)
                fake_scores = critic(fake_images)

                critic_loss = (
                    criterion(real_scores, 1.0)
                    + criterion(fake_scores, -1.0)
                )

                critic_loss.backward()
                opt_critic.step()

                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-clip_value, clip_value)

                critic_loss_sum += critic_loss.item()

            opt_generator.zero_grad(set_to_none=True)
            critic.requires_grad_(False)

            try:
                noise = torch.randn(
                    batch_size, model.latent_dim, device=device
                )
                fake_images = generator(noise)
                generator_loss = criterion(critic(fake_images), 1.0)

                generator_loss.backward()
                opt_generator.step()
            finally:
                critic.requires_grad_(True)

            total_critic_loss += (
                critic_loss_sum / n_critic
            ) * batch_size
            total_generator_loss += generator_loss.item() * batch_size
            total_samples += batch_size

            if (batch_index + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"| Batch {batch_index + 1} "
                    f"| Critic: {critic_loss.item():.4f} "
                    f"| Generator: {generator_loss.item():.4f}",
                    flush=True
                )

        if total_samples == 0:
            raise ValueError("No training samples were processed")

        print(
            f"Epoch {epoch + 1}/{epochs} complete "
            f"| Mean critic loss: "
            f"{total_critic_loss / total_samples:.4f} "
            f"| Mean generator loss: "
            f"{total_generator_loss / total_samples:.4f}",
            flush=True
        )

    return model

def train_diffusion(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_images = 0

        for images, _ in data_loader:
            images = images.to(device)

            loss = model.train_step(
                images,
                optimizer,
                criterion
            )

            batch_size = images.size(0)
            total_loss += loss * batch_size
            total_images += batch_size

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Noise loss: {total_loss / total_images:.4f}"
        )

    model.eval()
    return model