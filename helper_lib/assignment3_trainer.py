import torch


def train_mnist_gan(
    gan,
    data_loader,
    generator_optimizer,
    discriminator_optimizer,
    device,
    epochs=1,
):
    criterion = torch.nn.BCELoss()
    gan.to(device)
    gan.train()

    history = []

    for epoch in range(epochs):
        generator_loss_sum = 0.0
        discriminator_loss_sum = 0.0
        total_images = 0

        for batch_index, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # Update the discriminator.
            discriminator_optimizer.zero_grad(set_to_none=True)

            noise = torch.randn(
                batch_size, gan.latent_dim, device=device
            )
            fake_images = gan.generator(noise)

            real_predictions = gan.discriminator(real_images)
            fake_predictions = gan.discriminator(fake_images.detach())

            real_loss = criterion(real_predictions, real_targets)
            fake_loss = criterion(fake_predictions, fake_targets)
            discriminator_loss = (real_loss + fake_loss) / 2

            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Update the generator.
            generator_optimizer.zero_grad(set_to_none=True)
            gan.discriminator.requires_grad_(False)

            try:
                fake_predictions = gan.discriminator(fake_images)
                generator_loss = criterion(
                    fake_predictions, real_targets
                )
                generator_loss.backward()
                generator_optimizer.step()
            finally:
                gan.discriminator.requires_grad_(True)

            generator_loss_sum += generator_loss.item() * batch_size
            discriminator_loss_sum += (
                discriminator_loss.item() * batch_size
            )
            total_images += batch_size

            if (
                (batch_index + 1) % 200 == 0
                or batch_index + 1 == len(data_loader)
            ):
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_index + 1}/{len(data_loader)} | "
                    f"D loss: {discriminator_loss.item():.4f} | "
                    f"G loss: {generator_loss.item():.4f}",
                    flush=True,
                )

        metrics = {
            "generator_loss": generator_loss_sum / total_images,
            "discriminator_loss": (
                discriminator_loss_sum / total_images
            ),
        }
        history.append(metrics)

        print(
            f"Epoch {epoch + 1}/{epochs} complete | "
            f"Mean D loss: {metrics['discriminator_loss']:.4f} | "
            f"Mean G loss: {metrics['generator_loss']:.4f}",
            flush=True,
        )

    return history