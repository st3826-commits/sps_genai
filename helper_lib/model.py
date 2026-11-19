import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.convtrans1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.convtrans2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.convtrans3 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 128, 4, 4)
        x = torch.relu(self.convtrans1(x))
        x = torch.relu(self.convtrans2(x))
        return torch.sigmoid(self.convtrans3(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128, 256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                256, 512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                512, 1, kernel_size=4,
                stride=1, padding=0, bias=False
            ),
            nn.Flatten()
        )

    def forward(self, x):
        return self.network(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 512, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                512, 256, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.reshape(z.size(0), self.latent_dim, 1, 1)
        return self.network(z)


class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)

def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95
):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)

    diffusion_angles = (
        start_angle
        + diffusion_times * (end_angle - start_angle)
    )

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


def get_model(model_name):
    name = model_name.strip().lower()

    if name == "fcnn":
        return FCNN()
    elif name == "cnn":
        return CNN()
    elif name == "enhancedcnn":
        return EnhancedCNN()
    elif name == "vae":
        return VAE()
    elif name == "gan":
        return GAN()
    elif name == "diffusion":
        network = UNet(image_size=64, num_channels=3)
        return DiffusionModel(
            model=network,
            schedule_fn=offset_cosine_diffusion_schedule
        )

    raise ValueError(f"Unknown model name: {model_name}")

import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)
def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """
    diffusion_times: Tensor of shape (T,) with values in [0, 1)
    Returns:
        noise_rates: Tensor of shape (T,)
        signal_rates: Tensor of shape (T,)
    """
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        # x = self.norm(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size like TF reference
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)
import copy

class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        # Override to() to ensure both networks move to the same device
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Use EMA network for inference, main network for training
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            # Debug generation process
            if step % max(1, diffusion_steps // 4) == 0:  # Print 4 times during generation
                print(f"Generation Step {step}/{diffusion_steps}: t={1-step*step_size:.3f}")
                print(f"  Current images std: {current_images.std().item():.4f}")
                print(f"  Pred images std: {pred_images.std().item():.4f}")
                print(f"  Signal rate: {signal_rates.mean().item():.4f}, Noise rate: {noise_rates.mean().item():.4f}")

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        # Debug prints
        if torch.rand(1).item() < 0.01:  # Print more frequently to see output
            print(f"Debug - Loss: {loss.item():.4f}, Noise std: {noises.std().item():.4f}, Pred std: {pred_noises.std().item():.4f}")
            print(f"Signal rates range: {signal_rates.min().item():.4f}-{signal_rates.max().item():.4f}")
            print(f"Noise rates range: {noise_rates.min().item():.4f}-{noise_rates.max().item():.4f}")

        optimizer.zero_grad()
        loss.backward()

        # Check for gradient issues
        if torch.rand(1).item() < 0.01:
            total_norm = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Gradient norm: {total_norm:.4f}")

        optimizer.step()

        with torch.no_grad():
            # Debug EMA update occasionally
            if torch.rand(1).item() < 0.001:
                param_diff = 0
                for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                    param_diff += (ema_param - param).abs().mean().item()
                print(f"EMA Update Debug - Avg param difference: {param_diff:.6f}")

            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        generated_images = self.generate(
            num_images=num_rows * num_cols,
            image_size=self.network.image_size,
            diffusion_steps=20).cpu()

        grid = make_grid(generated_images, nrow=num_cols)
        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()
