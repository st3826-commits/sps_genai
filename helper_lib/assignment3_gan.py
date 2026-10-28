import torch
from torch import nn

class MNISTGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.network = nn.Sequential(
              nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )
    def forward(self,noise):
        features = self.fc(noise)
        features = features.reshape(noise.size(0), 128, 7, 7)
        return self.network(features) 
    

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.network(images)


class MNISTGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = MNISTGenerator(latent_dim)
        self.discriminator = MNISTDiscriminator()

    def forward(self, noise):
        return self.generator(noise)