"""
This code is inspired from https://github.com/pi-tau/vae
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.conv(x))


class UniversalVAE(nn.Module):
    """
    Variational Autoencoder that handles variable-channel inputs
    (grayscale, RGB, multispectral) and produces fixed-dimension latent codes.
    """

    def __init__(self,
                 max_channels: int = 16,
                 latent_dim: int = 32,
                 image_size: int = 128):
        """
        Args:
            max_channels: Maximum number of input channels (zero-padded if fewer)
            latent_dim: Dimension of latent space
            image_size: Input image resolution (assumed square)
        """
        super(UniversalVAE, self).__init__()

        self.max_channels = max_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder: (max_channels, 128, 128) -> latent_dim
        self.encoder = nn.Sequential(
            # Conv block 1: 128x128 -> 64x64
            nn.Conv2d(max_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            ResidualBlock(32),

            # Conv block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),

            # Conv block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            # Conv block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
        )

        # Encoder with reduced channels and depth
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(max_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 channels
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2),
        #     ResidualBlock(32),
        #
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 channels
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #     ResidualBlock(64),
        #
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 channels
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2),
        #     ResidualBlock(128),
        # )

        # Calculate flattened size after convolutions
        self.flatten_dim = 256 * (image_size // 16) * (image_size // 16)
        # self.flatten_dim = 128 * (image_size // 8) * (image_size // 8) # adjusted for 3 blocks

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: latent_dim -> (max_channels, 128, 128)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            # Deconv block 1: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),

            # Deconv block 2: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),

            # Deconv block 3: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            # Deconv block 4: 64x64 -> 128x128
            nn.ConvTranspose2d(32, max_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     ResidualBlock(64),
        #
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     ResidualBlock(32),
        #
        #     nn.ConvTranspose2d(32, max_channels, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh()
        # )

        self.init_weights()

    #===== From results 1 =====#
    # def __init__(self,
    #              max_channels: int = 16,
    #              latent_dim: int = 32,
    #              image_size: int = 128):
    #     """
    #     Args:
    #         max_channels: Maximum number of input channels (zero-padded if fewer)
    #         latent_dim: Dimension of latent space
    #         image_size: Input image resolution (assumed square)
    #     """
    #     super(UniversalVAE, self).__init__()
    #
    #     self.max_channels = max_channels
    #     self.latent_dim = latent_dim
    #     self.image_size = image_size
    #
    #     # Encoder: (max_channels, 128, 128) -> latent_dim
    #     self.encoder = nn.Sequential(
    #         # Conv block 1: 128x128 -> 64x64
    #         nn.Conv2d(max_channels, 64, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.LeakyReLU(0.2),
    #
    #         # Conv block 2: 64x64 -> 32x32
    #         nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.LeakyReLU(0.2),
    #
    #         # Conv block 3: 32x32 -> 16x16
    #         nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.LeakyReLU(0.2),
    #
    #         # Conv block 4: 16x16 -> 8x8
    #         nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(512),
    #         nn.LeakyReLU(0.2),
    #     )
    #
    #     # Calculate flattened size after convolutions
    #     self.flatten_dim = 512 * (image_size // 16) * (image_size // 16)
    #
    #     # Latent space projections
    #     self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
    #     self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    #
    #     # Decoder: latent_dim -> (max_channels, 128, 128)
    #     self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
    #
    #     self.decoder = nn.Sequential(
    #         # Deconv block 1: 8x8 -> 16x16
    #         nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(),
    #
    #         # Deconv block 2: 16x16 -> 32x32
    #         nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #
    #         # Deconv block 3: 32x32 -> 64x64
    #         nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #
    #         # Deconv block 4: 64x64 -> 128x128
    #         nn.ConvTranspose2d(64, max_channels, kernel_size=4, stride=2, padding=1),
    #         nn.Tanh()  # Output in [-1, 1]
    #     )
    #
    #     self.init_weights()

    def init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input images (batch_size, max_channels, H, W)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Reconstructed images (batch_size, max_channels, H, W)
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, self.image_size // 16, self.image_size // 16)
        # h = h.view(h.size(0), 128, self.image_size // 8, self.image_size // 8)  # Updated to match shallower encoder

        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            recon_x: Reconstructed images
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Extract latent representation (for fingerprinting).

        Args:
            x: Input images
            sample: If True, sample from q(z|x). If False, return mean.

        Returns:
            Latent codes (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        if sample:
            return self.reparameterize(mu, logvar)
        else:
            return mu  # Use mean for deterministic encoding


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    VAE loss made up of Reconstruction loss and KL divergence.

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (beta-VAE for disentanglement)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return {'total': recon_loss + beta * kl_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'beta': beta}
