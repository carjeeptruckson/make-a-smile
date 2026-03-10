import torch
import torch.nn as nn
from config import LATENT_DIM

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)
        
        self.fc3 = nn.Linear(LATENT_DIM, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 256)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h4 = torch.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 256))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, weights=None, kl_weight=1.5):
    # BCE is calculated per pixel
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 256), reduction='none')
    BCE = torch.sum(BCE, dim=1) # Sum per image
    
    if weights is not None:
        BCE = BCE * weights # Multiply the loss of specific images by their 'Severity'
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return torch.sum(BCE) + (kl_weight * KLD)

def add_noise(img_tensor, noise_factor=0.1):
    # Denoising Augmentation: Randomly flips 10% of pixels during training
    noise = torch.rand_like(img_tensor)
    noisy_img = torch.where(noise < noise_factor, 1.0 - img_tensor, img_tensor)
    return noisy_img