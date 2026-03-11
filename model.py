import torch
import torch.nn as nn
from config import STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z


class HeadVAE(nn.Module):
    """Stage 1: Unconditional VAE for head shape generation.

    Architecture: 256 -> 32 -> 16 -> (mu:4, logvar:4) / 4 -> 16 -> 32 -> 256
    ~18,000 parameters total.
    """

    def __init__(self):
        super().__init__()
        z_dim = STAGE1_Z

        # Encoder
        self.enc1 = nn.Linear(256, 32)
        self.enc2 = nn.Linear(32, 16)
        self.fc_mu = nn.Linear(16, z_dim)
        self.fc_logvar = nn.Linear(16, z_dim)

        # Decoder
        self.dec1 = nn.Linear(z_dim, 16)
        self.dec2 = nn.Linear(16, 32)
        self.dec3 = nn.Linear(32, 256)

    def encode(self, x):
        h = torch.relu(self.enc1(x))
        h = torch.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec1(z))
        h = torch.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))

    def forward(self, x):
        x_flat = x.view(-1, 256)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConditionalVAE(nn.Module):
    """Stages 2-4: Conditional VAE that conditions on previous stage output.

    Encoder:  256 (target) -> 32 -> (mu:3, logvar:3)
    Decoder:  (3 z + 256 condition) = 259 -> 32 -> 256
    ~26,000 parameters total per stage.
    """

    STAGE_Z = {
        "stage2": STAGE2_Z,
        "stage3": STAGE3_Z,
        "stage4": STAGE4_Z,
    }

    def __init__(self, stage_name="stage2"):
        super().__init__()
        self.stage_name = stage_name
        z_dim = self.STAGE_Z.get(stage_name, 3)

        # Encoder — encodes the full target image
        self.enc1 = nn.Linear(256, 32)
        self.fc_mu = nn.Linear(32, z_dim)
        self.fc_logvar = nn.Linear(32, z_dim)

        # Decoder — takes z concatenated with the condition (base image)
        self.dec1 = nn.Linear(z_dim + 256, 32)
        self.dec2 = nn.Linear(32, 256)

    def encode(self, x):
        h = torch.relu(self.enc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        combined = torch.cat([z, condition], dim=1)
        h = torch.relu(self.dec1(combined))
        return torch.sigmoid(self.dec2(h))

    def forward(self, target, condition):
        target_flat = target.view(-1, 256)
        condition_flat = condition.view(-1, 256)
        mu, logvar = self.encode(target_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition_flat)
        return recon, mu, logvar


def sharpening_loss(output):
    """Penalizes outputs near 0.5, encouraging crisp binary values.

    This computes the negative binary entropy of each pixel. Pixels at 0.5
    (maximum uncertainty) get the highest penalty; pixels near 0 or 1 get
    near-zero penalty. This prevents the "thick wall" problem where the
    model produces soft/smeared edges.

    Returns:
        Mean sharpening penalty (scalar, higher = more uncertain outputs)
    """
    eps = 1e-7
    clamped = output.clamp(eps, 1.0 - eps)
    # Binary entropy: H = -(p*log(p) + (1-p)*log(1-p))
    # This is maximized at p=0.5 (= 0.693) and 0 at p=0 or p=1
    entropy = -(clamped * torch.log(clamped) + (1 - clamped) * torch.log(1 - clamped))
    return entropy.mean()


def staged_loss(recon, target, base_image, mu, logvar, beta,
                new_pixel_weight=2.0, sharpness_weight=0.5):
    """Loss function for staged training with pixel weighting and sharpening.

    New pixels (present in target but not in base) are weighted higher so the
    model focuses on learning the new layer rather than simply copying the base.
    An additional sharpening term penalizes uncertain (near 0.5) outputs to
    produce crisp 1-pixel features.

    Args:
        recon: Reconstructed image tensor (batch, 256)
        target: Target image tensor (batch, 256)
        base_image: Base/condition image tensor (batch, 256)
        mu: Latent mean (batch, z_dim)
        logvar: Latent log-variance (batch, z_dim)
        beta: KL divergence weight (from annealing schedule)
        new_pixel_weight: Extra weight for newly drawn pixels (default 2.0)
        sharpness_weight: How strongly to penalize soft/uncertain pixels

    Returns:
        Total loss (scalar)
    """
    target_flat = target.view(-1, 256)
    base_flat = base_image.view(-1, 256)
    recon_flat = recon.view(-1, 256)

    # Per-pixel BCE loss
    bce = nn.functional.binary_cross_entropy(recon_flat, target_flat, reduction='none')

    # Pixel weight mask: new pixels (in target but not in base) get higher weight
    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight

    weighted_bce = (bce * weight_mask).sum()

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Sharpening: penalize outputs near 0.5 to encourage binary-like values
    sharp = sharpening_loss(recon_flat) * recon_flat.shape[0] * 256

    return weighted_bce + beta * kld + sharpness_weight * sharp


def kl_beta_schedule(epoch, warmup_start=100, warmup_end=400, final_beta=0.8):
    """Linear KL annealing schedule.

    Returns:
        0.0 for epochs < warmup_start
        Linear ramp from 0 to final_beta between warmup_start and warmup_end
        final_beta for epochs >= warmup_end
    """
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return final_beta * progress
    else:
        return final_beta


def add_noise(img_tensor, noise_factor=0.03):
    """Denoising augmentation: randomly flips a fraction of pixels during training.

    Default 3% — flips ~8 pixels per 16×16 image. Higher values (e.g. 10%)
    cause wall-thickening artifacts on thin 1-pixel features.
    """
    noise = torch.rand_like(img_tensor)
    noisy_img = torch.where(noise < noise_factor, 1.0 - img_tensor, img_tensor)
    return noisy_img