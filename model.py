import torch
import torch.nn as nn
import torch.nn.functional as F
from config import STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z

# Fixed 3x3 kernel for counting filled neighbors (center=0, surround=1/8)
_NEIGHBOR_KERNEL = torch.tensor(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]], dtype=torch.float32
).reshape(1, 1, 3, 3) / 8.0


class HeadVAE(nn.Module):
    """Stage 1: Unconditional VAE for head shape generation.

    Architecture: 256 -> 128 -> 64 -> (mu:4, logvar:4) / 4 -> 64 -> 128 -> 256
    """

    def __init__(self):
        super().__init__()
        z_dim = STAGE1_Z

        # Encoder
        self.enc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.enc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

        # Decoder
        self.dec1 = nn.Linear(z_dim, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dec2 = nn.Linear(64, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dec3 = nn.Linear(128, 256)

    def encode(self, x):
        h = torch.relu(self.bn1(self.enc1(x)))
        h = torch.relu(self.bn2(self.enc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.bn3(self.dec1(z)))
        h = torch.relu(self.bn4(self.dec2(h)))
        return torch.sigmoid(self.dec3(h))

    def forward(self, x):
        x_flat = x.view(-1, 256)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConditionalVAE(nn.Module):
    """Stages 2-4: Conditional VAE that conditions on previous stage output.

    Encoder:  256 (target) -> 64 -> (mu:3, logvar:3)
    Decoder:  (3 z + 256 condition) = 259 -> 64 -> 256
    ~34,000 parameters total per stage.
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
        self.enc1 = nn.Linear(256, 64)
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

        # Decoder — takes z concatenated with the condition (base image)
        self.dec1 = nn.Linear(z_dim + 256, 64)
        self.dec2 = nn.Linear(64, 256)

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


def neighbor_consistency_loss(output, stray_thresh=0.2, gap_thresh=0.6):
    """Differentiable loss penalizing stray pixels and gaps in the output.

    Uses conv2d with a fixed neighbor-counting kernel. Stray pixels are filled
    pixels with very few filled neighbors; gaps are empty pixels surrounded by
    filled ones.
    """
    out_2d = output.view(-1, 1, 16, 16)
    kernel = _NEIGHBOR_KERNEL.to(output.device)
    neighbor_density = F.conv2d(out_2d, kernel, padding=1)

    # Stray: filled pixel with low neighbor density
    stray = (out_2d * F.relu(stray_thresh - neighbor_density)).mean()
    # Gap: empty pixel with high neighbor density
    gap = ((1.0 - out_2d) * F.relu(neighbor_density - gap_thresh)).mean()
    return stray + gap


def score_structural_quality(img_tensor):
    """Score a batch of 16x16 images for structural quality (lower = better).

    Used for rejection sampling during generation.
    """
    with torch.no_grad():
        binary = (img_tensor > 0.5).float().view(-1, 1, 16, 16)
        kernel = _NEIGHBOR_KERNEL.to(img_tensor.device)
        neighbors = F.conv2d(binary, kernel, padding=1)

        # Count stray pixels (filled with <2 of 8 neighbors → density < 0.25)
        stray = (binary * (neighbors < 0.25).float()).sum(dim=(1, 2, 3))
        # Count gap pixels (empty with 5+ of 8 neighbors → density > 0.6)
        gap = ((1.0 - binary) * (neighbors > 0.6).float()).sum(dim=(1, 2, 3))

        # Pixel count penalty — heads should have ~20-150 filled pixels
        pixel_count = binary.sum(dim=(1, 2, 3))
        count_penalty = F.relu(20.0 - pixel_count) + F.relu(pixel_count - 150.0)

        return stray + gap + count_penalty * 0.5


def staged_loss(recon, target, base_image, mu, logvar, beta,
                new_pixel_weight=2.0, sharpness_weight=0.15,
                connectivity_weight=0.0):
    """Loss function for staged training with pixel weighting.

    New pixels (present in target but not in base) are weighted higher so the
    model focuses on learning the new layer rather than simply copying the base.
    An optional sharpening term penalizes uncertain (near 0.5) outputs.

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

    # Per-pixel BCE loss (mean reduction for stability across batch sizes)
    bce = nn.functional.binary_cross_entropy(recon_flat, target_flat, reduction='none')

    # Pixel weight mask: new pixels (in target but not in base) get higher weight
    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight

    weighted_bce = (bce * weight_mask).mean()

    # KL divergence (mean over batch)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Sharpening: penalize outputs near 0.5 to encourage binary-like values
    sharp = sharpening_loss(recon_flat)

    total = weighted_bce + beta * kld + sharpness_weight * sharp

    if connectivity_weight > 0:
        total = total + connectivity_weight * neighbor_consistency_loss(recon_flat)

    return total


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