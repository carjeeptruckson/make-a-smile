import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z, GRID_SIZE

# Fixed 3x3 kernel for counting filled neighbors (center=0, surround=1/8)
_NEIGHBOR_KERNEL = torch.tensor(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]], dtype=torch.float32
).reshape(1, 1, 3, 3) / 8.0

# 4-connected adjacency kernel (up/down/left/right, no diagonals)
_ADJACENCY_KERNEL = torch.tensor(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]], dtype=torch.float32
).reshape(1, 1, 3, 3)

# Directional kernels for opposing-pair gap detection.
# Each pair detects an empty pixel with filled neighbors on opposite sides.
_DIR_PAIRS = []
for dy1, dx1, dy2, dx2 in [
    (-1, 0, 1, 0),   # top-bottom
    (0, -1, 0, 1),   # left-right
    (-1, -1, 1, 1),  # top-left to bottom-right
    (-1, 1, 1, -1),  # top-right to bottom-left
]:
    k1 = torch.zeros(1, 1, 3, 3)
    k2 = torch.zeros(1, 1, 3, 3)
    k1[0, 0, dy1 + 1, dx1 + 1] = 1.0
    k2[0, 0, dy2 + 1, dx2 + 1] = 1.0
    _DIR_PAIRS.append((k1, k2))


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


def neighbor_consistency_loss(output, stray_thresh=0.2):
    """Differentiable loss penalizing stray pixels and 1-pixel gaps.

    Two components:
    - Stray: filled pixels with very few filled neighbors
    - Directional gap: empty pixels with filled neighbors on opposing sides
      (e.g., filled above and below but empty in between = gap in outline).
      This catches the 1-pixel gaps that density-based detection misses.
    """
    dev = output.device
    out_2d = output.view(-1, 1, GRID_SIZE, GRID_SIZE)
    kernel = _NEIGHBOR_KERNEL.to(dev)
    neighbor_density = F.conv2d(out_2d, kernel, padding=1)

    # Stray: filled pixel with low neighbor density
    stray = (out_2d * F.relu(stray_thresh - neighbor_density)).mean()

    # Directional gap: for each of 4 opposing direction pairs, compute
    # gap_score = (1 - pixel) * side_a * side_b
    # This is high when pixel is empty but both opposing neighbors are filled.
    gap = torch.zeros(1, device=dev)
    for k1, k2 in _DIR_PAIRS:
        side_a = F.conv2d(out_2d, k1.to(dev), padding=1)
        side_b = F.conv2d(out_2d, k2.to(dev), padding=1)
        gap = gap + ((1.0 - out_2d) * side_a * side_b).mean()

    return stray + gap


def base_boundary_loss(recon, base_image):
    """Penalizes new pixels that are directly adjacent (4-connected) to base pixels.

    For conditional stages (eyes, smile, details), the model should place new
    features INSIDE the face, not along the outline. This computes a mask of
    all non-base pixels that are orthogonally adjacent to a base pixel, then
    penalizes the model's output in those locations.

    Args:
        recon: Model output (batch, 256), values in [0, 1]
        base_image: Base/condition (batch, 256), binary

    Returns:
        Mean penalty (scalar). High when model outputs bright pixels next to base.
    """
    dev = recon.device
    base_2d = base_image.view(-1, 1, GRID_SIZE, GRID_SIZE)
    recon_2d = recon.view(-1, 1, GRID_SIZE, GRID_SIZE)

    # Convolve base with cross kernel: pixels adjacent to at least one base pixel
    adj = F.conv2d(base_2d, _ADJACENCY_KERNEL.to(dev), padding=1)
    adjacent_to_base = (adj > 0).float()

    # Boundary zone = adjacent to base AND not a base pixel itself
    boundary_mask = adjacent_to_base * (1.0 - base_2d)

    # Penalize model output in the boundary zone
    return (recon_2d * boundary_mask).mean()


def flood_fill_gap_score(img_tensor):
    """Score shapes by flood-filling from center through empty pixels.

    If the fill reaches the border, the shape has a gap. Returns the number
    of border pixels reached per image (lower = more closed = better).
    Works on binarized images. Not differentiable — used for rejection sampling.
    """
    batch = (img_tensor > 0.5).view(-1, GRID_SIZE, GRID_SIZE).cpu().numpy()
    scores = []
    center = GRID_SIZE // 2

    for i in range(batch.shape[0]):
        binary = batch[i]
        visited = np.zeros_like(binary, dtype=bool)
        queue = [(center, center)]
        border_leaks = 0

        while queue:
            y, x = queue.pop()
            if y < 0 or y >= GRID_SIZE or x < 0 or x >= GRID_SIZE:
                continue
            if visited[y, x] or binary[y, x]:
                continue
            visited[y, x] = True
            if y == 0 or y == GRID_SIZE - 1 or x == 0 or x == GRID_SIZE - 1:
                border_leaks += 1
            queue.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])

        scores.append(float(border_leaks))

    return torch.tensor(scores)


def score_structural_quality(img_tensor):
    """Score a batch of 16x16 images for structural quality (lower = better).

    Combines flood-fill gap detection (catches real gaps in outlines),
    stray pixel counting, and pixel count sanity check.
    """
    with torch.no_grad():
        binary = (img_tensor > 0.5).float().view(-1, 1, GRID_SIZE, GRID_SIZE)
        kernel = _NEIGHBOR_KERNEL.to(img_tensor.device)
        neighbors = F.conv2d(binary, kernel, padding=1)

        # Stray pixels (filled with <2 of 8 neighbors)
        stray = (binary * (neighbors < 0.25).float()).sum(dim=(1, 2, 3))

        # Flood fill from center — the primary gap detector
        flood_score = flood_fill_gap_score(img_tensor)

        # Pixel count sanity
        pixel_count = binary.sum(dim=(1, 2, 3))
        count_penalty = F.relu(20.0 - pixel_count) + F.relu(pixel_count - 150.0)

        return stray + flood_score * 5.0 + count_penalty * 0.5


def staged_loss(recon, target, base_image, mu, logvar, beta,
                new_pixel_weight=2.0, sharpness_weight=0.15,
                connectivity_weight=0.0, boundary_weight=0.0):
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
        boundary_weight: Penalty for new pixels adjacent to base boundary (stages 2+)

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

    if boundary_weight > 0:
        total = total + boundary_weight * base_boundary_loss(recon_flat, base_flat)

    return total


# ── Refine AI ────────────────────────────────────────────────────


class RefineModel(nn.Module):
    """Learns to predict user corrections for a stage's VAE output.

    Input:  256 (generator raw output) + 256 (base/condition) = 512
    Hidden: 512 → 256 → 256 (ReLU + BatchNorm)
    Output: 256 (sigmoid → predicted corrected pixels)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )

    def forward(self, generator_output, base):
        """Predict corrected pixels given generator output and base layer.

        Args:
            generator_output: Raw VAE output (batch, 256), values in [0, 1]
            base: Base/condition from previous stage (batch, 256), binary

        Returns:
            Predicted corrected pixels (batch, 256), values in [0, 1]
        """
        gen_flat = generator_output.view(-1, 256)
        base_flat = base.view(-1, 256)
        combined = torch.cat([gen_flat, base_flat], dim=1)
        return self.net(combined)


def refine_loss(prediction, target, new_pixel_weight=2.0):
    """BCE loss between RefineModel prediction and the user's actual correction.

    Applies higher weight to pixels that were changed by the user (new pixels
    present in target but not in the generator's original output).

    Args:
        prediction: RefineModel output (batch, 256)
        target: User's actual corrected pixels (batch, 256)
        new_pixel_weight: Extra weight for changed pixels

    Returns:
        Weighted BCE loss (scalar)
    """
    pred_flat = prediction.view(-1, 256)
    target_flat = target.view(-1, 256)

    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')

    # Weight pixels that differ between prediction input and target more heavily
    changed = (target_flat > 0.5) != (pred_flat.detach() > 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[changed] = new_pixel_weight

    return (bce * weight_mask).mean()


def critic_correction_magnitude(refine_model, vae_output, base):
    """Compute per-sample correction magnitude from the RefineModel.

    This measures how much the RefineModel thinks each sample needs
    to be corrected. Higher values = the critic thinks this output is worse.

    Args:
        refine_model: Trained RefineModel (will be used in eval/no_grad mode)
        vae_output: Raw VAE output (batch, 256)
        base: Base/condition (batch, 256)

    Returns:
        Per-sample correction magnitude (batch,), values >= 0
    """
    with torch.no_grad():
        refine_model.eval()
        predicted_correction = refine_model(vae_output, base)

    # L1 difference per sample: how many pixels the critic would change
    diff = torch.abs(predicted_correction - vae_output.view(-1, 256))
    return diff.mean(dim=1)  # (batch,)


def experimental_staged_loss(recon, target, base_image, mu, logvar, beta,
                             refine_model=None, critic_weight=0.0,
                             critic_warmup_progress=1.0, focal_alpha=2.0,
                             new_pixel_weight=2.0, sharpness_weight=0.15,
                             connectivity_weight=0.0, boundary_weight=0.0):
    """Extended staged_loss with RefineModel critic signal.

    Adds two mechanisms on top of the standard loss:
    1. Per-sample focal weighting: samples the critic thinks are bad get
       quadratically higher loss weight (hard-example mining).
    2. Critic loss term: direct penalty proportional to how much the
       critic would correct the output.

    Args:
        recon: VAE reconstructed output (batch, 256)
        target: Target image (batch, 256)
        base_image: Base/condition (batch, 256)
        mu, logvar: Latent parameters
        beta: KL weight
        refine_model: Trained RefineModel (or None to skip critic)
        critic_weight: Weight of the critic loss term
        critic_warmup_progress: 0→1 ramp for critic (0=off, 1=full)
        focal_alpha: Quadratic scaling for hard-example weighting
        (remaining args same as staged_loss)

    Returns:
        Total loss (scalar)
    """
    target_flat = target.view(-1, 256)
    base_flat = base_image.view(-1, 256)
    recon_flat = recon.view(-1, 256)

    # ── Standard VAE loss components ──
    bce = F.binary_cross_entropy(recon_flat, target_flat, reduction='none')

    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight

    # Per-sample weighted BCE (keep per-sample for focal weighting)
    per_sample_bce = (bce * weight_mask).mean(dim=1)  # (batch,)

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sharp = sharpening_loss(recon_flat)

    # ── Critic-guided components ──
    if refine_model is not None and critic_weight > 0 and critic_warmup_progress > 0:
        correction_mag = critic_correction_magnitude(refine_model, recon, base_image)

        # Focal weighting: weight = 1 + α * correction_magnitude²
        # Slightly wrong → small bump, very wrong → heavy punishment
        focal_weights = 1.0 + focal_alpha * (correction_mag ** 2)
        focal_weights = focal_weights / focal_weights.mean()  # normalize

        # Apply focal weights to per-sample BCE
        weighted_bce = (per_sample_bce * focal_weights).mean()

        # Direct critic penalty: push VAE to minimize correction magnitude
        critic_penalty = correction_mag.mean()

        effective_critic_weight = critic_weight * critic_warmup_progress
    else:
        weighted_bce = per_sample_bce.mean()
        critic_penalty = torch.tensor(0.0, device=recon.device)
        effective_critic_weight = 0.0

    total = weighted_bce + beta * kld + sharpness_weight * sharp
    total = total + effective_critic_weight * critic_penalty

    if connectivity_weight > 0:
        total = total + connectivity_weight * neighbor_consistency_loss(recon_flat)

    if boundary_weight > 0:
        total = total + boundary_weight * base_boundary_loss(recon_flat, base_flat)

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