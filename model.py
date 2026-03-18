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


# ── Convolutional Continuous VAE: Stage 1 (head shapes) ──────


class HeadVAE(nn.Module):
    """Conv VAE for head shape generation (stage 1).

    Encoder: (B, 1, 16, 16) -> conv -> (B, 32, 4, 4) -> flatten -> mu/logvar
    Decoder: z -> reshape 4x4 -> deconv -> (B, 1, 16, 16)

    Uses GroupNorm instead of BatchNorm for stability with small batches.
    """

    def __init__(self):
        super().__init__()
        z_dim = STAGE1_Z

        # Encoder: 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> 16ch, 8x8
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32ch, 4x4
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        # 32 * 4 * 4 = 512
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        # Decoder: z -> 4x4 -> 8x8 -> 16x16
        self.fc_dec = nn.Linear(z_dim, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # -> 16ch, 8x8
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # -> 1ch, 16x16
            nn.Sigmoid(),
        )

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 32, 4, 4)
        return self.decoder(h).view(-1, 256)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ── Convolutional Continuous VAE: Stages 2-4 (conditional) ───


class ConditionalVAE(nn.Module):
    """Conv VAE for conditional stages (eyes, smile, details).

    Encoder takes 2-channel input (target + condition).
    Decoder receives z concatenated with downsampled condition at 4x4.
    No full-res skip connection — forces z to carry meaningful information.
    """

    STAGE_Z = {
        "stage2": STAGE2_Z,
        "stage3": STAGE3_Z,
        "stage4": STAGE4_Z,
    }

    def __init__(self, stage_name="stage2"):
        super().__init__()
        self.stage_name = stage_name
        z_dim = self.STAGE_Z.get(stage_name, 4)

        # Encoder: 2-channel input (target + condition)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),   # -> 16ch, 8x8
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32ch, 4x4
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        # Decoder: z + downsampled condition at 4x4
        self.fc_dec = nn.Linear(z_dim, 512)
        self.dec_up1 = nn.ConvTranspose2d(32 + 1, 16, 4, stride=2, padding=1)  # -> 8x8
        self.dec_gn1 = nn.GroupNorm(4, 16)
        self.dec_up2 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)      # -> 16x16
        self.dec_gn2 = nn.GroupNorm(4, 16)
        self.dec_out = nn.Conv2d(16, 1, 3, padding=1)

        self.cond_down = nn.AvgPool2d(4)  # 16x16 -> 4x4

    def encode(self, target, condition):
        if target.dim() == 2:
            target = target.view(-1, 1, GRID_SIZE, GRID_SIZE)
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([target, condition], dim=1)
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)
        h = self.fc_dec(z).view(-1, 32, 4, 4)
        cond_small = self.cond_down(condition)  # (B, 1, 4, 4)
        h = torch.cat([h, cond_small], dim=1)   # (B, 33, 4, 4)
        h = F.relu(self.dec_gn1(self.dec_up1(h)))
        h = F.relu(self.dec_gn2(self.dec_up2(h)))
        return torch.sigmoid(self.dec_out(h)).view(-1, 256)

    def forward(self, target, condition):
        if target.dim() == 2:
            target = target.view(-1, 1, GRID_SIZE, GRID_SIZE)
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)
        mu, logvar = self.encode(target, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon.view(-1, 256), mu, logvar


# ── Loss functions ───────────────────────────────────────────


def sharpening_loss(output):
    """Penalizes outputs near 0.5, encouraging crisp binary values."""
    eps = 1e-7
    clamped = output.clamp(eps, 1.0 - eps)
    entropy = -(clamped * torch.log(clamped) + (1 - clamped) * torch.log(1 - clamped))
    return entropy.mean()


def neighbor_consistency_loss(output, stray_thresh=0.2):
    """Differentiable loss penalizing stray pixels and 1-pixel gaps."""
    dev = output.device
    out_2d = output.view(-1, 1, GRID_SIZE, GRID_SIZE)
    kernel = _NEIGHBOR_KERNEL.to(dev)
    neighbor_density = F.conv2d(out_2d, kernel, padding=1)

    stray = (out_2d * F.relu(stray_thresh - neighbor_density)).mean()

    gap = torch.zeros(1, device=dev)
    for k1, k2 in _DIR_PAIRS:
        side_a = F.conv2d(out_2d, k1.to(dev), padding=1)
        side_b = F.conv2d(out_2d, k2.to(dev), padding=1)
        gap = gap + ((1.0 - out_2d) * side_a * side_b).mean()

    return stray + gap


def base_boundary_loss(recon, base_image):
    """Penalizes new pixels directly adjacent (4-connected) to base pixels."""
    dev = recon.device
    base_2d = base_image.view(-1, 1, GRID_SIZE, GRID_SIZE)
    recon_2d = recon.view(-1, 1, GRID_SIZE, GRID_SIZE)

    adj = F.conv2d(base_2d, _ADJACENCY_KERNEL.to(dev), padding=1)
    adjacent_to_base = (adj > 0).float()
    boundary_mask = adjacent_to_base * (1.0 - base_2d)
    return (recon_2d * boundary_mask).mean()


def staged_loss(recon, target, base_image, mu, logvar, beta,
                new_pixel_weight=2.0, sharpness_weight=0.15,
                connectivity_weight=0.0, boundary_weight=0.0):
    """Loss function for staged training with pixel weighting."""
    target_flat = target.view(-1, 256)
    base_flat = base_image.view(-1, 256)
    recon_flat = recon.view(-1, 256)

    bce = nn.functional.binary_cross_entropy(recon_flat, target_flat, reduction='none')

    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight

    weighted_bce = (bce * weight_mask).mean()

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    sharp = sharpening_loss(recon_flat)

    total = weighted_bce + beta * kld + sharpness_weight * sharp

    if connectivity_weight > 0:
        total = total + connectivity_weight * neighbor_consistency_loss(recon_flat)

    if boundary_weight > 0:
        total = total + boundary_weight * base_boundary_loss(recon_flat, base_flat)

    return total


# ── Scoring functions (for rejection sampling) ───────────────


def flood_fill_gap_score(img_tensor):
    """Score shapes by flood-filling from center through empty pixels."""
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
    """Score a batch of 16x16 images for structural quality (lower = better)."""
    with torch.no_grad():
        binary = (img_tensor > 0.5).float().view(-1, 1, GRID_SIZE, GRID_SIZE)
        kernel = _NEIGHBOR_KERNEL.to(img_tensor.device)
        neighbors = F.conv2d(binary, kernel, padding=1)

        stray = (binary * (neighbors < 0.25).float()).sum(dim=(1, 2, 3))
        flood_score = flood_fill_gap_score(img_tensor)
        pixel_count = binary.sum(dim=(1, 2, 3))
        count_penalty = F.relu(20.0 - pixel_count) + F.relu(pixel_count - 150.0)

        return stray + flood_score * 5.0 + count_penalty * 0.5


# ── Refine AI ────────────────────────────────────────────────


class RefineModel(nn.Module):
    """Learns to predict user corrections for a stage's VAE output."""

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
        gen_flat = generator_output.view(-1, 256)
        base_flat = base.view(-1, 256)
        combined = torch.cat([gen_flat, base_flat], dim=1)
        return self.net(combined)


def refine_loss(prediction, target, new_pixel_weight=2.0):
    """BCE loss between RefineModel prediction and the user's actual correction."""
    pred_flat = prediction.view(-1, 256)
    target_flat = target.view(-1, 256)

    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
    changed = (target_flat > 0.5) != (pred_flat.detach() > 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[changed] = new_pixel_weight
    return (bce * weight_mask).mean()


def critic_correction_magnitude(refine_model, vae_output, base):
    """Compute per-sample correction magnitude from the RefineModel."""
    with torch.no_grad():
        refine_model.eval()
        predicted_correction = refine_model(vae_output, base)
    diff = torch.abs(predicted_correction - vae_output.view(-1, 256))
    return diff.mean(dim=1)


def experimental_staged_loss(recon, target, base_image, mu, logvar, beta,
                             refine_model=None, critic_weight=0.0,
                             critic_warmup_progress=1.0, focal_alpha=2.0,
                             new_pixel_weight=2.0, sharpness_weight=0.15,
                             connectivity_weight=0.0, boundary_weight=0.0):
    """Extended staged_loss with RefineModel critic signal."""
    target_flat = target.view(-1, 256)
    base_flat = base_image.view(-1, 256)
    recon_flat = recon.view(-1, 256)

    bce = F.binary_cross_entropy(recon_flat, target_flat, reduction='none')

    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight

    per_sample_bce = (bce * weight_mask).mean(dim=1)

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sharp = sharpening_loss(recon_flat)

    if refine_model is not None and critic_weight > 0 and critic_warmup_progress > 0:
        correction_mag = critic_correction_magnitude(refine_model, recon, base_image)
        focal_weights = 1.0 + focal_alpha * (correction_mag ** 2)
        focal_weights = focal_weights / focal_weights.mean()
        weighted_bce = (per_sample_bce * focal_weights).mean()
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


def kl_beta_schedule(epoch, warmup_start=100, warmup_end=400, final_beta=0.3):
    """Linear KL annealing schedule."""
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return final_beta * progress
    else:
        return final_beta


def add_noise(img_tensor, noise_factor=0.03):
    """Denoising augmentation: randomly flips a fraction of pixels."""
    noise = torch.rand_like(img_tensor)
    noisy_img = torch.where(noise < noise_factor, 1.0 - img_tensor, img_tensor)
    return noisy_img


# ── Data augmentation ────────────────────────────────────────


def augment_batch(targets, bases, stage, symmetry_stages=None):
    """Apply augmentations to expand the training set.

    For all stages: horizontal flip + 4 translations (1px each direction).
    For symmetry_stages (e.g. eyes): also mirror-average to enforce symmetry.

    Args:
        targets: list of 256-float lists
        bases: list of 256-float lists (empty list for stage 1)
        stage: stage number
        symmetry_stages: set of stage numbers that get symmetry enforcement

    Returns:
        (aug_targets, aug_bases) — expanded lists
    """
    if symmetry_stages is None:
        symmetry_stages = set()

    gs = GRID_SIZE
    aug_targets = []
    aug_bases = []

    for i, t in enumerate(targets):
        t_grid = np.array(t).reshape(gs, gs)
        b_grid = np.array(bases[i]).reshape(gs, gs) if bases else np.zeros((gs, gs))

        # Original
        aug_targets.append(t_grid.flatten().tolist())
        if bases:
            aug_bases.append(b_grid.flatten().tolist())
        else:
            aug_bases.append([0.0] * 256)

        # Horizontal flip
        t_flip = np.fliplr(t_grid)
        b_flip = np.fliplr(b_grid)
        aug_targets.append(t_flip.flatten().tolist())
        aug_bases.append(b_flip.flatten().tolist() if bases else [0.0] * 256)

        # Translations (1px in 4 directions)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            t_shifted = np.zeros_like(t_grid)
            b_shifted = np.zeros_like(b_grid)

            # Source and destination slices
            src_y = (max(0, -dy), min(gs, gs - dy))
            dst_y = (max(0, dy), min(gs, gs + dy))
            src_x = (max(0, -dx), min(gs, gs - dx))
            dst_x = (max(0, dx), min(gs, gs + dx))

            t_shifted[dst_y[0]:dst_y[1], dst_x[0]:dst_x[1]] = \
                t_grid[src_y[0]:src_y[1], src_x[0]:src_x[1]]
            b_shifted[dst_y[0]:dst_y[1], dst_x[0]:dst_x[1]] = \
                b_grid[src_y[0]:src_y[1], src_x[0]:src_x[1]]

            aug_targets.append(t_shifted.flatten().tolist())
            aug_bases.append(b_shifted.flatten().tolist() if bases else [0.0] * 256)

        # Symmetry enforcement for eyes: average with horizontal mirror
        if stage in symmetry_stages:
            t_sym = ((t_grid + t_flip) / 2.0)
            t_sym = (t_sym > 0.5).astype(float)
            aug_targets.append(t_sym.flatten().tolist())
            aug_bases.append(b_grid.flatten().tolist() if bases else [0.0] * 256)

    return aug_targets, aug_bases
