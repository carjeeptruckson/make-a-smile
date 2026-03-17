import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    GRID_SIZE,
    VQ_NUM_EMBEDDINGS, VQ_EMBED_DIM, VQ_EMA_DECAY, VQ_DEAD_CODE_THRESHOLD,
)

# ── Spatial convolution kernels (used by loss functions) ─────────

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


# ── Vector Quantizer ─────────────────────────────────────────────


class VectorQuantizer(nn.Module):
    """Discrete codebook with EMA updates (no gradient-based codebook learning).

    Each spatial position in the encoder output independently selects the
    nearest codebook entry. Gradients pass through via straight-through
    estimator. Codebook vectors are updated using exponential moving averages
    of the encoder outputs assigned to them.

    Args:
        num_embeddings: Number of codebook entries (K)
        embed_dim: Dimension of each codebook vector
        decay: EMA decay rate (higher = slower updates)
        dead_threshold: Reset codes unused for this many forward passes
    """

    def __init__(self, num_embeddings=VQ_NUM_EMBEDDINGS, embed_dim=VQ_EMBED_DIM,
                 decay=VQ_EMA_DECAY, dead_threshold=VQ_DEAD_CODE_THRESHOLD):
        super().__init__()
        self.K = num_embeddings
        self.D = embed_dim
        self.decay = decay
        self.dead_threshold = dead_threshold

        # Codebook: K vectors of dimension D
        self.register_buffer("embedding", torch.randn(num_embeddings, embed_dim))
        # EMA tracking buffers
        self.register_buffer("ema_count", torch.ones(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        # Usage tracking for dead code reset
        self.register_buffer("usage_count", torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, z_e):
        """Quantize encoder output and compute commitment loss.

        Args:
            z_e: Encoder output, shape (B, D, H, W)

        Returns:
            z_q: Quantized output (same shape as z_e), with straight-through gradient
            commitment_loss: ||z_e - sg(z_q)||^2, scalar
            indices: Codebook indices, shape (B, H, W)
            codebook_usage: Fraction of codebook entries used in this batch
        """
        B, D, H, W = z_e.shape

        # Reshape to (B*H*W, D) for distance computation
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # (N, D)

        # Find nearest codebook entry for each position (L2 distance)
        # dist = ||flat||^2 + ||emb||^2 - 2*flat@emb.T
        dist = (flat.pow(2).sum(dim=1, keepdim=True)
                + self.embedding.pow(2).sum(dim=1)
                - 2 * flat @ self.embedding.t())
        indices = dist.argmin(dim=1)  # (N,)

        # Look up quantized vectors
        z_q_flat = self.embedding[indices]  # (N, D)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # EMA codebook update (training only)
        if self.training:
            one_hot = F.one_hot(indices, self.K).float()  # (N, K)
            # Count assignments per code
            counts = one_hot.sum(dim=0)  # (K,)
            # Sum of encoder outputs assigned to each code
            assigned_sum = one_hot.t() @ flat  # (K, D)

            self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(assigned_sum, alpha=1 - self.decay)

            # Laplace smoothing to avoid division by zero
            n = self.ema_count.sum()
            smoothed = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
            self.embedding.copy_(self.ema_weight / smoothed.unsqueeze(1))

            # Track usage for dead code reset
            used = (counts > 0)
            self.usage_count[used] = 0
            self.usage_count[~used] += 1
            self._reset_dead_codes(flat)

        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()

        indices = indices.reshape(B, H, W)
        codebook_usage = (indices.unique().numel() / self.K)
        return z_q, commitment_loss, indices, codebook_usage

    def _reset_dead_codes(self, flat):
        """Reinitialize codebook entries that haven't been used recently."""
        dead = self.usage_count >= self.dead_threshold
        n_dead = dead.sum().item()
        if n_dead == 0:
            return
        # Pick random encoder outputs to replace dead codes
        rand_idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        self.embedding[dead] = flat[rand_idx].detach()
        self.ema_weight[dead] = flat[rand_idx].detach()
        self.ema_count[dead] = 1.0
        self.usage_count[dead] = 0

    def embed(self, indices):
        """Look up codebook entries by index (used during generation).

        Args:
            indices: (B, H, W) integer indices into the codebook

        Returns:
            (B, D, H, W) codebook vectors
        """
        B, H, W = indices.shape
        flat_idx = indices.reshape(-1)
        vectors = self.embedding[flat_idx]  # (B*H*W, D)
        return vectors.reshape(B, H, W, self.D).permute(0, 3, 1, 2)


# ── Conv VQ-VAE: Stage 1 (unconditional head shapes) ────────────


class HeadConvVQVAE(nn.Module):
    """Convolutional VQ-VAE for generating head shapes (stage 1).

    Encoder: (B, 1, 16, 16) -> conv layers -> (B, D, 4, 4)
    Quantizer: snap each 4x4 position to nearest codebook entry
    Decoder: (B, D, 4, 4) -> deconv layers -> (B, 1, 16, 16)

    No KL divergence — uses commitment loss + EMA codebook updates.
    """

    def __init__(self):
        super().__init__()
        D = VQ_EMBED_DIM

        # Encoder: 16x16 -> 8x8 -> 4x4 -> project to embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, D, 1),  # 1x1 conv to project to embed_dim
        )

        self.vq = VectorQuantizer()

        # Decoder: 4x4 -> 8x8 -> 16x16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(D, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode input image to pre-quantization embeddings.

        Args:
            x: (B, 1, 16, 16) or (B, 256) flat input

        Returns:
            z_e: (B, D, 4, 4) continuous embeddings (before quantization)
        """
        if x.dim() == 2:
            x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        return self.encoder(x)

    def decode(self, codes=None, indices=None):
        """Decode from quantized embeddings or codebook indices.

        Args:
            codes: (B, D, 4, 4) quantized embeddings (from training forward pass)
            indices: (B, 4, 4) codebook indices (for generation)

        Returns:
            (B, 1, 16, 16) reconstructed image
        """
        if indices is not None:
            codes = self.vq.embed(indices)
        return self.decoder(codes)

    def forward(self, x):
        """Full forward pass: encode -> quantize -> decode.

        Args:
            x: (B, 1, 16, 16) or (B, 256) input image

        Returns:
            recon: (B, 1, 16, 16) reconstruction
            commit_loss: commitment loss scalar
            indices: (B, 4, 4) selected codebook indices
            usage: fraction of codebook used
        """
        if x.dim() == 2:
            x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        z_e = self.encoder(x)
        z_q, commit_loss, indices, usage = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, commit_loss, indices, usage


# ── Conv VQ-VAE: Stages 2-4 (conditional) ───────────────────────


class ConditionalConvVQVAE(nn.Module):
    """Convolutional VQ-VAE for conditional stages (eyes, smile, details).

    The encoder takes a 2-channel input: target + condition (base layer).
    This way the latent codes only need to represent the DELTA between
    base and target, not the entire image.

    The decoder receives the quantized codes plus the condition at two
    resolutions (4x4 downsampled + 16x16 full) for precise spatial alignment.
    """

    def __init__(self, stage_name="stage2"):
        super().__init__()
        self.stage_name = stage_name
        D = VQ_EMBED_DIM

        # Encoder: 2 input channels (target + condition)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, D, 1),
        )

        self.vq = VectorQuantizer()

        # Decoder: codes + downsampled condition at 4x4
        self.dec_up1 = nn.ConvTranspose2d(D + 1, 32, 4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_up2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        # Final conv: decoded features + full-res condition skip connection
        self.dec_out = nn.Conv2d(16 + 1, 1, 3, padding=1)

        # Downsampler for condition -> 4x4
        self.cond_down = nn.AvgPool2d(4)

    def encode(self, target, condition):
        """Encode target+condition pair to pre-quantization embeddings.

        Args:
            target: (B, 1, 16, 16) or (B, 256) target image
            condition: (B, 1, 16, 16) or (B, 256) base/condition image

        Returns:
            z_e: (B, D, 4, 4)
        """
        if target.dim() == 2:
            target = target.view(-1, 1, GRID_SIZE, GRID_SIZE)
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([target, condition], dim=1)  # (B, 2, 16, 16)
        return self.encoder(x)

    def decode(self, codes=None, indices=None, condition=None):
        """Decode from quantized embeddings or codebook indices.

        Args:
            codes: (B, D, 4, 4) quantized embeddings
            indices: (B, 4, 4) codebook indices (for generation)
            condition: (B, 1, 16, 16) or (B, 256) base layer (required)

        Returns:
            (B, 1, 16, 16) reconstructed image
        """
        if indices is not None:
            codes = self.vq.embed(indices)
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)

        # Inject downsampled condition at the bottleneck
        cond_small = self.cond_down(condition)  # (B, 1, 4, 4)
        h = torch.cat([codes, cond_small], dim=1)  # (B, D+1, 4, 4)
        h = F.relu(self.dec_bn1(self.dec_up1(h)))   # (B, 32, 8, 8)
        h = F.relu(self.dec_bn2(self.dec_up2(h)))    # (B, 16, 16, 16)

        # Skip connection: concat full-resolution condition
        h = torch.cat([h, condition], dim=1)          # (B, 17, 16, 16)
        return torch.sigmoid(self.dec_out(h))          # (B, 1, 16, 16)

    def forward(self, target, condition):
        """Full forward pass.

        Args:
            target: (B, 1, 16, 16) or (B, 256) target image
            condition: (B, 1, 16, 16) or (B, 256) base/condition image

        Returns:
            recon: (B, 1, 16, 16)
            commit_loss: commitment loss scalar
            indices: (B, 4, 4) codebook indices
            usage: fraction of codebook used
        """
        if target.dim() == 2:
            target = target.view(-1, 1, GRID_SIZE, GRID_SIZE)
        if condition.dim() == 2:
            condition = condition.view(-1, 1, GRID_SIZE, GRID_SIZE)

        z_e = self.encoder(torch.cat([target, condition], dim=1))
        z_q, commit_loss, indices, usage = self.vq(z_e)
        recon = self.decode(codes=z_q, condition=condition)
        return recon, commit_loss, indices, usage


# ── Loss functions ───────────────────────────────────────────────


def sharpening_loss(output):
    """Penalizes outputs near 0.5, encouraging crisp binary values.

    Computes negative binary entropy per pixel. Pixels at 0.5 get the
    highest penalty; pixels near 0 or 1 get near-zero penalty.
    """
    eps = 1e-7
    clamped = output.clamp(eps, 1.0 - eps)
    entropy = -(clamped * torch.log(clamped) + (1 - clamped) * torch.log(1 - clamped))
    return entropy.mean()


def neighbor_consistency_loss(output, stray_thresh=0.2):
    """Differentiable loss penalizing stray pixels and 1-pixel gaps.

    Two components:
    - Stray: filled pixels with very few filled neighbors
    - Directional gap: empty pixels with filled neighbors on opposing sides
    """
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
    """Penalizes new pixels directly adjacent (4-connected) to base pixels.

    For conditional stages, new features should go INSIDE the face, not
    along the outline.
    """
    dev = recon.device
    base_2d = base_image.view(-1, 1, GRID_SIZE, GRID_SIZE)
    recon_2d = recon.view(-1, 1, GRID_SIZE, GRID_SIZE)

    adj = F.conv2d(base_2d, _ADJACENCY_KERNEL.to(dev), padding=1)
    adjacent_to_base = (adj > 0).float()
    boundary_mask = adjacent_to_base * (1.0 - base_2d)
    return (recon_2d * boundary_mask).mean()


def vq_staged_loss(recon, target, base_image, commit_loss,
                   new_pixel_weight=2.0, sharpness_weight=0.15,
                   commitment_weight=0.25,
                   connectivity_weight=0.0, boundary_weight=0.0):
    """Loss function for VQ-VAE staged training.

    Replaces the old staged_loss — no KL divergence, uses commitment loss instead.

    Args:
        recon: Reconstructed image (B, 1, 16, 16) or (B, 256)
        target: Target image, same shape
        base_image: Base/condition image, same shape
        commit_loss: Commitment loss from VectorQuantizer
        new_pixel_weight: Extra weight for newly drawn pixels
        sharpness_weight: Penalty for soft/uncertain pixels
        commitment_weight: Weight for commitment loss term
        connectivity_weight: Stray pixel / gap penalty (stage 1)
        boundary_weight: Base-adjacent pixel penalty (stages 2+)

    Returns:
        Total loss (scalar)
    """
    target_flat = target.view(-1, 256)
    base_flat = base_image.view(-1, 256)
    recon_flat = recon.view(-1, 256)

    # Per-pixel BCE with higher weight on new pixels
    bce = F.binary_cross_entropy(recon_flat, target_flat, reduction='none')
    new_pixels = (target_flat > 0.5) & (base_flat < 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[new_pixels] = new_pixel_weight
    weighted_bce = (bce * weight_mask).mean()

    # Sharpening: penalize outputs near 0.5
    sharp = sharpening_loss(recon_flat)

    total = weighted_bce + commitment_weight * commit_loss + sharpness_weight * sharp

    if connectivity_weight > 0:
        total = total + connectivity_weight * neighbor_consistency_loss(recon_flat)

    if boundary_weight > 0:
        total = total + boundary_weight * base_boundary_loss(recon_flat, base_flat)

    return total


# ── Scoring functions (for rejection sampling) ───────────────────


def flood_fill_gap_score(img_tensor):
    """Score shapes by flood-filling from center through empty pixels.

    If the fill reaches the border, the shape has a gap. Returns the number
    of border pixels reached per image (lower = more closed = better).
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

    Combines flood-fill gap detection, stray pixel counting, and pixel
    count sanity check.
    """
    with torch.no_grad():
        binary = (img_tensor > 0.5).float().view(-1, 1, GRID_SIZE, GRID_SIZE)
        kernel = _NEIGHBOR_KERNEL.to(img_tensor.device)
        neighbors = F.conv2d(binary, kernel, padding=1)

        stray = (binary * (neighbors < 0.25).float()).sum(dim=(1, 2, 3))
        flood_score = flood_fill_gap_score(img_tensor)
        pixel_count = binary.sum(dim=(1, 2, 3))
        count_penalty = F.relu(20.0 - pixel_count) + F.relu(pixel_count - 150.0)

        return stray + flood_score * 5.0 + count_penalty * 0.5


# ── Refine AI ────────────────────────────────────────────────────


class RefineModel(nn.Module):
    """Learns to predict user corrections for a stage's VQ-VAE output.

    Input:  256 (generator raw output) + 256 (base/condition) = 512
    Hidden: 512 -> 256 -> 256 (ReLU + BatchNorm)
    Output: 256 (sigmoid -> predicted corrected pixels)
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
        gen_flat = generator_output.view(-1, 256)
        base_flat = base.view(-1, 256)
        combined = torch.cat([gen_flat, base_flat], dim=1)
        return self.net(combined)


def refine_loss(prediction, target, new_pixel_weight=2.0):
    """BCE loss between RefineModel prediction and user's actual correction."""
    pred_flat = prediction.view(-1, 256)
    target_flat = target.view(-1, 256)

    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
    changed = (target_flat > 0.5) != (pred_flat.detach() > 0.5)
    weight_mask = torch.ones_like(bce)
    weight_mask[changed] = new_pixel_weight
    return (bce * weight_mask).mean()


def add_noise(img_tensor, noise_factor=0.03):
    """Denoising augmentation: randomly flips a fraction of pixels.

    Default 3% — flips ~8 pixels per 16x16 image.
    """
    noise = torch.rand_like(img_tensor)
    return torch.where(noise < noise_factor, 1.0 - img_tensor, img_tensor)
