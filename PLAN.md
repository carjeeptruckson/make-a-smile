# Redesign Plan: Make-a-Smile v2

## The Core Idea: Curriculum Learning with Stacked CVAEs

The AI should learn to draw a face the same way a person would — one piece at a time, each piece building on the last. This is called **curriculum learning**: start with the simplest task, get good at it, then add complexity.

The system will have 4 small neural networks, each a **Conditional Variational Autoencoder (CVAE)**. Each one learns a single layer of the face and conditions on everything that came before it:

```
Stage 1 (VAE)   : noise → head shape
Stage 2 (CVAE)  : head image + noise → head + eyes
Stage 3 (CVAE)  : head+eyes image + noise → head + eyes + smile
Stage 4 (CVAE)  : head+eyes+smile image + noise → complete face (add hair and minor details)
```

The "noise" is a small learned latent code (z) that captures the *style* of that layer — what arrangement of eyes to use, what kind of smile, etc. This is what makes it generative: you can sample a new z to get a different variation.

The user provides the training data for each stage sequentially. Because each model is tiny and its job is narrowly scoped, it can learn well from far fewer examples than a model trying to do everything at once.

---

## Why This Preserves the Sketchy Feel

The AI learns directly from **your hand-drawn pixels**. If your eyes are sometimes single pixels, sometimes 2 stacked, sometimes a small cluster — the model learns that distribution. It doesn't enforce geometry. The latent code z will discover that these different arrangements are the main axis of variation for eyes, and it will learn to sample between them.

The current VAE failed because it tried to learn all of this simultaneously from scratch with a model too large for the data. Each stage here is a much simpler learning problem.

---

## Model Architecture

### Stage 1: Head VAE (no conditioning)

**Job**: Generate a believable sketchy head shape from scratch.

```
Encoder:  256 → 32 → 16 → (mu: 4, logvar: 4)
Decoder:  4 → 16 → 32 → 256
```

**Parameters**: ~18,000 total
- Encoder: (256×32 + 32) + (32×16 + 16) + (16×4 + 4)×2 ≈ 9,100
- Decoder: (4×16 + 16) + (16×32 + 32) + (32×256 + 256) ≈ 8,900

**Latent dim**: 4 — enough to capture rough shape variation (round vs. wider vs. taller vs. lumpy)

### Stages 2, 3, 4: Conditional VAEs

**Job**: Given the accumulated face so far as a base image, learn to add the next layer.

The encoder only sees the *target* (e.g., the head+eyes image). The decoder sees the latent z **plus the base image** as a conditioning input. This way the model learns to "read" the existing face and decide where the new layer fits.

```
Encoder:  256 (target) → 32 → (mu: 3, logvar: 3)
Decoder:  3 (z) + 256 (condition) = 259 → 32 → 256
```

**Parameters**: ~26,000 total per stage
- Encoder: (256×32 + 32) + (32×3 + 3)×2 ≈ 8,450
- Decoder: (259×32 + 32) + (32×256 + 256) ≈ 17,200

**Latent dim**: 3 per stage — tight regularization, forces the model to capture only the most important variation. Each dimension will naturally correspond to something meaningful (e.g., for eyes: horizontal position, size/density, height relative to head).

**Total model size across all 4 stages**: ~96,000 parameters, but split across 4 separate models, each trained on its own narrow task. Compare to the current single model (83,700 params) trained on everything at once.

---

## How the Loss Works

### Stage 1 (normal VAE loss):
```
Loss = BCE(reconstruction, target) + β × KLD(mu, logvar)
```

### Stages 2–4 (conditional, but same formula):
The encoder encodes the full target image. The decoder conditions on the base layer and reconstructs the full target. The model must learn: "given this base, how do I reconstruct base+new_layer?" — implicitly learning what the new layer looks like and where it goes.

**Important detail**: The loss should weight the *new pixels more heavily* than the base pixels. The base pixels are already known; what matters is getting the new layer right. This can be done by multiplying the per-pixel BCE loss by a mask that is 2.0 on new pixels and 1.0 on base pixels.

### KL Annealing:
The current code has a fixed `kl_weight=1.5`, which is too aggressive for early training. Use a linear warmup:
- Epochs 1–100: β = 0.0 (pure reconstruction)
- Epochs 100–400: β ramps from 0.0 → 0.8
- Epochs 400+: β = 0.8 (stable regularization)

This lets the model first learn to reconstruct, then organize the latent space — otherwise the KL term dominates early and the model collapses.

---

## Data Collection: The Stage-Aware Drawing Studio

This is the biggest change to the UI. The Drawing Studio needs to support layered drawing.

### How it works:

The canvas has two layers:
- **Base layer** (shown in gray, not editable): the accumulated face from previous stages
- **Current layer** (shown in black, editable): what the user is drawing now

The user can't accidentally erase the base layer — it's locked. They only draw the new component.

When the user saves, the app stores **two things**:
1. The base image (condition input for the model)
2. The full combined image (target output for the model)

### Stage 1 (Heads):
- Base layer: blank (nothing to build on)
- User draws head shapes — closed outlines, organic, varied, no facial features
- Save → stored as both condition (blank) and target (head only)
- Goal: 40–60 unique head shapes

### Stage 2 (Eyes):
- User picks any saved Stage 1 head to use as a base
- Base layer loads that head image in gray
- User draws eyes on top — any arrangement they feel like
- Multiple saves with the *same head base* but different eye layouts count as separate training examples
- Save → stores (head-only image, head+eyes image) pair
- Goal: ~3–5 eye variations per head shape, total ~100–150 pairs

### Stage 3 (Smile):
- User picks any saved Stage 2 image (head+eyes) as a base
- Draws the smile
- Save → stores (head+eyes image, head+eyes+smile image) pair
- Goal: ~2–3 smile variations per head+eyes image, total ~80–120 pairs

### Stage 4 (Details — optional):
- Base: Stage 3 image
- User adds hair, nose, ears, eyebrows, anything
- Goal: ~60–80 pairs

### The base layer picker UI:
A small panel below the canvas shows a scrollable row of thumbnail previews of all saved images from the previous stage. User clicks one to load it as the base layer. A "Clear base" button unloads it (for Stage 1 or free drawing).

---

## Data Storage

Replace the current single `my_faces.csv` with stage-specific pair files:

```
data/
  stage1_heads.csv          — 256 values per row (head images only)
  stage2_eyes_base.csv      — 256 values per row (the head used as condition)
  stage2_eyes_target.csv    — 256 values per row (head + eyes)
  stage3_smile_base.csv     — 256 values per row (head+eyes condition)
  stage3_smile_target.csv   — 256 values per row (head+eyes+smile)
  stage4_detail_base.csv    — 256 values per row (head+eyes+smile condition)
  stage4_detail_target.csv  — 256 values per row (complete face)

  model_stage1.pth
  model_stage2.pth
  model_stage3.pth
  model_stage4.pth
```

Each pair file has matching row counts: row N of `_base.csv` is the condition for row N of `_target.csv`.

---

## Training Per Stage

Each stage trains independently. The menu shows a training button per stage that only enables once the previous stage's model exists and the current stage has enough data.

**Augmentation**: Horizontal flip on both base and target simultaneously (keeping them paired). This doubles the data.

**Epochs**: 500 (less than current 600 because each task is simpler). With KL annealing, convergence is faster.

**Batch size**: All data at once (batch gradient descent). With only ~100–300 examples, this is fine and more stable than mini-batching.

**Recommended minimum data to enable training**:
- Stage 1: 30 head images
- Stage 2: 60 pairs
- Stage 3: 40 pairs
- Stage 4: 40 pairs (optional)

---

## The Generator: Stage-by-Stage Assembly

The Generator screen becomes a **face assembly studio**. Instead of one canvas and 6 unlabeled sliders, it shows:

```
[Canvas: 16x16 display]

[HEAD]    z1 ●──────── z2 ●──────── z3 ●──────── z4 ●────────   [New Head]
[EYES]    z1 ●──────── z2 ●──────── z3 ●────────                [New Eyes]
[SMILE]   z1 ●──────── z2 ●──────── z3 ●────────                [New Smile]
[DETAILS] z1 ●──────── z2 ●──────── z3 ●────────                [New Details]

[Surprise Me (all)]  [Morph]  [Main Menu]
```

Each row's sliders control the latent z for that stage. Moving a slider rerenders only that stage and everything above it. The "New X" button randomizes just that row's z.

This means the user can:
- Keep a head they like and try many different eye combinations
- Keep head+eyes and explore smiles
- Lock in a complete face and explore details
- Morph only one component at a time

Only stages that have trained models are shown. If only Stage 1 is trained, only the Head row appears.

---

## The Refinement Loop (Completely Redesigned)

Because generation is now per-component, the refinement loop is much clearer:

1. **Generate a face** (all stages run in sequence)
2. User looks at each layer:
   - "Head looks good" → nothing to do for this layer
   - "Eyes are wrong" → click "New Eyes" to resample, or draw a correction
3. **Drawing a correction for a specific layer**:
   - The app shows the current base layer (e.g., head) in gray
   - User draws the correction for that layer (e.g., new eyes)
   - This gets saved as a new training pair for that stage
   - A mini-retraining step (10–20 gradient steps) updates only that stage's model

This means the user always knows what they're correcting. They're not redrawing the whole face — just the component that was wrong. And the model update only affects the relevant stage.

---

## Implementation Order

### Step 1: New model file
Replace `model.py` with a new file containing:
- `HeadVAE`: the Stage 1 unconditional VAE (encoder/decoder as spec'd above)
- `ConditionalVAE`: the Stage 2–4 conditional VAE
- `staged_loss(recon, target, base, mu, logvar, beta, new_pixel_weight=2.0)`: loss with pixel weighting and KL beta
- `kl_beta(epoch, warmup_start=100, warmup_end=400, final_beta=0.8)`: annealing schedule

### Step 2: New config
Update `config.py`:
- Replace `LATENT_DIM = 6` with `STAGE1_Z = 4`, `STAGE2_Z = 3`, `STAGE3_Z = 3`, `STAGE4_Z = 3`
- Add stage-specific file paths
- Add minimum data thresholds per stage

### Step 3: New data layer in Drawing Studio
Modify `drawer.py`:
- Add a `base_layer` grid (separate from `grid_data`)
- Add a base layer picker panel (thumbnails from previous stage's data)
- Render base layer pixels in a distinct color (e.g., `#aaaaaa`)
- On save: write to the correct stage's base and target CSV files

### Step 4: Training in Menu
Modify `menu.py`:
- Replace the single "Train AI" button with 4 stage-specific training buttons
- Each button enables once: (a) previous model exists and (b) current stage has enough pairs
- Each trains the correct model class with the correct data files
- Training window shows the stage being trained

### Step 5: New Generator
Rewrite `generator.py`:
- Per-stage slider rows with stage labels
- "New X" buttons per row
- Render pipeline: run Stage 1 decoder → pass result to Stage 2 decoder → etc.
- Only show rows for trained stages

### Step 6: New Refinement
Rewrite `refine.py`:
- Generate face layer by layer
- Show which layer is being evaluated
- "Redraw this layer" uses base-layer drawer mode
- Mini-training step updates only the relevant stage's model

---

## Summary

| Aspect | Current | New |
|--------|---------|-----|
| Architecture | 1 big VAE, everything at once | 4 tiny CVAEs, one per layer |
| Params per model | 83,700 | ~18,000 (S1) / ~26,000 (S2–4) |
| Training data needed | 50 complete faces | 30 heads, then 60/40/40 pairs per stage |
| Latent space meaning | Random (Trait 1–6) | Learned per component |
| What AI learns | "What is a face?" | "What are good eye arrangements given this head?" |
| Generation | All-at-once, often incoherent | Staged: head → eyes → smile → details |
| Regeneration | All-or-nothing | Regenerate one layer, keep the rest |
| Refinement | Redraw whole pixels, blind | Redraw one specific layer with visible base |
| Sketchy feel | Lost (model too constrained) | Preserved (learned directly from your drawings) |
