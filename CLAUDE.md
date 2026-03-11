# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
./run.sh
```

This creates a Python 3.13 virtual environment, installs `torch` and `numpy`, then launches the Tkinter GUI via `main.py`.

## Architecture

**Face Studio** (Make-a-Smile v2) is a face drawing/generation studio built around **4 staged Conditional VAEs** that learn one face component at a time.

### Flow

1. **Draw** (`ui/drawer.py`): User draws 16×16 pixel face components stage-by-stage. Stage 1 draws heads; stages 2–4 draw eyes, smile, and details on top of locked base layers from the previous stage. Saves as paired CSV rows (base + target) per stage.
2. **Train** (`ui/menu.py`): Per-stage training buttons enable once minimum sample counts are met (30/60/40/40). Each stage trains its own model with KL annealing and denoising augmentation. Models saved to `data/model_stage{1-4}.pth`.
3. **Generate** (`ui/generator.py`): Pipeline runs stages sequentially: head → eyes → smile → details. Per-stage slider rows control latent z values. Only trained stages appear.
4. **Refine** (`ui/refine.py`): Per-component refinement — user rates faces, fixes specific layers, and mini-retrains that stage's model with 18 gradient steps.

### Key files

| File | Role |
|------|------|
| `config.py` | All constants: `GRID_SIZE=16`, per-stage z dims (4/3/3/3), file paths, training hyperparams |
| `model.py` | `HeadVAE` (~18k params), `ConditionalVAE` (~26k params), `staged_loss` with pixel weighting, `kl_beta_schedule` |
| `main.py` | Tkinter app shell; manages screen transitions between the four UI modules |
| `ui/` | One file per screen: `menu.py`, `drawer.py`, `generator.py`, `refine.py` |

### Model Architecture

- **Stage 1 (HeadVAE)**: Unconditional VAE — `256→32→16→(μ:4)` / `4→16→32→256`
- **Stages 2–4 (ConditionalVAE)**: Encoder `256→32→(μ:3)`, Decoder `(3+256)→32→256`
- **Loss**: BCE with 2× weight on new pixels + β·KLD with linear warmup (0→0.8 over epochs 100–400)

### Data Format

```
data/stage1_heads.csv          — 256 values per row (head images)
data/stage2_eyes_base.csv      — condition images (heads only)
data/stage2_eyes_target.csv    — target images (heads + eyes)
data/stage3_smile_base.csv     — condition images
data/stage3_smile_target.csv   — target images
data/stage4_detail_base.csv    — condition images
data/stage4_detail_target.csv  — target images
data/model_stage{1-4}.pth      — trained model weights
```
