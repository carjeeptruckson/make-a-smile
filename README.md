# Face Studio

Face Studio is a creative AI that learns to draw sketchy faces from your hand-drawn sketches, one layer at a time. Draw heads, eyes, smiles, and details separately — each stage trains a small neural network that understands that specific part of the face.

## Quick Start

```bash
./run.sh
```

This sets up a Python virtual environment, installs dependencies (PyTorch, NumPy), and launches the app.

## Workflow Overview

```
1. Draw Stage 1: Head Shapes (30+ sketches)
   ↓
2. Train Stage 1 Model
   ↓
3. Draw Stage 2: Add Eyes (60+ pairs on your trained heads)
   ↓
4. Train Stage 2 Model
   ↓
5. Draw Stage 3: Add Smiles (40+ pairs)
   ↓
6. Train Stage 3 Model
   ↓
7. Generate & Refine: Create new faces, fix individual layers
```

Each stage builds on the last. You can start generating faces after training just Stage 1!

## How It Works

### 🎨 Drawing Studio
Draw one component at a time on a 16×16 pixel canvas. For stages 2+, a base layer from the previous stage is shown in light blue and locked — you draw your new component on top.

### 🧠 Training
Each stage trains a small AI model (18,000–26,000 parameters) specific to that face component. Training runs for 500 epochs with intelligent difficulty ramping (KL annealing). Hit the "Train" button once you've collected enough samples.

### ✨ Generator
Explore face variations by adjusting sliders for each trained layer. Each row controls one component — keep a head you like and try different eye combinations, or lock in eyes and explore smiles.

### 🛠 Refine Studio
Rate generated faces: Love It, Okay, or Fix a specific layer. When fixing, you draw a correction on the locked base layer, and the AI runs a quick 18-step retraining to learn from your feedback.

## Architecture

The app uses **4 staged Conditional VAEs** that each learn one layer of the face:

| Stage | Component | Type | Latent Dim | Params |
|-------|-----------|------|------------|--------|
| 1 | Head Shape | VAE | 4 | ~18,000 |
| 2 | Eyes | CVAE | 3 | ~26,000 |
| 3 | Smile | CVAE | 3 | ~26,000 |
| 4 | Details | CVAE | 3 | ~26,000 |

Why this works better: smaller, focused models learn better than one big model trying to do everything. Each CVAE conditions on the accumulated face so far, learning "where do eyes go on *this* head?"

## File Structure

```
data/
  stage1_heads.csv              (your head drawings)
  stage2_eyes_base.csv          (heads used as base for eyes)
  stage2_eyes_target.csv        (heads + eyes combined)
  stage3_smile_base.csv         (heads+eyes used as base)
  stage3_smile_target.csv       (heads+eyes+smile combined)
  stage4_detail_base.csv        (base for details)
  stage4_detail_target.csv      (complete faces)
  model_stage1.pth              (trained AI for heads)
  model_stage2.pth              (trained AI for eyes)
  model_stage3.pth              (trained AI for smiles)
  model_stage4.pth              (trained AI for details)
ui/
  drawer.py                     (drawing interface)
  menu.py                       (dashboard & training)
  generator.py                  (face generation)
  refine.py                     (feedback & improvement)
config.py                       (settings & constants)
model.py                        (neural network code)
main.py                         (app controller)
```

## Tips & Tricks

- **Varied heads**: Draw different sizes and shapes — round, oval, wide, tall, lumpy. Variety teaches the AI more.
- **Multiple variations**: For eyes/smile stages, use the same base head but draw different eye arrangements each time. This teaches the AI that different variations are valid.
- **When to move on**: Once you have at least the minimum samples (shown in the dashboard), train and move to the next stage. More data = better results.
- **Sliders**: Each slider controls a "dimension of variation" the AI discovered. Slide slowly to see how faces change smoothly.
- **Morph**: Click Morph to see a smooth animation from your current face to a random new one.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Training takes too long | Normal — typically 2–5 minutes per stage. The progress bar shows current epoch. |
| Generated faces look bad | Collect more training data and retrain. Quality improves with variety. |
| Can't train Stage 2 | Train Stage 1 first — each stage requires the previous one. |
| App won't launch | Ensure Python 3.13, `torch`, and `numpy` are installed. Run `./run.sh`. |
| Stage button grayed out | You need more samples before training. Check the dashboard for counts. |
