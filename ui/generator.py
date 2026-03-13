import tkinter as tk
from tkinter import ttk
import torch
import torch.optim as optim
import csv
import random
import threading
import numpy as np
import os
from config import (
    GRID_SIZE, CELL_SIZE, RENDER_THRESHOLD,
    STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z,
    STAGE_NAMES, STAGE_ICONS, STAGE_FILES,
    REJECTION_SAMPLE_COUNT, TRAINING_LR,
    CONNECTIVITY_WEIGHT, SHARPNESS_WEIGHT,
)
from model import (
    HeadVAE, ConditionalVAE, score_structural_quality,
    flood_fill_gap_score, staged_loss,
)

# Design system colors
CLR_PRIMARY = "#3B82F6"
CLR_PRIMARY_HOVER = "#2563EB"
CLR_TEXT = "#111827"
CLR_TEXT_SECONDARY = "#6B7280"
CLR_BORDER = "#E5E7EB"
CLR_BG = "#FFFFFF"
CLR_BG_LIGHT = "#F9FAFB"
CLR_BG_HOVER = "#F3F4F6"
CLR_DRAW_PIXEL = "#1F2937"

# Per-stage z dimensions
STAGE_Z_DIMS = {1: STAGE1_Z, 2: STAGE2_Z, 3: STAGE3_Z, 4: STAGE4_Z}

# Display scaling
DISPLAY_SIZE = 480
PIXEL_SIZE = DISPLAY_SIZE // GRID_SIZE


class GeneratorUI(tk.Frame):
    """Face assembly studio with per-stage slider controls."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR_BG)
        self.controller = controller
        self.models = {}  # stage_num -> loaded model
        self.stage_z = {}  # stage_num -> list of z values
        self.slider_widgets = {}  # stage_num -> list of (slider, value_label)
        self._debounce_id = None
        self._generating = False
        self._training = False
        self._current_stage_imgs = {}  # stage -> tensor, last generated

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG_LIGHT, pady=16)
        header.pack(fill="x")

        tk.Button(
            header, text="← Menu", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG_LIGHT, relief="flat",
            padx=12, pady=4, cursor="hand2",
            command=self.controller.show_menu,
        ).pack(side="left", padx=12)

        tk.Label(
            header, text="Face Generator",
            font=("SF Pro", 24, "bold"), fg=CLR_TEXT, bg=CLR_BG_LIGHT,
        ).pack()
        tk.Label(
            header, text="Create variations using trained models",
            font=("SF Pro", 12), fg=CLR_TEXT_SECONDARY, bg=CLR_BG_LIGHT,
        ).pack(pady=(2, 0))

        tk.Frame(self, bg=CLR_BORDER, height=1).pack(fill="x")

        # ── Canvas display ──────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=CLR_BG, pady=12)
        canvas_frame.pack()

        self.canvas = tk.Canvas(
            canvas_frame, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
            bg=CLR_BG,
            highlightthickness=2, highlightbackground=CLR_PRIMARY,
        )
        self.canvas.pack()

        self.rects = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * PIXEL_SIZE, y * PIXEL_SIZE
                x2, y2 = x1 + PIXEL_SIZE, y1 + PIXEL_SIZE
                self.rects[y][x] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=CLR_BG, outline="",
                )

        # Empty state message
        self.empty_label = tk.Label(
            self, text="No models trained yet. Train Stage 1 first!",
            font=("SF Pro", 12), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )

        # ── Slider panel ────────────────────────────────────────
        self.slider_panel = tk.Frame(self, bg=CLR_BG, pady=8)
        self.slider_panel.pack(fill="x", padx=20)

        tk.Label(
            self.slider_panel, text="Stage Controls",
            font=("SF Pro", 13, "bold"), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(anchor="w")
        tk.Frame(self.slider_panel, bg=CLR_BORDER, height=1).pack(fill="x", pady=(4, 8))

        self.slider_container = tk.Frame(self.slider_panel, bg=CLR_BG)
        self.slider_container.pack(fill="x")

        # ── Bottom controls ─────────────────────────────────────
        btn_frame = tk.Frame(self, bg=CLR_BG, pady=12)
        btn_frame.pack()

        tk.Button(
            btn_frame, text="🎲 Surprise Me", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground=CLR_PRIMARY_HOVER,
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=8, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            command=self._randomize_all,
        ).pack(side="left", padx=8)

        self.fix_btn = tk.Button(
            btn_frame, text="🔧 Fix Gaps & Train", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg="#10B981", activebackground="#059669",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=8, cursor="hand2",
            highlightbackground="#10B981",
            command=self._fix_gaps_and_train,
        )
        self.fix_btn.pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="✨ Morph", font=("SF Pro", 12),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=16, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self._morph_animation,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="← Main Menu", font=("SF Pro", 12),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=16, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self.controller.show_menu,
        ).pack(side="left", padx=8)

        # Status label for auto-training feedback
        self.status_label = tk.Label(
            self, text="", font=("SF Pro", 11, "bold"),
            fg="#10B981", bg=CLR_BG,
        )
        self.status_label.pack(pady=2)

    # ── Model loading ───────────────────────────────────────────

    def load_model(self):
        """Load all available stage models and rebuild slider UI."""
        self.models = {}
        for stage in range(1, 5):
            _, _, model_path = STAGE_FILES[stage]
            if os.path.exists(model_path):
                try:
                    if stage == 1:
                        model = HeadVAE()
                    else:
                        model = ConditionalVAE(stage_name=f"stage{stage}")
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.eval()
                    self.models[stage] = model
                except Exception:
                    pass

        self._rebuild_sliders()

        if self.models:
            self.empty_label.pack_forget()
            self._randomize_all()
        else:
            self.empty_label.pack(pady=8)

    def _rebuild_sliders(self):
        """Rebuild the slider rows for all trained stages."""
        for child in self.slider_container.winfo_children():
            child.destroy()
        self.slider_widgets = {}
        self.stage_z = {}

        for stage in sorted(self.models.keys()):
            z_dim = STAGE_Z_DIMS[stage]
            name = STAGE_NAMES.get(stage, f"Stage {stage}")
            icon = STAGE_ICONS.get(stage, "")

            # Stage row frame
            row = tk.Frame(
                self.slider_container, bg=CLR_BG,
                highlightbackground=CLR_BORDER, highlightthickness=1,
                padx=12, pady=8,
            )
            row.pack(fill="x", pady=3)

            # Stage label
            tk.Label(
                row, text=f"{icon} {name.upper()}",
                font=("SF Pro", 11, "bold"), fg=CLR_TEXT, bg=CLR_BG,
                width=10, anchor="w",
            ).pack(side="left", padx=(0, 12))

            # Sliders
            sliders = []
            slider_frame = tk.Frame(row, bg=CLR_BG)
            slider_frame.pack(side="left", fill="x", expand=True)

            for i in range(z_dim):
                s_frame = tk.Frame(slider_frame, bg=CLR_BG)
                s_frame.pack(side="left", padx=8)

                tk.Label(
                    s_frame, text=f"z{i+1}",
                    font=("SF Pro", 10), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
                ).pack()

                slider = ttk.Scale(
                    s_frame, from_=-3.0, to=3.0,
                    orient=tk.HORIZONTAL, length=120,
                    command=lambda val, s=stage: self._on_slider_change(s),
                )
                slider.set(0.0)
                slider.pack()

                val_label = tk.Label(
                    s_frame, text="0.00",
                    font=("Courier", 10), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
                )
                val_label.pack()

                sliders.append((slider, val_label))

            # Refresh button
            tk.Button(
                row, text="🔄", font=("SF Pro", 12),
                fg=CLR_TEXT_SECONDARY, bg=CLR_BG_HOVER,
                relief="flat", padx=8, pady=4,
                command=lambda s=stage: self._randomize_stage(s),
            ).pack(side="right", padx=(8, 0))

            self.slider_widgets[stage] = sliders
            self.stage_z[stage] = [0.0] * z_dim

    # ── Slider handling ─────────────────────────────────────────

    def _on_slider_change(self, changed_stage):
        """Debounced handler for slider changes."""
        # Update value labels immediately
        for stage, sliders in self.slider_widgets.items():
            for slider, label in sliders:
                val = slider.get()
                label.config(text=f"{val:.2f}")

        # Debounce generation
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(100, self._generate_face)

    def _randomize_stage(self, stage):
        """Randomize sliders for one stage and regenerate from that stage only."""
        if stage not in self.slider_widgets:
            return
        if stage == 1 and stage in self.models:
            self._randomize_with_rejection(stage)
        else:
            for slider, label in self.slider_widgets[stage]:
                val = random.uniform(-2.0, 2.0)
                slider.set(val)
                label.config(text=f"{val:.2f}")
        # Regenerate from this stage only — lower layers are unchanged
        self._generate_from(stage)

    def _randomize_all(self):
        """Randomize all stage sliders and regenerate."""
        # Rejection-sample stage 1 first
        if 1 in self.slider_widgets and 1 in self.models:
            self._randomize_with_rejection(1)
        # Randomize stages 2+ normally
        for stage in self.slider_widgets:
            if stage == 1:
                continue
            for slider, label in self.slider_widgets[stage]:
                val = random.uniform(-2.0, 2.0)
                slider.set(val)
                label.config(text=f"{val:.2f}")
        self._generate_face()

    def _randomize_with_rejection(self, stage):
        """Generate N candidate z vectors and pick the structurally best one."""
        model = self.models[stage]
        z_dim = STAGE_Z_DIMS[stage]
        n = REJECTION_SAMPLE_COUNT

        with torch.no_grad():
            candidates = torch.randn(n, z_dim) * 1.5
            outputs = model.decode(candidates)
            scores = score_structural_quality(outputs)
            best_idx = scores.argmin().item()
            best_z = candidates[best_idx]

        for i, (slider, label) in enumerate(self.slider_widgets[stage]):
            val = best_z[i].item()
            slider.set(val)
            label.config(text=f"{val:.2f}")

    # ── Generation pipeline ─────────────────────────────────────

    def _generate_from(self, from_stage):
        """Regenerate from from_stage upward, reusing cached outputs below it."""
        if self._generating or not self.models:
            return
        self._generating = True

        def do_generate():
            try:
                # Seed current_img from the cached output just below from_stage
                current_img = None
                for s in sorted(self.models.keys()):
                    if s < from_stage:
                        current_img = self._current_stage_imgs.get(s, None)
                    else:
                        break

                stage_imgs = {s: v for s, v in self._current_stage_imgs.items()
                              if s < from_stage}

                for stage in sorted(self.models.keys()):
                    if stage < from_stage:
                        continue
                    z_values = [s.get() for s, _ in self.slider_widgets[stage]]
                    z_tensor = torch.tensor([z_values], dtype=torch.float32)
                    model = self.models[stage]

                    with torch.no_grad():
                        if stage == 1:
                            current_img = model.decode(z_tensor)
                        else:
                            if current_img is None:
                                break
                            condition = (current_img > RENDER_THRESHOLD).float()
                            raw = model.decode(z_tensor, condition)
                            current_img = torch.max(raw, condition)

                    stage_imgs[stage] = current_img.clone()

                self._current_stage_imgs = stage_imgs

                if current_img is not None:
                    img_np = current_img.view(GRID_SIZE, GRID_SIZE).numpy()
                    self.after(0, lambda: self._render_face(img_np))

            except Exception:
                pass
            finally:
                self._generating = False

        threading.Thread(target=do_generate, daemon=True).start()

    def _generate_face(self):
        """Run the staged generation pipeline."""
        if self._generating or not self.models:
            return

        self._generating = True

        def do_generate():
            try:
                current_img = None
                stage_imgs = {}

                for stage in sorted(self.models.keys()):
                    # Get z values from sliders
                    z_values = [s.get() for s, _ in self.slider_widgets[stage]]
                    z_tensor = torch.tensor([z_values], dtype=torch.float32)

                    model = self.models[stage]

                    with torch.no_grad():
                        if stage == 1:
                            current_img = model.decode(z_tensor)
                        else:
                            if current_img is None:
                                break
                            # Binarize the condition for cleaner conditioning
                            condition = (current_img > RENDER_THRESHOLD).float()
                            raw = model.decode(z_tensor, condition)
                            # Later stages can only ADD pixels, never erase the base
                            current_img = torch.max(raw, condition)

                    stage_imgs[stage] = current_img.clone()

                self._current_stage_imgs = stage_imgs

                if current_img is not None:
                    img_np = current_img.view(GRID_SIZE, GRID_SIZE).numpy()
                    self.after(0, lambda: self._render_face(img_np))

            except Exception:
                pass
            finally:
                self._generating = False

        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()

    def _render_face(self, img_array):
        """Render a 16×16 numpy array to the canvas."""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = CLR_DRAW_PIXEL if img_array[y][x] > RENDER_THRESHOLD else CLR_BG
                self.canvas.itemconfig(self.rects[y][x], fill=color)

    # ── Morph animation ─────────────────────────────────────────

    def _morph_animation(self):
        """Animate morphing from current face to a random target."""
        if not self.models:
            return

        # Capture start z values
        start_z = {}
        target_z = {}
        for stage in self.slider_widgets:
            start_z[stage] = [s.get() for s, _ in self.slider_widgets[stage]]
            target_z[stage] = [random.uniform(-2.0, 2.0) for _ in self.slider_widgets[stage]]

        steps = 15

        def step_morph(current_step):
            alpha = current_step / float(steps)
            for stage in self.slider_widgets:
                for i, (slider, label) in enumerate(self.slider_widgets[stage]):
                    new_val = start_z[stage][i] * (1 - alpha) + target_z[stage][i] * alpha
                    slider.set(new_val)
                    label.config(text=f"{new_val:.2f}")

            self._generate_face()

            if current_step < steps:
                self.after(60, lambda: step_morph(current_step + 1))

        step_morph(0)

    # ── Auto gap-fix and train ───────────────────────────────────

    @staticmethod
    def _fill_gaps_from_center(binary_np):
        """Fill 1-2 pixel gaps in a head outline.

        Flood fills from center to find the leak path, then from the border
        inward to find which leak-path pixels are closest to the outline.
        Those pixels at the narrowest point of the gap get filled.
        """
        gs = GRID_SIZE
        filled = binary_np.copy()
        center = gs // 2

        def flood_from(sy, sx, grid):
            """BFS from (sy,sx) through empty pixels. Returns (visited, dist)."""
            visited = np.zeros((gs, gs), dtype=bool)
            dist = np.full((gs, gs), -1, dtype=int)
            queue = [(sy, sx)]
            visited[sy, sx] = True
            dist[sy, sx] = 0
            while queue:
                y, x = queue.pop(0)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < gs and 0 <= nx < gs and not visited[ny,nx] and not grid[ny,nx]:
                        visited[ny,nx] = True
                        dist[ny,nx] = dist[y,x] + 1
                        queue.append((ny, nx))
            return visited, dist

        for _ in range(4):  # Max 4 fill attempts
            center_vis, center_dist = flood_from(center, center, filled)

            # Check for border leaks
            border_leaks = []
            for y in range(gs):
                for x in range(gs):
                    if (y == 0 or y == gs-1 or x == 0 or x == gs-1) and center_vis[y, x]:
                        border_leaks.append((y, x))
            if not border_leaks:
                break

            # BFS from all border leak pixels inward
            border_dist = np.full((gs, gs), -1, dtype=int)
            border_vis = np.zeros((gs, gs), dtype=bool)
            queue = []
            for by, bx in border_leaks:
                border_vis[by, bx] = True
                border_dist[by, bx] = 0
                queue.append((by, bx))
            while queue:
                y, x = queue.pop(0)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < gs and 0 <= nx < gs and not border_vis[ny,nx] and not filled[ny,nx]:
                        border_vis[ny,nx] = True
                        border_dist[ny,nx] = border_dist[y,x] + 1
                        queue.append((ny, nx))

            # The gap narrows where center_dist + border_dist is minimized.
            # Find empty pixels on the leak path (reachable from both center
            # and border) that are adjacent to outline pixels.
            best = None
            best_score = float('inf')
            for y in range(1, gs-1):
                for x in range(1, gs-1):
                    if filled[y, x] or center_dist[y, x] < 0 or border_dist[y, x] < 0:
                        continue
                    # Must be adjacent to at least 1 outline pixel
                    adj = sum(
                        1 for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                        if filled[y+dy, x+dx]
                    )
                    if adj < 1:
                        continue
                    # Score: total path length through this pixel (lower = narrower gap)
                    # Break ties by preferring more outline neighbors
                    path_len = center_dist[y, x] + border_dist[y, x]
                    score = (path_len, -adj)
                    if best is None or score < best_score:
                        best_score = score
                        best = (y, x)

            if best is None:
                break
            filled[best[0], best[1]] = 1

        return filled

    def _fix_gaps_and_train(self):
        """Auto-detect gaps in current stage-1 output, fill them, save as
        training data, and run a mini-retrain."""
        if 1 not in self.models or 1 not in self._current_stage_imgs:
            self.status_label.config(text="Generate a face first", fg="#EF4444")
            return
        if self._training:
            return

        stage1_img = self._current_stage_imgs[1]
        binary = (stage1_img > RENDER_THRESHOLD).float()
        binary_np = binary.view(GRID_SIZE, GRID_SIZE).numpy().astype(np.uint8)

        # Check if there are gaps
        gap_score = flood_fill_gap_score(stage1_img)
        if gap_score.item() == 0:
            self.status_label.config(text="No gaps detected!", fg=CLR_PRIMARY)
            self.after(2000, lambda: self.status_label.config(text=""))
            return

        # Fill the gaps
        fixed_np = self._fill_gaps_from_center(binary_np)
        fixed_flat = fixed_np.flatten().tolist()
        fixed_int = [int(v) for v in fixed_flat]

        # Save the corrected face to training data
        _, target_path, model_path = STAGE_FILES[1]
        try:
            with open(target_path, "a", newline="") as f:
                csv.writer(f).writerow(fixed_int)
            # Also save the horizontally flipped version
            flipped = np.fliplr(fixed_np).flatten().tolist()
            with open(target_path, "a", newline="") as f:
                csv.writer(f).writerow([int(v) for v in flipped])
        except Exception as e:
            self.status_label.config(text=f"Save failed: {e}", fg="#EF4444")
            return

        # Show the fixed version immediately
        self._current_stage_imgs[1] = torch.tensor(
            fixed_np.reshape(1, -1), dtype=torch.float32,
        )
        # Re-render with the fixed image visible at stage 1
        final_stage = max(self._current_stage_imgs.keys())
        if final_stage == 1:
            self._render_face(fixed_np.astype(float))
        else:
            # Regenerate stages 2+ with the fixed base
            self._generate_face()

        self.status_label.config(text="Gap fixed! Training...", fg="#10B981")
        self._training = True

        def do_auto_train():
            try:
                model = self.models[1]
                model.train()

                # Load all stage 1 training data
                all_targets = []
                _, target_file, _ = STAGE_FILES[1]
                if os.path.exists(target_file):
                    with open(target_file, "r") as f:
                        for row in csv.reader(f):
                            if len(row) == 256:
                                all_targets.append([float(v) for v in row])

                if not all_targets:
                    return

                target_tensor = torch.tensor(all_targets, dtype=torch.float32)
                base_tensor = torch.zeros_like(target_tensor)

                optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR * 0.3)
                train_steps = 40

                n_samples = target_tensor.shape[0]
                batch_size = min(32, n_samples)

                for step in range(1, train_steps + 1):
                    perm = torch.randperm(n_samples)
                    for start in range(0, n_samples, batch_size):
                        idx = perm[start:start + batch_size]
                        batch_t = target_tensor[idx]
                        batch_b = base_tensor[idx]

                        optimizer.zero_grad()
                        recon, mu, logvar = model(batch_t)
                        loss = staged_loss(
                            recon, batch_t, batch_b, mu, logvar, beta=0.1,
                            sharpness_weight=SHARPNESS_WEIGHT,
                            connectivity_weight=CONNECTIVITY_WEIGHT,
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    if step % 5 == 0:
                        try:
                            self.after(0, lambda s=step: self.status_label.config(
                                text=f"Training... {s}/{train_steps}",
                            ))
                        except Exception:
                            break

                model.eval()
                torch.save(model.state_dict(), model_path)
                self.models[1] = model

                try:
                    self.after(0, lambda: self.status_label.config(
                        text="Done! Generating improved face...",
                    ))
                    self.after(500, self._randomize_all)
                    self.after(2500, lambda: self.status_label.config(text=""))
                except Exception:
                    pass

            except Exception as e:
                try:
                    self.after(0, lambda: self.status_label.config(
                        text=f"Training error: {e}", fg="#EF4444",
                    ))
                except Exception:
                    pass
            finally:
                self._training = False

        thread = threading.Thread(target=do_auto_train, daemon=True)
        thread.start()