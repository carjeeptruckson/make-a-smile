import tkinter as tk
from tkinter import ttk
import torch
import torch.optim as optim
import csv
import os
import random
import threading
import numpy as np
from config import (
    GRID_SIZE, CELL_SIZE, RENDER_THRESHOLD,
    STAGE_NAMES, STAGE_ICONS, STAGE_FILES, STAGE_MIN_SAMPLES,
    STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z,
    TRAINING_LR,
)
from model import HeadVAE, ConditionalVAE, staged_loss

# Design system colors
CLR_PRIMARY = "#3B82F6"
CLR_SUCCESS = "#10B981"
CLR_WARNING = "#F59E0B"
CLR_DANGER = "#EF4444"
CLR_TEXT = "#111827"
CLR_TEXT_SECONDARY = "#6B7280"
CLR_TEXT_MUTED = "#9CA3AF"
CLR_BORDER = "#E5E7EB"
CLR_BG = "#FFFFFF"
CLR_BG_LIGHT = "#F9FAFB"
CLR_BG_HOVER = "#F3F4F6"
CLR_BASE_PIXEL = "#BFDBFE"
CLR_DRAW_PIXEL = "#1F2937"

STAGE_Z_DIMS = {1: STAGE1_Z, 2: STAGE2_Z, 3: STAGE3_Z, 4: STAGE4_Z}

# Display scaling
DISPLAY_SIZE = 400
PIXEL_SIZE = DISPLAY_SIZE // GRID_SIZE
REFINE_STEPS = 60


class RefineUI(tk.Frame):
    """Per-component refinement studio with rating and editing."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR_BG)
        self.controller = controller
        self.models = {}
        self.current_face_imgs = {}  # stage -> tensor
        self.current_stage_fixing = None
        self.can_draw = False
        self.face_count = 0
        self.saved_count = 0

        self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.base_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG, pady=12)
        header.pack(fill="x")

        tk.Label(
            header, text="Refine Your Face",
            font=("SF Pro", 24, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack()
        tk.Label(
            header, text="Rate and improve specific features",
            font=("SF Pro", 12), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(pady=(2, 0))

        tk.Frame(self, bg=CLR_BORDER, height=1).pack(fill="x")

        # ── Status bar ──────────────────────────────────────────
        self.status_frame = tk.Frame(self, bg=CLR_BG, pady=6)
        self.status_frame.pack(fill="x")

        self.face_counter_label = tk.Label(
            self.status_frame, text="",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        self.face_counter_label.pack(side="left", padx=20)

        self.saved_counter_label = tk.Label(
            self.status_frame, text="",
            font=("SF Pro", 11), fg=CLR_SUCCESS, bg=CLR_BG,
        )
        self.saved_counter_label.pack(side="right", padx=20)

        # ── Instruction ─────────────────────────────────────────
        self.instruction_label = tk.Label(
            self, text="How do you feel about this face?",
            font=("SF Pro", 13), fg=CLR_TEXT, bg=CLR_BG,
        )
        self.instruction_label.pack(pady=(8, 4))

        # ── Canvas ──────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=CLR_BG, pady=8)
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
                    x1, y1, x2, y2, fill=CLR_BG, outline="#E5E7EB",
                )
        self.canvas.config(cursor="crosshair")

        # Drawing binds
        self.canvas.bind("<Button-1>", lambda e: self._paint(e, 1))
        self.canvas.bind("<B1-Motion>", lambda e: self._paint(e, 1))
        self.canvas.bind("<Button-3>", lambda e: self._paint(e, 0))
        self.canvas.bind("<B3-Motion>", lambda e: self._paint(e, 0))
        self.canvas.bind("<Button-2>", lambda e: self._paint(e, 0))
        self.canvas.bind("<B2-Motion>", lambda e: self._paint(e, 0))

        # ── Rating buttons (Step 1) ─────────────────────────────
        self.rating_frame = tk.Frame(self, bg=CLR_BG, pady=8)
        self.rating_frame.pack()

        tk.Button(
            self.rating_frame, text="❤️ Love It", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg=CLR_SUCCESS, activebackground="#059669",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=10, cursor="hand2",
            highlightbackground=CLR_SUCCESS,
            command=lambda: self._rate("love"),
        ).pack(side="left", padx=6)

        tk.Button(
            self.rating_frame, text="😐 Okay", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground="#2563EB",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=10, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            command=lambda: self._rate("okay"),
        ).pack(side="left", padx=6)

        # Per-stage fix buttons — built dynamically
        self.fix_buttons_frame = tk.Frame(self.rating_frame, bg=CLR_BG)
        self.fix_buttons_frame.pack(side="left", padx=6)

        tk.Button(
            self.rating_frame, text="🚫 Skip", font=("SF Pro", 12),
            fg=CLR_TEXT, bg=CLR_BG_HOVER,
            relief="solid", bd=1, padx=14, pady=8, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=lambda: self._rate("skip"),
        ).pack(side="left", padx=6)

        # ── Edit controls (Step 2 — hidden initially) ───────────
        self.edit_frame = tk.Frame(self, bg=CLR_BG, pady=8)

        edit_buttons = tk.Frame(self.edit_frame, bg=CLR_BG)
        edit_buttons.pack()

        tk.Button(
            edit_buttons, text="Clear Layer", font=("SF Pro", 10),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=8, pady=4,
            command=self._clear_edit_layer,
        ).pack(side="left", padx=4)

        tk.Button(
            edit_buttons, text="✓ Save & Refine", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg=CLR_SUCCESS, activebackground="#059669",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=8, cursor="hand2",
            highlightbackground=CLR_SUCCESS,
            command=self._submit_correction,
        ).pack(side="left", padx=4)

        tk.Button(
            edit_buttons, text="← Cancel", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=6,
            command=self._cancel_edit,
        ).pack(side="left", padx=4)

        self.train_status_label = tk.Label(
            self.edit_frame, text="",
            font=("SF Pro", 11, "bold"), fg=CLR_PRIMARY, bg=CLR_BG,
        )
        self.train_status_label.pack(pady=4)

        # ── Notification ────────────────────────────────────────
        self.notif_label = tk.Label(
            self, text="", font=("SF Pro", 12, "bold"),
            fg=CLR_SUCCESS, bg=CLR_BG,
        )
        self.notif_label.pack(pady=4)

        # ── Bottom nav ──────────────────────────────────────────
        tk.Button(
            self, text="← Main Menu", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self.controller.show_menu,
        ).pack(pady=12)

        # Keyboard shortcuts
        self.bind_all("<Key-1>", lambda e: self._rate("love"))
        self.bind_all("<Key-2>", lambda e: self._rate("okay"))
        self.bind_all("<Key-3>", lambda e: self._rate("fix", 1))
        self.bind_all("<Key-4>", lambda e: self._rate("fix", 2))
        self.bind_all("<Key-5>", lambda e: self._rate("fix", 3))

    # ── Model loading ───────────────────────────────────────────

    def load_model(self):
        """Load all available stage models."""
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

        self._rebuild_fix_buttons()
        self.face_count = 0
        self.saved_count = 0
        self._generate_new_face()

    def _rebuild_fix_buttons(self):
        """Build fix buttons for each trained stage."""
        for child in self.fix_buttons_frame.winfo_children():
            child.destroy()

        for stage in sorted(self.models.keys()):
            name = STAGE_NAMES.get(stage, f"Stage {stage}")
            tk.Button(
                self.fix_buttons_frame,
                text=f"🔧 Fix {name}",
                font=("SF Pro", 11),
                fg=CLR_BG, bg=CLR_WARNING,
                activebackground="#D97706",
                relief="flat", padx=12, pady=8,
                command=lambda s=stage: self._rate("fix", s),
            ).pack(side="left", padx=3)

    # ── Face generation ─────────────────────────────────────────

    def _generate_new_face(self):
        """Generate a complete face through all trained stages."""
        self.can_draw = False
        self.current_stage_fixing = None
        self.face_count += 1

        # Show rating UI
        self.edit_frame.pack_forget()
        self.rating_frame.pack(pady=8)
        self.instruction_label.config(text="How do you feel about this face?")
        self._update_counters()

        self.current_face_imgs = {}
        current_img = None

        for stage in sorted(self.models.keys()):
            z_dim = STAGE_Z_DIMS[stage]
            z = torch.randn(1, z_dim)

            model = self.models[stage]
            with torch.no_grad():
                if stage == 1:
                    current_img = model.decode(z)
                else:
                    if current_img is None:
                        break
                    condition = (current_img > RENDER_THRESHOLD).float()
                    current_img = model.decode(z, condition)

            self.current_face_imgs[stage] = current_img.clone()

        if current_img is not None:
            img_np = current_img.view(GRID_SIZE, GRID_SIZE).numpy()
            self._render_face(img_np)

    def _render_face(self, img_array):
        """Render a 16×16 array to the canvas."""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = CLR_DRAW_PIXEL if img_array[y][x] > RENDER_THRESHOLD else CLR_BG
                self.canvas.itemconfig(self.rects[y][x], fill=color)

    def _update_counters(self):
        self.face_counter_label.config(text=f"Face #{self.face_count}")
        if self.saved_count > 0:
            self.saved_counter_label.config(
                text=f"✓ {self.saved_count} saved to training data"
            )

    # ── Rating ──────────────────────────────────────────────────

    def _rate(self, action, fix_stage=None):
        if action == "love":
            self.saved_count += 1
            self._save_all_layers()
            self._show_notification("❤️ Saved!", CLR_SUCCESS)
            self.after(500, self._generate_new_face)

        elif action == "okay":
            self._generate_new_face()

        elif action == "skip":
            self._generate_new_face()

        elif action == "fix" and fix_stage is not None:
            if fix_stage in self.models:
                self._enter_edit_mode(fix_stage)

    def _save_all_layers(self):
        """Save the current face's layers to training data for all stages."""
        for stage in sorted(self.current_face_imgs.keys()):
            base_path, target_path, _ = STAGE_FILES[stage]
            target_data = (self.current_face_imgs[stage] > RENDER_THRESHOLD).float()
            target_flat = target_data.view(-1).tolist()
            target_int = [int(v) for v in target_flat]

            try:
                if stage == 1:
                    with open(target_path, "a", newline="") as f:
                        csv.writer(f).writerow(target_int)
                else:
                    # Get base from previous stage
                    prev_stage = stage - 1
                    if prev_stage in self.current_face_imgs:
                        base_data = (self.current_face_imgs[prev_stage] > RENDER_THRESHOLD).float()
                        base_flat = [int(v) for v in base_data.view(-1).tolist()]
                    else:
                        base_flat = [0] * 256

                    if base_path:
                        with open(base_path, "a", newline="") as f:
                            csv.writer(f).writerow(base_flat)
                    with open(target_path, "a", newline="") as f:
                        csv.writer(f).writerow(target_int)
            except Exception:
                pass

    # ── Edit mode ───────────────────────────────────────────────

    def _enter_edit_mode(self, stage):
        """Switch to editing mode for a specific stage."""
        self.current_stage_fixing = stage
        self.can_draw = True

        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        self.instruction_label.config(
            text=f"Draw corrections for {name}. Base layer is locked (blue)."
        )

        # Show edit controls, hide rating
        self.rating_frame.pack_forget()
        self.edit_frame.pack(pady=8)
        self.train_status_label.config(text="")

        # Set up base layer = output of stage before the one we're fixing
        prev_stage = stage - 1
        if prev_stage >= 1 and prev_stage in self.current_face_imgs:
            base_img = (self.current_face_imgs[prev_stage] > RENDER_THRESHOLD).float()
        else:
            base_img = torch.zeros(1, 256)

        base_np = base_img.view(GRID_SIZE, GRID_SIZE).numpy()
        self.base_data = [[int(base_np[y][x] > RENDER_THRESHOLD) for x in range(GRID_SIZE)]
                          for y in range(GRID_SIZE)]

        # Pre-fill grid_data with the AI's current output for this stage
        # so the user can SEE and correct what the AI generated
        if stage in self.current_face_imgs:
            stage_img = (self.current_face_imgs[stage] > RENDER_THRESHOLD).float()
            stage_np = stage_img.view(GRID_SIZE, GRID_SIZE).numpy()
            self.grid_data = [
                [int(stage_np[y][x] > RENDER_THRESHOLD and self.base_data[y][x] == 0)
                 for x in range(GRID_SIZE)]
                for y in range(GRID_SIZE)
            ]
        else:
            self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        # Render: base in blue, AI's new pixels in dark gray, empty in white
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.base_data[y][x] == 1:
                    color = CLR_BASE_PIXEL
                elif self.grid_data[y][x] == 1:
                    color = CLR_DRAW_PIXEL
                else:
                    color = CLR_BG
                self.canvas.itemconfig(self.rects[y][x], fill=color)

    def _paint(self, event, val):
        """Paint on the canvas (only in edit mode)."""
        if not self.can_draw:
            return
        x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return
        # Don't modify base layer pixels
        if self.base_data[y][x] == 1:
            return

        self.grid_data[y][x] = val
        if val == 1:
            color = CLR_DRAW_PIXEL
        else:
            color = CLR_BG
        self.canvas.itemconfig(self.rects[y][x], fill=color)

    def _clear_edit_layer(self):
        """Clear just the edit layer (keep base)."""
        self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.base_data[y][x] != 1:
                    self.canvas.itemconfig(self.rects[y][x], fill=CLR_BG)

    def _cancel_edit(self):
        """Cancel editing and return to rating view."""
        self.can_draw = False
        self.current_stage_fixing = None

        # Re-render the original face
        max_stage = max(self.current_face_imgs.keys())
        if max_stage in self.current_face_imgs:
            img_np = self.current_face_imgs[max_stage].view(GRID_SIZE, GRID_SIZE).numpy()
            self._render_face(img_np)

        self.edit_frame.pack_forget()
        self.rating_frame.pack(pady=8)
        self.instruction_label.config(text="How do you feel about this face?")

    def _submit_correction(self):
        """Save correction and run mini-retrain."""
        stage = self.current_stage_fixing
        if stage is None:
            return

        # Build target = base + current drawing
        target_flat = []
        base_flat = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                target_val = 1 if (self.base_data[y][x] == 1 or self.grid_data[y][x] == 1) else 0
                target_flat.append(target_val)
                base_flat.append(self.base_data[y][x])

        drawn = sum(self.grid_data[y][x] for y in range(GRID_SIZE) for x in range(GRID_SIZE))
        if drawn == 0:
            self._show_notification("⚠ Draw something first!", CLR_DANGER)
            return

        # Save to training data
        base_path, target_path, model_path = STAGE_FILES[stage]
        try:
            if stage == 1:
                with open(target_path, "a", newline="") as f:
                    csv.writer(f).writerow(target_flat)
            else:
                if base_path:
                    with open(base_path, "a", newline="") as f:
                        csv.writer(f).writerow(base_flat)
                with open(target_path, "a", newline="") as f:
                    csv.writer(f).writerow(target_flat)
        except Exception as e:
            self._show_notification(f"⚠ Save failed: {e}", CLR_DANGER)
            return

        self.saved_count += 1
        self._update_counters()

        # Mini-retrain in background
        self.can_draw = False
        self.train_status_label.config(text=f"Training... 0/{REFINE_STEPS} steps")

        def do_retrain():
            try:
                model = self.models[stage]
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR * 0.5)

                # Load ALL training data for this stage (not just the single correction)
                all_targets = []
                all_bases = []
                base_file, target_file, _ = STAGE_FILES[stage]

                if os.path.exists(target_file):
                    with open(target_file, "r") as f:
                        for row in csv.reader(f):
                            if len(row) == 256:
                                all_targets.append([float(v) for v in row])

                if stage > 1 and base_file and os.path.exists(base_file):
                    with open(base_file, "r") as f:
                        for row in csv.reader(f):
                            if len(row) == 256:
                                all_bases.append([float(v) for v in row])

                if not all_targets:
                    # Fallback: just use the single correction
                    all_targets = [target_flat]
                    all_bases = [base_flat]

                target_tensor = torch.tensor(all_targets, dtype=torch.float32)
                if stage == 1:
                    base_tensor = torch.zeros_like(target_tensor)
                elif all_bases:
                    base_tensor = torch.tensor(all_bases, dtype=torch.float32)
                else:
                    base_tensor = torch.zeros_like(target_tensor)

                for step in range(1, REFINE_STEPS + 1):
                    optimizer.zero_grad()
                    if stage == 1:
                        recon, mu, logvar = model(target_tensor)
                    else:
                        recon, mu, logvar = model(target_tensor, base_tensor)

                    loss = staged_loss(recon, target_tensor, base_tensor, mu, logvar, beta=0.3)
                    loss.backward()
                    optimizer.step()

                    if step % 5 == 0:
                        try:
                            self.after(0, lambda s=step: self.train_status_label.config(
                                text=f"Refining on {len(all_targets)} samples... {s}/{REFINE_STEPS}"
                            ))
                        except Exception:
                            break

                model.eval()
                torch.save(model.state_dict(), model_path)

                # Regenerate from the fixed stage onward
                self.after(0, self._regenerate_from_stage, stage)

            except Exception as e:
                try:
                    self.after(0, lambda: self._show_notification(f"⚠ Training failed: {e}", CLR_DANGER))
                except Exception:
                    pass

        thread = threading.Thread(target=do_retrain, daemon=True)
        thread.start()

    def _regenerate_from_stage(self, from_stage):
        """Regenerate the face from a specific stage onward."""
        current_img = None

        # Use existing images up to from_stage - 1
        for stage in sorted(self.current_face_imgs.keys()):
            if stage < from_stage:
                current_img = self.current_face_imgs[stage]
            else:
                break

        # Regenerate from from_stage onward
        for stage in sorted(self.models.keys()):
            if stage < from_stage:
                continue

            z_dim = STAGE_Z_DIMS[stage]
            z = torch.randn(1, z_dim)
            model = self.models[stage]

            with torch.no_grad():
                if stage == 1:
                    current_img = model.decode(z)
                else:
                    if current_img is None:
                        break
                    condition = (current_img > RENDER_THRESHOLD).float()
                    current_img = model.decode(z, condition)

            self.current_face_imgs[stage] = current_img.clone()

        if current_img is not None:
            img_np = current_img.view(GRID_SIZE, GRID_SIZE).numpy()
            self._render_face(img_np)

        # Return to rating view
        self.can_draw = False
        self.current_stage_fixing = None
        self.edit_frame.pack_forget()
        self.rating_frame.pack(pady=8)
        self.instruction_label.config(text="How do you feel about this face?")
        self.train_status_label.config(text="")
        self._show_notification("✓ Refined!", CLR_SUCCESS)

    # ── Helpers ─────────────────────────────────────────────────

    def _show_notification(self, text, color):
        self.notif_label.config(text=text, fg=color)
        self.after(2000, lambda: self.notif_label.config(text=""))