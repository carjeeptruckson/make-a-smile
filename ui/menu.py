import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import threading
import torch
import torch.optim as optim
import numpy as np
from config import (
    GRID_SIZE, DATA_DIR,
    STAGE_NAMES, STAGE_ICONS, STAGE_FILES, STAGE_MIN_SAMPLES,
    STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z,
    KL_WARMUP_START, KL_WARMUP_END, KL_FINAL_BETA,
    TRAINING_EPOCHS, TRAINING_LR, NOISE_FACTOR, SHARPNESS_WEIGHT,
    CONNECTIVITY_WEIGHT, CONNECTIVITY_WARMUP_START, CONNECTIVITY_WARMUP_END,
)
from model import HeadVAE, ConditionalVAE, staged_loss, kl_beta_schedule, add_noise

# Design system colors
CLR_PRIMARY = "#3B82F6"
CLR_PRIMARY_HOVER = "#2563EB"
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


class MainMenu(tk.Frame):
    """Dashboard-style main menu with per-stage progress and training."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR_BG)
        self.controller = controller
        self._training = False

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG, pady=20)
        header.pack(fill="x")

        tk.Label(
            header, text="Face Studio",
            font=("SF Pro", 28, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack()
        tk.Label(
            header, text="Learn to draw with AI",
            font=("SF Pro", 14), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(pady=(2, 0))

        tk.Frame(self, bg=CLR_BORDER, height=1).pack(fill="x")

        # ── Main content ────────────────────────────────────────
        content = tk.Frame(self, bg=CLR_BG)
        content.pack(fill="both", expand=True, padx=20, pady=16)

        # Left sidebar: Training Progress
        sidebar = tk.Frame(content, bg=CLR_BG_LIGHT, width=320, padx=16, pady=16)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Training Progress",
            font=("SF Pro", 14, "bold"), fg=CLR_TEXT, bg=CLR_BG_LIGHT,
        ).pack(anchor="w", pady=(0, 12))

        self.stage_cards = {}
        for stage in range(1, 5):
            self._build_stage_card(sidebar, stage)

        # Right area: Actions
        right = tk.Frame(content, bg=CLR_BG)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(
            right, text="Your Creative Journey",
            font=("SF Pro", 14, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack(anchor="w", pady=(0, 12))

        cards_frame = tk.Frame(right, bg=CLR_BG)
        cards_frame.pack(fill="both", expand=True)

        # 2×2 grid of action cards
        for i, (stage, info) in enumerate([
            (1, {"title": "📝 Draw Heads", "desc": "Draw head shapes to teach the AI", "action": "draw"}),
            (2, {"title": "📝 Draw Eyes", "desc": "Add eyes to head shapes", "action": "draw"}),
            (3, {"title": "📝 Draw Smiles", "desc": "Add smiles to head+eyes", "action": "draw"}),
            (4, {"title": "📝 Draw Details", "desc": "Add final details to faces", "action": "draw"}),
        ]):
            row, col = divmod(i, 2)
            card = self._build_action_card(cards_frame, stage, info)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)
        cards_frame.grid_rowconfigure(0, weight=1)
        cards_frame.grid_rowconfigure(1, weight=1)

        # ── Generate & Refine bar ──────────────────────────────────
        gen_bar = tk.Frame(right, bg=CLR_BG_LIGHT, padx=16, pady=12,
                           highlightbackground=CLR_BORDER, highlightthickness=1)
        gen_bar.pack(fill="x", pady=(12, 0))

        tk.Label(
            gen_bar, text="✨ AI Generation",
            font=("SF Pro", 12, "bold"), fg=CLR_TEXT, bg=CLR_BG_LIGHT,
        ).pack(side="left")

        self._gen_bar_refine_btn = tk.Button(
            gen_bar, text="🛠 Refine", font=("SF Pro", 10),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=10, pady=4, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self.controller.show_refine,
        )
        self._gen_bar_refine_btn.pack(side="right", padx=(8, 0))

        self._gen_bar_btn = tk.Button(
            gen_bar, text="Generate Faces", font=("SF Pro", 11, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground=CLR_PRIMARY_HOVER,
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=14, pady=6, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            command=self.controller.show_generator,
        )
        self._gen_bar_btn.pack(side="right")

    def _build_stage_card(self, parent, stage):
        """Build a progress card for a single stage."""
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        icon = STAGE_ICONS.get(stage, "")
        minimum = STAGE_MIN_SAMPLES.get(stage, 30)

        card = tk.Frame(parent, bg=CLR_BG, padx=12, pady=10,
                        highlightbackground=CLR_BORDER, highlightthickness=1)
        card.pack(fill="x", pady=4)

        # Title
        tk.Label(
            card, text=f"{icon} Stage {stage}: {name}",
            font=("SF Pro", 11, "bold"), fg=CLR_TEXT, bg=CLR_BG,
            anchor="w",
        ).pack(fill="x")

        # Status indicator — on its own row so it never clips
        status_indicator = tk.Label(
            card, text="● Locked", font=("SF Pro", 10),
            fg=CLR_TEXT_MUTED, bg=CLR_BG, anchor="w",
        )
        status_indicator.pack(fill="x", pady=(2, 4))

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            card, variable=progress_var, maximum=minimum,
        )
        progress_bar.pack(fill="x", pady=(0, 2))

        # Status text + train button on the same row
        bottom_row = tk.Frame(card, bg=CLR_BG)
        bottom_row.pack(fill="x", pady=(2, 0))

        status_text = tk.Label(
            bottom_row, text=f"0 / {minimum} samples",
            font=("SF Pro", 10), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        status_text.pack(side="left")

        train_btn = tk.Button(
            bottom_row, text="Train", font=("SF Pro", 10, "bold"),
            fg=CLR_BG, bg=CLR_SUCCESS, activebackground="#059669",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=12, pady=3, cursor="hand2",
            highlightbackground=CLR_SUCCESS,
            command=lambda s=stage: self._train_stage(s),
        )
        train_btn.pack(side="right")

        # Preview / Refine buttons (shown when model is trained)
        extras_row = tk.Frame(card, bg=CLR_BG)
        extras_row.pack(fill="x", pady=(4, 0))

        preview_btn = tk.Button(
            extras_row, text="🔍 Preview", font=("SF Pro", 9),
            fg=CLR_TEXT_SECONDARY, bg=CLR_BG_HOVER,
            relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=lambda s=stage: self._preview_stage(s),
        )
        preview_btn.pack(side="left", padx=(0, 4))

        refine_btn = tk.Button(
            extras_row, text="🛠 Refine", font=("SF Pro", 9),
            fg=CLR_TEXT_SECONDARY, bg=CLR_BG_HOVER,
            relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=lambda s=stage: self._refine_stage(s),
        )
        refine_btn.pack(side="left", padx=(0, 4))

        data_btn = tk.Button(
            extras_row, text="📋 View Data", font=("SF Pro", 9),
            fg=CLR_TEXT_SECONDARY, bg=CLR_BG_HOVER,
            relief="solid", bd=1, padx=6, pady=2, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=lambda s=stage: self._view_data(s),
        )
        data_btn.pack(side="left")

        self.stage_cards[stage] = {
            "progress_var": progress_var,
            "status_text": status_text,
            "status_indicator": status_indicator,
            "train_btn": train_btn,
            "preview_btn": preview_btn,
            "refine_btn": refine_btn,
            "data_btn": data_btn,
        }

    def _build_action_card(self, parent, stage, info):
        """Build an action card for drawing or generating."""
        card = tk.Frame(
            parent, bg=CLR_BG, padx=16, pady=16,
            highlightbackground=CLR_BORDER, highlightthickness=1,
        )

        tk.Label(
            card, text=info["title"],
            font=("SF Pro", 13, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack(anchor="w")

        tk.Label(
            card, text=info["desc"],
            font=("SF Pro", 10), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
            wraplength=220, justify="left",
        ).pack(anchor="w", pady=(4, 8))

        minimum = STAGE_MIN_SAMPLES.get(stage, 30)
        progress_label = tk.Label(
            card, text=f"0/{minimum} samples collected",
            font=("SF Pro", 10), fg=CLR_TEXT_MUTED, bg=CLR_BG,
        )
        progress_label.pack(anchor="w", pady=(0, 8))

        btn = tk.Button(
            card, text="Continue Drawing", font=("SF Pro", 11, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground=CLR_PRIMARY_HOVER,
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=14, pady=6, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            command=lambda s=stage: self._open_drawer(s),
        )
        btn.pack(anchor="w")

        # Store references for updating
        card._progress_label = progress_label
        card._action_btn = btn
        card._stage = stage
        card._info = info
        return card

    # ── State queries ───────────────────────────────────────────

    def _get_stage_sample_count(self, stage):
        """Count rows in a stage's target data file."""
        _, target_path, _ = STAGE_FILES[stage]
        if target_path is None:
            return 0
        if not os.path.exists(target_path):
            return 0
        try:
            with open(target_path, "r") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _stage_model_exists(self, stage):
        """Check if a trained model exists for this stage."""
        _, _, model_path = STAGE_FILES[stage]
        return os.path.exists(model_path)

    def _get_stage_status(self, stage):
        """Returns 'complete', 'ready', or 'locked'."""
        if self._stage_model_exists(stage):
            return "complete"
        # Stage 1 is always unlocked
        if stage == 1:
            return "ready"
        # Stages 2+ require previous stage model
        if self._stage_model_exists(stage - 1):
            return "ready"
        return "locked"

    # ── UI updates ──────────────────────────────────────────────

    def update_stats(self):
        """Refresh all progress indicators and button states."""
        for stage in range(1, 5):
            card = self.stage_cards[stage]
            count = self._get_stage_sample_count(stage)
            minimum = STAGE_MIN_SAMPLES.get(stage, 30)
            status = self._get_stage_status(stage)

            card["progress_var"].set(min(count, minimum))
            card["status_text"].config(text=f"{count} / {minimum} samples")

            # Status indicator color
            if status == "complete":
                card["status_indicator"].config(fg=CLR_SUCCESS, text="● Trained")
            elif status == "ready":
                card["status_indicator"].config(fg=CLR_WARNING, text="● Ready")
            else:
                card["status_indicator"].config(fg=CLR_TEXT_MUTED, text="● Locked")

            # Train button state
            can_train = (status in ("ready", "complete")) and count >= minimum
            card["train_btn"].config(
                state="normal" if can_train else "disabled",
                bg=CLR_SUCCESS if can_train else CLR_BG_HOVER,
                fg=CLR_BG if can_train else CLR_TEXT_MUTED,
            )

            # Preview / Refine buttons: only if model trained
            has_model = self._stage_model_exists(stage)
            for btn_key in ("preview_btn", "refine_btn"):
                card[btn_key].config(
                    state="normal" if has_model else "disabled",
                    fg=CLR_TEXT_SECONDARY if has_model else CLR_TEXT_MUTED,
                    bg=CLR_BG_HOVER if has_model else CLR_BG,
                )

            # View Data button: enabled when there's data
            has_data = count > 0
            card["data_btn"].config(
                state="normal" if has_data else "disabled",
                fg=CLR_TEXT_SECONDARY if has_data else CLR_TEXT_MUTED,
                bg=CLR_BG_HOVER if has_data else CLR_BG,
            )

        # Update action cards
        for child in self.winfo_children():
            self._update_action_cards_recursive(child)

        # Update generate bar buttons
        has_any_model = any(self._stage_model_exists(s) for s in range(1, 5))
        self._gen_bar_btn.config(
            state="normal" if has_any_model else "disabled",
            bg=CLR_PRIMARY if has_any_model else CLR_BG_HOVER,
            fg=CLR_BG if has_any_model else CLR_TEXT_MUTED,
        )
        self._gen_bar_refine_btn.config(
            state="normal" if has_any_model else "disabled",
        )

    def _update_action_cards_recursive(self, widget):
        """Walk widget tree to find and update action cards."""
        if hasattr(widget, '_info'):
            stage = widget._stage
            count = self._get_stage_sample_count(stage)
            minimum = STAGE_MIN_SAMPLES.get(stage, 30)
            status = self._get_stage_status(stage)
            widget._progress_label.config(text=f"{count}/{minimum} samples collected")

            enabled = status in ("ready", "complete")
            widget._action_btn.config(
                state="normal" if enabled else "disabled",
                bg=CLR_PRIMARY if enabled else CLR_BG_HOVER,
                fg=CLR_BG if enabled else CLR_TEXT_MUTED,
            )

        for child in widget.winfo_children():
            self._update_action_cards_recursive(child)

    # ── Navigation ──────────────────────────────────────────────

    def _open_drawer(self, stage):
        """Open the drawing studio for a specific stage."""
        status = self._get_stage_status(stage)
        if status == "locked":
            messagebox.showinfo(
                "Stage Locked",
                f"Train Stage {stage - 1} first before drawing Stage {stage}."
            )
            return
        self.controller.show_drawer(stage)

    def _preview_stage(self, stage):
        """Open the generator with only this stage's model loaded for preview."""
        if not self._stage_model_exists(stage):
            messagebox.showinfo("No Model", f"Train Stage {stage} first.")
            return
        self.controller.show_generator()

    def _refine_stage(self, stage):
        """Open the refine studio focused on a specific stage."""
        if not self._stage_model_exists(stage):
            messagebox.showinfo("No Model", f"Train Stage {stage} first.")
            return
        self.controller.show_refine()

    def _view_data(self, stage):
        """Open the data browser for a specific stage."""
        count = self._get_stage_sample_count(stage)
        if count == 0:
            messagebox.showinfo("No Data", f"No training data for Stage {stage} yet.\nDraw some samples first!")
            return
        self.controller.show_browser(stage)

    # ── Data curation ────────────────────────────────────────────

    @staticmethod
    def _curate_training_data(targets, bases, stage):
        """Filter out bad training samples before training.

        Returns (filtered_targets, filtered_bases, n_removed).
        """
        if stage == 1:
            keep = []
            for i, row in enumerate(targets):
                arr = np.array(row).reshape(GRID_SIZE, GRID_SIZE)
                filled = (arr > 0.5).sum()
                # Reject nearly empty or nearly solid
                if filled < 15 or filled > 180:
                    continue
                # Check for stray pixels: count neighbors for each filled pixel
                padded = np.pad(arr > 0.5, 1, mode='constant').astype(float)
                neighbor_count = sum(
                    np.roll(np.roll(padded, dy, 0), dx, 1)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy, dx) != (0, 0)
                )[1:-1, 1:-1]
                filled_mask = arr > 0.5
                if filled_mask.sum() > 0:
                    stray_frac = ((neighbor_count < 2) & filled_mask).sum() / filled_mask.sum()
                    if stray_frac > 0.3:
                        continue
                keep.append(i)
        else:
            # Stages 2-4: only reject if literally nothing new was drawn
            keep = []
            for i, row in enumerate(targets):
                if bases and i < len(bases):
                    new_pixels = sum(1 for t, b in zip(row, bases[i]) if t > 0.5 and b < 0.5)
                    if new_pixels < 1:
                        continue
                keep.append(i)

        # Don't curate if we'd lose too much data
        if len(keep) < len(targets) * 0.7:
            return targets, bases, 0

        n_removed = len(targets) - len(keep)
        filtered_targets = [targets[i] for i in keep]
        filtered_bases = [bases[i] for i in keep] if bases else []
        return filtered_targets, filtered_bases, n_removed

    # ── Training ────────────────────────────────────────────────

    def _train_stage(self, stage):
        """Train a specific stage's model."""
        count = self._get_stage_sample_count(stage)
        minimum = STAGE_MIN_SAMPLES.get(stage, 30)

        if count < minimum:
            messagebox.showwarning(
                "Not Enough Data",
                f"Stage {stage} needs at least {minimum} samples.\n"
                f"You have {count}. Keep drawing!"
            )
            return

        if self._training:
            messagebox.showinfo("Training", "A training session is already in progress.")
            return

        self._training = True
        name = STAGE_NAMES.get(stage, f"Stage {stage}")

        # ── Training modal ──────────────────────────────────────
        modal = tk.Toplevel(self)
        modal.title(f"Training Stage {stage}: {name}")
        modal.geometry("500x400")
        modal.configure(bg=CLR_BG)
        modal.transient(self.winfo_toplevel())
        modal.grab_set()

        tk.Label(
            modal, text=f"Training Stage {stage}: {name}",
            font=("SF Pro", 16, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack(pady=(20, 4))

        tk.Label(
            modal, text=f"Training on {count} samples (×2 with augmentation)",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(pady=(0, 16))

        # Large epoch display
        epoch_label = tk.Label(
            modal, text="Epoch 0 / 500",
            font=("SF Pro", 32, "bold"), fg=CLR_PRIMARY, bg=CLR_BG,
        )
        epoch_label.pack(pady=8)

        # Loss display
        loss_label = tk.Label(
            modal, text="Loss: —",
            font=("SF Pro", 13), fg=CLR_TEXT, bg=CLR_BG,
        )
        loss_label.pack()

        beta_label = tk.Label(
            modal, text="KL β: 0.000",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        beta_label.pack(pady=(4, 16))

        progress_var = tk.DoubleVar()
        ttk.Progressbar(
            modal, variable=progress_var, maximum=TRAINING_EPOCHS, length=400,
        ).pack(pady=4)

        cancel_var = {"cancelled": False}

        cancel_btn = tk.Button(
            modal, text="Cancel", font=("SF Pro", 11, "bold"),
            fg=CLR_BG, bg=CLR_DANGER, activebackground="#DC2626",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=6, cursor="hand2",
            highlightbackground=CLR_DANGER,
            command=lambda: cancel_var.update({"cancelled": True}),
        )
        cancel_btn.pack(pady=12)

        status_label = tk.Label(
            modal, text="", font=("SF Pro", 12, "bold"),
            fg=CLR_SUCCESS, bg=CLR_BG,
        )
        status_label.pack()

        # ── Run training in background thread ───────────────────
        def do_train():
            try:
                base_path, target_path, model_path = STAGE_FILES[stage]

                # Load data
                targets = []
                bases = []

                with open(target_path, "r") as f:
                    for row in csv.reader(f):
                        if len(row) == 256:
                            targets.append([float(v) for v in row])

                if stage > 1 and base_path and os.path.exists(base_path):
                    with open(base_path, "r") as f:
                        for row in csv.reader(f):
                            if len(row) == 256:
                                bases.append([float(v) for v in row])

                # Curate: remove bad samples
                targets, bases, n_removed = self._curate_training_data(
                    targets, bases, stage,
                )
                if n_removed > 0:
                    try:
                        modal.after(0, lambda n=n_removed: status_label.config(
                            text=f"Cleaned: removed {n} bad sample(s)",
                            fg=CLR_WARNING,
                        ))
                    except Exception:
                        pass

                # Augmentation: horizontal flip (doubles data)
                aug_targets = []
                aug_bases = []
                for i, t in enumerate(targets):
                    aug_targets.append(t)
                    grid = np.array(t).reshape(GRID_SIZE, GRID_SIZE)
                    aug_targets.append(np.fliplr(grid).flatten().tolist())

                    if bases:
                        b = bases[i] if i < len(bases) else [0.0] * 256
                        aug_bases.append(b)
                        bgrid = np.array(b).reshape(GRID_SIZE, GRID_SIZE)
                        aug_bases.append(np.fliplr(bgrid).flatten().tolist())

                target_tensor = torch.tensor(aug_targets, dtype=torch.float32)

                if stage == 1:
                    base_tensor = torch.zeros_like(target_tensor)
                else:
                    if aug_bases:
                        base_tensor = torch.tensor(aug_bases, dtype=torch.float32)
                    else:
                        base_tensor = torch.zeros_like(target_tensor)

                # Create model
                if stage == 1:
                    model = HeadVAE()
                else:
                    model = ConditionalVAE(stage_name=f"stage{stage}")

                # Backup existing model
                if os.path.exists(model_path):
                    backup = model_path + ".bak"
                    try:
                        import shutil
                        shutil.copy2(model_path, backup)
                    except Exception:
                        pass

                optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=TRAINING_EPOCHS, eta_min=TRAINING_LR * 0.05,
                )

                n_samples = target_tensor.shape[0]
                batch_size = min(32, n_samples)

                for epoch in range(1, TRAINING_EPOCHS + 1):
                    if cancel_var["cancelled"]:
                        break

                    model.train()

                    # Shuffle data each epoch
                    perm = torch.randperm(n_samples)
                    epoch_loss = 0.0
                    n_batches = 0

                    for start in range(0, n_samples, batch_size):
                        idx = perm[start:start + batch_size]
                        batch_target = target_tensor[idx]
                        batch_base = base_tensor[idx]

                        optimizer.zero_grad()

                        # Denoising augmentation
                        noisy_target = add_noise(batch_target, noise_factor=NOISE_FACTOR)

                        # Forward pass
                        if stage == 1:
                            recon, mu, logvar = model(noisy_target)
                        else:
                            recon, mu, logvar = model(noisy_target, batch_base)

                        # Loss with KL annealing
                        beta = kl_beta_schedule(
                            epoch, KL_WARMUP_START, KL_WARMUP_END, KL_FINAL_BETA,
                        )
                        npw = 1.0 if stage == 1 else 2.0
                        # Connectivity loss ramps up for stage 1
                        if stage == 1 and epoch > CONNECTIVITY_WARMUP_START:
                            conn_progress = min(1.0, (epoch - CONNECTIVITY_WARMUP_START)
                                                / (CONNECTIVITY_WARMUP_END - CONNECTIVITY_WARMUP_START))
                            conn_w = CONNECTIVITY_WEIGHT * conn_progress
                        else:
                            conn_w = 0.0
                        loss = staged_loss(
                            recon, batch_target, batch_base, mu, logvar, beta,
                            new_pixel_weight=npw,
                            sharpness_weight=SHARPNESS_WEIGHT,
                            connectivity_weight=conn_w,
                        )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    scheduler.step()

                    # Update UI periodically
                    if epoch % 5 == 0 or epoch == 1:
                        avg_loss = epoch_loss / n_batches
                        try:
                            modal.after(0, lambda e=epoch, l=avg_loss, b=beta: _update_ui(e, l, b))
                        except Exception:
                            break

                # Save model
                if not cancel_var["cancelled"]:
                    torch.save(model.state_dict(), model_path)
                    try:
                        modal.after(0, _training_complete)
                    except Exception:
                        pass
                else:
                    try:
                        modal.after(0, lambda: _training_cancelled())
                    except Exception:
                        pass

            except Exception as e:
                try:
                    modal.after(0, lambda: _training_error(str(e)))
                except Exception:
                    pass
            finally:
                self._training = False

        def _update_ui(epoch, loss, beta):
            try:
                epoch_label.config(text=f"Epoch {epoch} / {TRAINING_EPOCHS}")
                loss_label.config(text=f"Loss: {loss:.4f}")
                beta_label.config(text=f"KL β: {beta:.3f}")
                progress_var.set(epoch)
            except Exception:
                pass

        def _training_complete():
            try:
                cancel_btn.config(state="disabled")
                status_label.config(text="✓ Training complete!")
                epoch_label.config(text="Done!", fg=CLR_SUCCESS)

                # Add close / next stage buttons
                btn_frame = tk.Frame(modal, bg=CLR_BG)
                btn_frame.pack(pady=8)

                tk.Button(
                    btn_frame, text="Close", font=("SF Pro", 11),
                    fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
                    padx=12, pady=6, cursor="hand2",
                    highlightbackground=CLR_BORDER,
                    command=lambda: _close_modal(),
                ).pack(side="left", padx=4)

                if stage < 4:
                    tk.Button(
                        btn_frame, text=f"Draw Stage {stage + 1} →",
                        font=("SF Pro", 11, "bold"),
                        fg=CLR_BG, bg=CLR_PRIMARY, relief="solid", bd=1,
                        padx=12, pady=6, cursor="hand2",
                        highlightbackground=CLR_PRIMARY, activeforeground=CLR_BG,
                        command=lambda: (_close_modal(), self._open_drawer(stage + 1)),
                    ).pack(side="left", padx=4)

                tk.Button(
                    btn_frame, text="See Results →",
                    font=("SF Pro", 11, "bold"),
                    fg=CLR_BG, bg=CLR_SUCCESS, relief="solid", bd=1,
                    padx=12, pady=6, cursor="hand2",
                    highlightbackground=CLR_SUCCESS, activeforeground=CLR_BG,
                    command=lambda: (_close_modal(), self.controller.show_generator()),
                ).pack(side="left", padx=4)

            except Exception:
                pass

        def _training_cancelled():
            try:
                status_label.config(text="Training cancelled.", fg=CLR_DANGER)
                modal.after(1500, _close_modal)
            except Exception:
                pass

        def _training_error(msg):
            try:
                status_label.config(text=f"Error: {msg}", fg=CLR_DANGER)
            except Exception:
                pass

        def _close_modal():
            try:
                modal.grab_release()
                modal.destroy()
                self.update_stats()
            except Exception:
                pass

        thread = threading.Thread(target=do_train, daemon=True)
        thread.start()