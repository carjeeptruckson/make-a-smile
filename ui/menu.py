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
    BOUNDARY_WEIGHT, AUGMENT_SYMMETRY_STAGES,
    STAGE_REFINE_FILES, REFINE_MIN_SAMPLES, REFINE_TRAINING_EPOCHS,
    CRITIC_WEIGHT, CRITIC_WARMUP_END, FOCAL_ALPHA,
)
from model import (
    HeadVAE, ConditionalVAE, RefineModel,
    staged_loss, experimental_staged_loss, refine_loss,
    kl_beta_schedule, add_noise, augment_batch,
)

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
        sidebar_container = tk.Frame(content, bg=CLR_BG_LIGHT, width=420)
        sidebar_container.pack(side="left", fill="y", padx=(0, 16))
        sidebar_container.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_container, bg=CLR_BG_LIGHT, highlightthickness=0)
        sidebar_scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(sidebar_canvas, bg=CLR_BG_LIGHT, padx=16, pady=16)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _configure_sidebar(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
            
        def _configure_canvas(event):
            sidebar_canvas.itemconfig(sidebar_window, width=event.width)

        sidebar.bind("<Configure>", _configure_sidebar)
        sidebar_canvas.bind("<Configure>", _configure_canvas)

        # Mousewheel support for macOS and Windows
        def _on_mousewheel(event):
            # macOS scroll delta is usually small, so we don't multiply by -1 if we can avoid it.
            # but on macOS event.delta is positive for scrolling up. 
            if event.delta:
                sidebar_canvas.yview_scroll(int(-1 * np.sign(event.delta)), "units")
            else:
                # Linux Button-4/5
                if event.num == 5:
                    sidebar_canvas.yview_scroll(1, "units")
                elif event.num == 4:
                    sidebar_canvas.yview_scroll(-1, "units")

        def _bind_mousewheel(event):
            sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)
            sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(event):
            sidebar_canvas.unbind_all("<MouseWheel>")
            sidebar_canvas.unbind_all("<Button-4>")
            sidebar_canvas.unbind_all("<Button-5>")

        sidebar_container.bind("<Enter>", _bind_mousewheel)
        sidebar_container.bind("<Leave>", _unbind_mousewheel)

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

        # ── Refine AI row ──
        refine_row = tk.Frame(card, bg=CLR_BG)
        refine_row.pack(fill="x", pady=(4, 0))

        refine_status = tk.Label(
            refine_row, text="Refine ● —",
            font=("SF Pro", 9), fg=CLR_TEXT_MUTED, bg=CLR_BG,
        )
        refine_status.pack(side="left")

        train_experimental_btn = tk.Button(
            refine_row, text="Train (Exp)", font=("SF Pro", 9, "bold"),
            fg=CLR_BG, bg="#8B5CF6", activebackground="#7C3AED",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=8, pady=2, cursor="hand2",
            highlightbackground="#8B5CF6",
            command=lambda s=stage: self._train_experimental(s),
        )
        train_experimental_btn.pack(side="right", padx=(4, 0))

        train_refine_btn = tk.Button(
            refine_row, text="Train Refine", font=("SF Pro", 9, "bold"),
            fg=CLR_BG, bg=CLR_WARNING, activebackground="#D97706",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=8, pady=2, cursor="hand2",
            highlightbackground=CLR_WARNING,
            command=lambda s=stage: self._train_refine(s),
        )
        train_refine_btn.pack(side="right")

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
            "train_refine_btn": train_refine_btn,
            "train_experimental_btn": train_experimental_btn,
            "refine_status": refine_status,
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

    def _get_refine_sample_count(self, stage):
        """Count rows in a stage's refine target data file."""
        _, _, target_path, _ = STAGE_REFINE_FILES[stage]
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

    def _refine_model_exists(self, stage):
        """Check if a trained refine model exists for this stage."""
        _, _, _, model_path = STAGE_REFINE_FILES[stage]
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
            refine_count = self._get_refine_sample_count(stage)
            has_refine_model = self._refine_model_exists(stage)
            has_vae_model = self._stage_model_exists(stage)

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

            # ── Refine AI status & buttons ──
            if has_refine_model:
                card["refine_status"].config(
                    fg=CLR_SUCCESS,
                    text=f"Refine ● Trained ({refine_count} samples)",
                )
            elif refine_count > 0:
                card["refine_status"].config(
                    fg=CLR_WARNING,
                    text=f"Refine ● {refine_count}/{REFINE_MIN_SAMPLES}",
                )
            else:
                card["refine_status"].config(
                    fg=CLR_TEXT_MUTED,
                    text="Refine ● No data",
                )

            # Train Refine: enabled when VAE trained + enough refine data
            can_train_refine = has_vae_model and refine_count >= REFINE_MIN_SAMPLES
            card["train_refine_btn"].config(
                state="normal" if can_train_refine else "disabled",
                bg=CLR_WARNING if can_train_refine else CLR_BG_HOVER,
                fg=CLR_BG if can_train_refine else CLR_TEXT_MUTED,
            )

            # Train Experimental: enabled when VAE trained + RefineModel trained
            can_train_exp = has_vae_model and has_refine_model
            card["train_experimental_btn"].config(
                state="normal" if can_train_exp else "disabled",
                bg="#8B5CF6" if can_train_exp else CLR_BG_HOVER,
                fg=CLR_BG if can_train_exp else CLR_TEXT_MUTED,
            )

            # Preview / Refine buttons: only if model trained
            has_model = has_vae_model
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

                # Augmentation: flip + translations + symmetry
                aug_targets, aug_bases = augment_batch(
                    targets, bases, stage,
                    symmetry_stages=AUGMENT_SYMMETRY_STAGES,
                )

                target_tensor = torch.tensor(aug_targets, dtype=torch.float32)
                base_tensor = torch.tensor(aug_bases, dtype=torch.float32)

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
                        # Boundary loss for stages 2+: penalize pixels
                        # adjacent to the base face outline
                        bnd_w = BOUNDARY_WEIGHT if stage > 1 else 0.0
                        loss = staged_loss(
                            recon, batch_target, batch_base, mu, logvar, beta,
                            new_pixel_weight=npw,
                            sharpness_weight=SHARPNESS_WEIGHT,
                            connectivity_weight=conn_w,
                            boundary_weight=bnd_w,
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

    # ── Refine AI Training ──────────────────────────────────────

    def _train_refine(self, stage):
        """Full training of the RefineModel for a stage."""
        refine_count = self._get_refine_sample_count(stage)
        if refine_count < REFINE_MIN_SAMPLES:
            messagebox.showwarning(
                "Not Enough Refine Data",
                f"Stage {stage} refine needs at least {REFINE_MIN_SAMPLES} samples.\n"
                f"You have {refine_count}. Keep refining faces!"
            )
            return

        if self._training:
            messagebox.showinfo("Training", "A training session is already in progress.")
            return

        self._training = True
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        input_csv, base_csv, target_csv, model_path = STAGE_REFINE_FILES[stage]

        # ── Training modal ──
        modal = tk.Toplevel(self)
        modal.title(f"Training Refine Model: {name}")
        modal.geometry("500x400")
        modal.configure(bg=CLR_BG)
        modal.transient(self.winfo_toplevel())
        modal.grab_set()

        tk.Label(
            modal, text=f"Training Refine AI: {name}",
            font=("SF Pro", 16, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        ).pack(pady=(20, 4))

        tk.Label(
            modal, text=f"Training on {refine_count} correction samples",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(pady=(0, 16))

        epoch_label = tk.Label(
            modal, text="Epoch 0 / 400",
            font=("SF Pro", 32, "bold"), fg=CLR_WARNING, bg=CLR_BG,
        )
        epoch_label.pack(pady=8)

        loss_label = tk.Label(
            modal, text="Loss: —",
            font=("SF Pro", 13), fg=CLR_TEXT, bg=CLR_BG,
        )
        loss_label.pack()

        progress_var = tk.DoubleVar()
        ttk.Progressbar(
            modal, variable=progress_var, maximum=REFINE_TRAINING_EPOCHS, length=400,
        ).pack(pady=(16, 4))

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

        def do_train():
            try:
                # Load refine data
                inputs, bases, targets = [], [], []
                for path, dest in [(input_csv, inputs), (base_csv, bases), (target_csv, targets)]:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            for row in csv.reader(f):
                                if len(row) == 256:
                                    dest.append([float(v) for v in row])

                min_len = min(len(inputs), len(bases), len(targets))
                if min_len == 0:
                    raise ValueError("No valid refine data found")

                input_t = torch.tensor(inputs[-min_len:], dtype=torch.float32)
                base_t = torch.tensor(bases[-min_len:], dtype=torch.float32)
                target_t = torch.tensor(targets[-min_len:], dtype=torch.float32)

                # Augmentation: horizontal flip
                aug_inputs, aug_bases, aug_targets = [], [], []
                for i in range(min_len):
                    aug_inputs.append(input_t[i])
                    aug_bases.append(base_t[i])
                    aug_targets.append(target_t[i])
                    # Flipped versions
                    aug_inputs.append(input_t[i].view(GRID_SIZE, GRID_SIZE).flip(1).flatten())
                    aug_bases.append(base_t[i].view(GRID_SIZE, GRID_SIZE).flip(1).flatten())
                    aug_targets.append(target_t[i].view(GRID_SIZE, GRID_SIZE).flip(1).flatten())

                input_t = torch.stack(aug_inputs)
                base_t = torch.stack(aug_bases)
                target_t = torch.stack(aug_targets)

                model = RefineModel()
                if os.path.exists(model_path):
                    try:
                        model.load_state_dict(torch.load(model_path, weights_only=True))
                    except Exception:
                        pass

                optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=REFINE_TRAINING_EPOCHS, eta_min=TRAINING_LR * 0.05,
                )

                n_samples = input_t.shape[0]
                batch_size = min(32, n_samples)

                for epoch in range(1, REFINE_TRAINING_EPOCHS + 1):
                    if cancel_var["cancelled"]:
                        break

                    model.train()
                    perm = torch.randperm(n_samples)
                    epoch_loss = 0.0
                    n_batches = 0

                    for start in range(0, n_samples, batch_size):
                        idx = perm[start:start + batch_size]
                        bi, bb, bt = input_t[idx], base_t[idx], target_t[idx]

                        optimizer.zero_grad()
                        pred = model(bi, bb)
                        loss = refine_loss(pred, bt)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    scheduler.step()

                    if epoch % 5 == 0 or epoch == 1:
                        avg_loss = epoch_loss / max(n_batches, 1)
                        try:
                            modal.after(0, lambda e=epoch, l=avg_loss: _update_ui(e, l))
                        except Exception:
                            break

                if not cancel_var["cancelled"]:
                    torch.save(model.state_dict(), model_path)
                    try:
                        modal.after(0, _training_complete)
                    except Exception:
                        pass
                else:
                    try:
                        modal.after(0, _training_cancelled)
                    except Exception:
                        pass

            except Exception as e:
                try:
                    modal.after(0, lambda: _training_error(str(e)))
                except Exception:
                    pass
            finally:
                self._training = False

        def _update_ui(epoch, loss):
            try:
                epoch_label.config(text=f"Epoch {epoch} / {REFINE_TRAINING_EPOCHS}")
                loss_label.config(text=f"Loss: {loss:.4f}")
                progress_var.set(epoch)
            except Exception:
                pass

        def _training_complete():
            try:
                cancel_btn.config(state="disabled")
                status_label.config(text="✓ Refine model trained!")
                epoch_label.config(text="Done!", fg=CLR_SUCCESS)
                tk.Button(
                    modal, text="Close", font=("SF Pro", 11),
                    fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
                    padx=12, pady=6, cursor="hand2",
                    highlightbackground=CLR_BORDER,
                    command=_close_modal,
                ).pack(pady=8)
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

    def _train_experimental(self, stage):
        """Experimental training: retrain VAE using RefineModel as a live critic."""
        if not self._refine_model_exists(stage):
            messagebox.showwarning(
                "No Refine Model",
                f"Train the Refine model for stage {stage} first."
            )
            return

        count = self._get_stage_sample_count(stage)
        minimum = STAGE_MIN_SAMPLES.get(stage, 30)
        if count < minimum:
            messagebox.showwarning(
                "Not Enough Data",
                f"Stage {stage} needs at least {minimum} samples.\n"
                f"You have {count}."
            )
            return

        if self._training:
            messagebox.showinfo("Training", "A training session is already in progress.")
            return

        self._training = True
        name = STAGE_NAMES.get(stage, f"Stage {stage}")

        # ── Training modal ──
        modal = tk.Toplevel(self)
        modal.title(f"Experimental Training: {name}")
        modal.geometry("500x440")
        modal.configure(bg=CLR_BG)
        modal.transient(self.winfo_toplevel())
        modal.grab_set()

        tk.Label(
            modal, text=f"⚡ Experimental: {name}",
            font=("SF Pro", 16, "bold"), fg="#8B5CF6", bg=CLR_BG,
        ).pack(pady=(20, 4))

        tk.Label(
            modal, text=f"VAE + Critic training on {count} samples",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        ).pack(pady=(0, 4))

        tk.Label(
            modal, text="RefineModel acts as a live critic to guide the VAE",
            font=("SF Pro", 10), fg=CLR_TEXT_MUTED, bg=CLR_BG,
        ).pack(pady=(0, 16))

        epoch_label = tk.Label(
            modal, text="Epoch 0 / 500",
            font=("SF Pro", 32, "bold"), fg="#8B5CF6", bg=CLR_BG,
        )
        epoch_label.pack(pady=8)

        loss_label = tk.Label(
            modal, text="Loss: —  |  Critic: —",
            font=("SF Pro", 13), fg=CLR_TEXT, bg=CLR_BG,
        )
        loss_label.pack()

        beta_label = tk.Label(
            modal, text="KL β: 0.000  |  Critic ramp: 0%",
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

        def do_train():
            try:
                base_path, target_path, model_path = STAGE_FILES[stage]
                _, _, _, refine_model_path = STAGE_REFINE_FILES[stage]

                # Load VAE training data
                targets, bases = [], []
                with open(target_path, "r") as f:
                    for row in csv.reader(f):
                        if len(row) == 256:
                            targets.append([float(v) for v in row])

                if stage > 1 and base_path and os.path.exists(base_path):
                    with open(base_path, "r") as f:
                        for row in csv.reader(f):
                            if len(row) == 256:
                                bases.append([float(v) for v in row])

                # Curate
                targets, bases, n_removed = self._curate_training_data(targets, bases, stage)
                if n_removed > 0:
                    try:
                        modal.after(0, lambda n=n_removed: status_label.config(
                            text=f"Cleaned: removed {n} bad sample(s)", fg=CLR_WARNING))
                    except Exception:
                        pass

                # Augmentation: flip + translations + symmetry
                aug_targets, aug_bases = augment_batch(
                    targets, bases, stage,
                    symmetry_stages=AUGMENT_SYMMETRY_STAGES,
                )

                target_tensor = torch.tensor(aug_targets, dtype=torch.float32)
                base_tensor = torch.tensor(aug_bases, dtype=torch.float32)

                # Create fresh VAE model
                if stage == 1:
                    model = HeadVAE()
                else:
                    model = ConditionalVAE(stage_name=f"stage{stage}")

                # Load existing weights as starting point
                if os.path.exists(model_path):
                    try:
                        import shutil
                        shutil.copy2(model_path, model_path + ".bak")
                        model.load_state_dict(torch.load(model_path, weights_only=True))
                    except Exception:
                        pass

                # Load the frozen RefineModel (critic)
                refine_model = RefineModel()
                refine_model.load_state_dict(torch.load(refine_model_path, weights_only=True))
                refine_model.eval()
                for param in refine_model.parameters():
                    param.requires_grad = False

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
                    perm = torch.randperm(n_samples)
                    epoch_loss = 0.0
                    n_batches = 0

                    # Critic warmup: ramp from 0 to 1 over CRITIC_WARMUP_END epochs
                    critic_progress = min(1.0, epoch / CRITIC_WARMUP_END) if CRITIC_WARMUP_END > 0 else 1.0

                    for start in range(0, n_samples, batch_size):
                        idx = perm[start:start + batch_size]
                        batch_target = target_tensor[idx]
                        batch_base = base_tensor[idx]

                        optimizer.zero_grad()

                        noisy_target = add_noise(batch_target, noise_factor=NOISE_FACTOR)

                        if stage == 1:
                            recon, mu, logvar = model(noisy_target)
                        else:
                            recon, mu, logvar = model(noisy_target, batch_base)

                        # KL annealing (slightly lower beta for experimental to preserve diversity)
                        beta = kl_beta_schedule(
                            epoch, KL_WARMUP_START, KL_WARMUP_END, KL_FINAL_BETA * 0.8,
                        )
                        npw = 1.0 if stage == 1 else 2.0

                        if stage == 1 and epoch > CONNECTIVITY_WARMUP_START:
                            conn_progress = min(1.0, (epoch - CONNECTIVITY_WARMUP_START)
                                                / (CONNECTIVITY_WARMUP_END - CONNECTIVITY_WARMUP_START))
                            conn_w = CONNECTIVITY_WEIGHT * conn_progress
                        else:
                            conn_w = 0.0

                        bnd_w = BOUNDARY_WEIGHT if stage > 1 else 0.0

                        loss = experimental_staged_loss(
                            recon, batch_target, batch_base, mu, logvar, beta,
                            refine_model=refine_model,
                            critic_weight=CRITIC_WEIGHT,
                            critic_warmup_progress=critic_progress,
                            focal_alpha=FOCAL_ALPHA,
                            new_pixel_weight=npw,
                            sharpness_weight=SHARPNESS_WEIGHT,
                            connectivity_weight=conn_w,
                            boundary_weight=bnd_w,
                        )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    scheduler.step()

                    if epoch % 5 == 0 or epoch == 1:
                        avg_loss = epoch_loss / max(n_batches, 1)
                        try:
                            modal.after(0, lambda e=epoch, l=avg_loss, b=beta, cp=critic_progress:
                                        _update_ui(e, l, b, cp))
                        except Exception:
                            break

                if not cancel_var["cancelled"]:
                    torch.save(model.state_dict(), model_path)
                    try:
                        modal.after(0, _training_complete)
                    except Exception:
                        pass
                else:
                    try:
                        modal.after(0, _training_cancelled)
                    except Exception:
                        pass

            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    modal.after(0, lambda: _training_error(str(e)))
                except Exception:
                    pass
            finally:
                self._training = False

        def _update_ui(epoch, loss, beta, critic_progress):
            try:
                epoch_label.config(text=f"Epoch {epoch} / {TRAINING_EPOCHS}")
                loss_label.config(text=f"Loss: {loss:.4f}")
                beta_label.config(
                    text=f"KL β: {beta:.3f}  |  Critic ramp: {int(critic_progress * 100)}%")
                progress_var.set(epoch)
            except Exception:
                pass

        def _training_complete():
            try:
                cancel_btn.config(state="disabled")
                status_label.config(text="✓ Experimental training complete!")
                epoch_label.config(text="Done!", fg=CLR_SUCCESS)

                btn_frame = tk.Frame(modal, bg=CLR_BG)
                btn_frame.pack(pady=8)

                tk.Button(
                    btn_frame, text="Close", font=("SF Pro", 11),
                    fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
                    padx=12, pady=6, cursor="hand2",
                    highlightbackground=CLR_BORDER,
                    command=_close_modal,
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