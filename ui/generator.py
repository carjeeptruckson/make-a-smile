import tkinter as tk
from tkinter import ttk
import torch
import random
import threading
import os
from config import (
    GRID_SIZE, CELL_SIZE, RENDER_THRESHOLD,
    STAGE1_Z, STAGE2_Z, STAGE3_Z, STAGE4_Z,
    STAGE_NAMES, STAGE_ICONS, STAGE_FILES,
)
from model import HeadVAE, ConditionalVAE

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

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG_LIGHT, pady=16)
        header.pack(fill="x")

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
        """Randomize sliders for a single stage and regenerate."""
        if stage in self.slider_widgets:
            for slider, label in self.slider_widgets[stage]:
                val = random.uniform(-2.0, 2.0)
                slider.set(val)
                label.config(text=f"{val:.2f}")
        self._generate_face()

    def _randomize_all(self):
        """Randomize all stage sliders and regenerate."""
        for stage in self.slider_widgets:
            for slider, label in self.slider_widgets[stage]:
                val = random.uniform(-2.0, 2.0)
                slider.set(val)
                label.config(text=f"{val:.2f}")
        self._generate_face()

    # ── Generation pipeline ─────────────────────────────────────

    def _generate_face(self):
        """Run the staged generation pipeline."""
        if self._generating or not self.models:
            return

        self._generating = True

        def do_generate():
            try:
                current_img = None

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
                            current_img = model.decode(z_tensor, condition)

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