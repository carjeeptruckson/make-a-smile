import tkinter as tk
from tkinter import ttk
import csv
import os
import numpy as np
from config import (
    GRID_SIZE, CELL_SIZE, DATA_DIR,
    STAGE_NAMES, STAGE_ICONS, STAGE_FILES, STAGE_MIN_SAMPLES,
)

# Design system colors
CLR_PRIMARY = "#3B82F6"
CLR_SUCCESS = "#10B981"
CLR_DANGER = "#EF4444"
CLR_TEXT = "#111827"
CLR_TEXT_SECONDARY = "#6B7280"
CLR_BORDER = "#E5E7EB"
CLR_BG = "#FFFFFF"
CLR_BG_LIGHT = "#F9FAFB"
CLR_BG_HOVER = "#F3F4F6"
CLR_CANVAS_BG = "#F3F4F6"
CLR_BASE_PIXEL = "#BFDBFE"
CLR_DRAW_PIXEL = "#1F2937"
CLR_GRID_LINE = "#E5E7EB"


class DrawerUI(tk.Frame):
    """Stage-aware drawing studio with base layer support and gallery."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR_BG)
        self.controller = controller
        self.grid_visible = True
        self.current_stage = 1
        self.has_unsaved_changes = False
        self.selected_base_index = -1
        self._thumbnail_cache = {}

        # Grid data: current drawing and locked base layer
        self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.base_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.last_x, self.last_y = None, None

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG, pady=12)
        header.pack(fill="x")

        self.stage_label = tk.Label(
            header, text="Stage 1: Head Shapes",
            font=("SF Pro", 18, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        )
        self.stage_label.pack(side="left", padx=20)

        self.count_label = tk.Label(
            header, text="0/30 samples",
            font=("SF Pro", 12), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        self.count_label.pack(side="right", padx=20)

        tk.Frame(self, bg=CLR_BORDER, height=1).pack(fill="x")

        # ── Status message ──────────────────────────────────────
        self.status_label = tk.Label(
            self, text="Draw head shapes. Base: None.",
            font=("SF Pro", 11), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        self.status_label.pack(pady=(8, 4))

        # ── Canvas ──────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=CLR_CANVAS_BG, padx=8, pady=8)
        canvas_frame.pack(pady=8)

        canvas_size = GRID_SIZE * CELL_SIZE
        self.canvas = tk.Canvas(
            canvas_frame, width=canvas_size, height=canvas_size,
            bg=CLR_BG, cursor="crosshair",
            highlightthickness=1, highlightbackground="#D1D5DB",
        )
        self.canvas.pack()

        self.rects = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.rects[y][x] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=CLR_BG, outline=CLR_GRID_LINE,
                )

        # Pixel counter overlay
        self.pixel_count_label = tk.Label(
            self, text="Pixels: 0",
            font=("SF Pro", 10), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        self.pixel_count_label.pack()

        # Canvas binds
        self.canvas.bind("<Button-1>", lambda e: self._start_draw(e, 1))
        self.canvas.bind("<B1-Motion>", lambda e: self._drag_draw(e, 1))
        self.canvas.bind("<Button-3>", lambda e: self._start_draw(e, 0))
        self.canvas.bind("<B3-Motion>", lambda e: self._drag_draw(e, 0))
        self.canvas.bind("<Button-2>", lambda e: self._start_draw(e, 0))
        self.canvas.bind("<B2-Motion>", lambda e: self._drag_draw(e, 0))

        # ── Base Layer Gallery ──────────────────────────────────
        self.gallery_frame = tk.Frame(self, bg=CLR_BG_LIGHT, pady=8)
        # Not packed yet — shown only when stage > 1

        self.gallery_label = tk.Label(
            self.gallery_frame, text="Select base layer:",
            font=("SF Pro", 11, "bold"), fg=CLR_TEXT_SECONDARY, bg=CLR_BG_LIGHT,
        )
        self.gallery_label.pack(anchor="w", padx=12)

        gallery_inner = tk.Frame(self.gallery_frame, bg=CLR_BG_LIGHT)
        gallery_inner.pack(fill="x", padx=12, pady=4)

        self.gallery_canvas = tk.Canvas(
            gallery_inner, height=72, bg=CLR_BG_LIGHT,
            highlightthickness=0,
        )
        self.gallery_canvas.pack(side="left", fill="x", expand=True)

        gallery_btn_frame = tk.Frame(gallery_inner, bg=CLR_BG_LIGHT)
        gallery_btn_frame.pack(side="right", padx=(8, 0))

        self.clear_base_btn = tk.Button(
            gallery_btn_frame, text="Clear selection", font=("SF Pro", 10),
            fg=CLR_TEXT_SECONDARY, bg=CLR_BG, relief="solid", bd=1,
            command=self._clear_base_layer,
        )
        self.clear_base_btn.pack()

        self.gallery_canvas.bind("<Button-1>", self._on_gallery_click)

        # ── Controls ────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=CLR_BG, pady=8)
        btn_frame.pack()

        self.save_btn = tk.Button(
            btn_frame, text="Save & Next", font=("SF Pro", 12, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground="#2563EB",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=16, pady=8, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            command=self._save_drawing,
        )
        self.save_btn.grid(row=0, column=0, padx=6)

        tk.Button(
            btn_frame, text="Clear Canvas", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self._clear_grid,
        ).grid(row=0, column=1, padx=6)

        tk.Button(
            btn_frame, text="Toggle Grid", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self._toggle_grid,
        ).grid(row=0, column=2, padx=6)

        tk.Button(
            btn_frame, text="Main Menu", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self.controller.show_menu,
        ).grid(row=0, column=3, padx=6)

        # ── Save notification ──────────────────────────────────
        self.notif_label = tk.Label(
            self, text="", font=("SF Pro", 12, "bold"),
            fg=CLR_SUCCESS, bg=CLR_BG,
        )
        self.notif_label.pack(pady=4)

    # ── Public API ──────────────────────────────────────────────

    def set_stage(self, stage):
        """Configure the drawer for a specific stage."""
        self.current_stage = stage
        self.selected_base_index = -1
        self._thumbnail_cache = {}

        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        icon = STAGE_ICONS.get(stage, "")
        self.stage_label.config(text=f"{icon} Stage {stage}: {name}")
        self._update_sample_count()

        if stage > 1:
            self._load_gallery_thumbnails()
            self.gallery_frame.pack(fill="x", before=self.notif_label)
            base_desc = f"Select a base from Stage {stage - 1}"
        else:
            self.gallery_frame.pack_forget()
            base_desc = "None (draw from scratch)"

        self.status_label.config(
            text=f"Stage {stage}: Draw {name.lower()}. Base: {base_desc}."
        )
        self._clear_grid()
        self._clear_base_layer()

    # ── Drawing ─────────────────────────────────────────────────

    def _start_draw(self, event, color_val):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        self.last_x, self.last_y = x, y
        self._color_pixel(x, y, color_val)

    def _drag_draw(self, event, color_val):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if self.last_x is not None and self.last_y is not None:
            for px, py in self._bresenham(self.last_x, self.last_y, x, y):
                self._color_pixel(px, py, color_val)
        self.last_x, self.last_y = x, y

    def _color_pixel(self, x, y, color_val):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return
        # Don't allow drawing over base layer pixels
        if self.base_data[y][x] == 1 and color_val == 1:
            return
        # Don't allow erasing base layer pixels
        if self.base_data[y][x] == 1 and color_val == 0:
            return

        self.grid_data[y][x] = color_val
        self.has_unsaved_changes = True
        if color_val == 1:
            color = CLR_DRAW_PIXEL
        elif self.base_data[y][x] == 1:
            color = CLR_BASE_PIXEL
        else:
            color = CLR_BG
        self.canvas.itemconfig(self.rects[y][x], fill=color)
        self._update_pixel_count()

    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm for smooth drawing."""
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    # ── Grid management ─────────────────────────────────────────

    def _clear_grid(self):
        """Clear current drawing layer (keeps base layer)."""
        self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.has_unsaved_changes = False
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = CLR_BASE_PIXEL if self.base_data[y][x] == 1 else CLR_BG
                self.canvas.itemconfig(self.rects[y][x], fill=color)
        self._update_pixel_count()

    def _toggle_grid(self):
        self.grid_visible = not self.grid_visible
        outline = CLR_GRID_LINE if self.grid_visible else ""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.canvas.itemconfig(self.rects[y][x], outline=outline)

    def _update_pixel_count(self):
        count = sum(self.grid_data[y][x] for y in range(GRID_SIZE) for x in range(GRID_SIZE))
        self.pixel_count_label.config(text=f"Pixels: {count}")

    def _update_sample_count(self):
        count = self._get_sample_count()
        minimum = STAGE_MIN_SAMPLES.get(self.current_stage, 30)
        self.count_label.config(text=f"{count}/{minimum} samples")

    def _get_sample_count(self):
        """Count rows in the current stage's target data file."""
        _, target_path, _ = STAGE_FILES[self.current_stage]
        if target_path is None:
            return 0
        if not os.path.exists(target_path):
            return 0
        try:
            with open(target_path, "r") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    # ── Base layer gallery ──────────────────────────────────────

    def _load_gallery_thumbnails(self):
        """Load thumbnails from the previous stage's target data."""
        self.gallery_canvas.delete("all")
        self._gallery_images = []  # Keep references to prevent GC
        self._gallery_data = []

        prev_stage = self.current_stage - 1
        if prev_stage < 1:
            return

        # For stage 2, previous data is stage1_heads (target only)
        # For stages 3+, previous data is the target CSV of the previous stage
        _, prev_target_path, _ = STAGE_FILES[prev_stage]
        if prev_target_path is None or not os.path.exists(prev_target_path):
            return

        try:
            with open(prev_target_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 256:
                        self._gallery_data.append([float(v) for v in row])
        except Exception:
            return

        thumb_size = 64
        padding = 4
        for i, data in enumerate(self._gallery_data):
            # Create a PhotoImage thumbnail
            img = tk.PhotoImage(width=thumb_size, height=thumb_size)
            grid = np.array(data).reshape(GRID_SIZE, GRID_SIZE)
            pixel_size = thumb_size // GRID_SIZE

            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    color = "#4B5563" if grid[y][x] > 0.5 else "#F9FAFB"
                    for py in range(pixel_size):
                        for px in range(pixel_size):
                            img.put(color, (x * pixel_size + px, y * pixel_size + py))

            x_pos = i * (thumb_size + padding) + padding
            self.gallery_canvas.create_image(
                x_pos, 4, anchor="nw", image=img, tags=f"thumb_{i}",
            )
            self._gallery_images.append(img)

        total_width = len(self._gallery_data) * (thumb_size + padding) + padding
        self.gallery_canvas.config(scrollregion=(0, 0, total_width, 72))

    def _on_gallery_click(self, event):
        """Handle click on a gallery thumbnail."""
        thumb_size = 64
        padding = 4
        index = (event.x - padding) // (thumb_size + padding)

        if 0 <= index < len(self._gallery_data):
            self.selected_base_index = index
            self._load_base_from_data(self._gallery_data[index])
            self.status_label.config(
                text=f"Stage {self.current_stage}: Base #{index + 1} loaded."
            )

    def _load_base_from_data(self, flat_data):
        """Load a flat 256-value list as the base layer."""
        self.base_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        for i, val in enumerate(flat_data):
            y, x = divmod(i, GRID_SIZE)
            self.base_data[y][x] = 1 if val > 0.5 else 0

        # Clear current drawing and render base
        self.grid_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = CLR_BASE_PIXEL if self.base_data[y][x] == 1 else CLR_BG
                self.canvas.itemconfig(self.rects[y][x], fill=color)
        self._update_pixel_count()

    def _clear_base_layer(self):
        """Clear the base layer (for Stage 1 or free drawing)."""
        self.base_data = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.selected_base_index = -1
        self._clear_grid()
        if self.current_stage > 1:
            self.status_label.config(
                text=f"Stage {self.current_stage}: Select a base from Stage {self.current_stage - 1}."
            )

    # ── Saving ──────────────────────────────────────────────────

    def _save_drawing(self):
        """Save the current drawing to the appropriate stage CSV files."""
        # Build the combined target image (base + current drawing)
        target_flat = []
        base_flat = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Target = base layer OR current drawing
                target_val = 1 if (self.base_data[y][x] == 1 or self.grid_data[y][x] == 1) else 0
                target_flat.append(target_val)
                base_flat.append(self.base_data[y][x])

        # Require at least one drawn pixel (not just a blank save)
        drawn_pixels = sum(self.grid_data[y][x] for y in range(GRID_SIZE) for x in range(GRID_SIZE))
        if drawn_pixels == 0:
            self._show_notification("⚠ Draw something first!", CLR_DANGER)
            return

        stage = self.current_stage
        base_path, target_path, _ = STAGE_FILES[stage]

        try:
            if stage == 1:
                # Stage 1: save target only (no base conditioning)
                with open(target_path, "a", newline="") as f:
                    csv.writer(f).writerow(target_flat)
            else:
                # Stages 2-4: save matched base and target rows
                if self.selected_base_index < 0:
                    self._show_notification("⚠ Select a base layer first!", CLR_DANGER)
                    return

                with open(base_path, "a", newline="") as f:
                    csv.writer(f).writerow(base_flat)
                with open(target_path, "a", newline="") as f:
                    csv.writer(f).writerow(target_flat)

            self._show_notification("✓ Saved", CLR_SUCCESS)
            self.has_unsaved_changes = False
            self._update_sample_count()
            self._clear_grid()  # Clear drawing but keep base loaded

            # Re-render base so user can draw another variation
            if stage > 1 and self.selected_base_index >= 0:
                self._load_base_from_data(self._gallery_data[self.selected_base_index])

        except Exception as e:
            self._show_notification(f"⚠ Save failed: {e}", CLR_DANGER)

    def _show_notification(self, text, color):
        """Show a brief notification that fades after 2 seconds."""
        self.notif_label.config(text=text, fg=color)
        self.after(2000, lambda: self.notif_label.config(text=""))