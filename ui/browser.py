import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import numpy as np
from config import (
    GRID_SIZE, STAGE_NAMES, STAGE_ICONS, STAGE_FILES, STAGE_MIN_SAMPLES,
)

# Design system colors
CLR_PRIMARY = "#3B82F6"
CLR_PRIMARY_HOVER = "#2563EB"
CLR_SUCCESS = "#10B981"
CLR_DANGER = "#EF4444"
CLR_TEXT = "#111827"
CLR_TEXT_SECONDARY = "#6B7280"
CLR_TEXT_MUTED = "#9CA3AF"
CLR_BORDER = "#E5E7EB"
CLR_BG = "#FFFFFF"
CLR_BG_LIGHT = "#F9FAFB"
CLR_BG_HOVER = "#F3F4F6"

THUMB_SIZE = 64
THUMB_PAD = 6
COLS = 8


class DataBrowserUI(tk.Frame):
    """Scrollable browser for viewing and editing training data per stage."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR_BG)
        self.controller = controller
        self.current_stage = 1
        self._targets = []
        self._bases = []
        self._thumb_images = []
        self._selected_index = -1

        self._build_ui()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_BG, pady=12)
        header.pack(fill="x")

        tk.Button(
            header, text="← Back", font=("SF Pro", 11),
            fg=CLR_TEXT, bg=CLR_BG, relief="solid", bd=1,
            padx=12, pady=4, cursor="hand2",
            highlightbackground=CLR_BORDER,
            command=self.controller.show_menu,
        ).pack(side="left", padx=16)

        self.title_label = tk.Label(
            header, text="Training Data: Stage 1",
            font=("SF Pro", 18, "bold"), fg=CLR_TEXT, bg=CLR_BG,
        )
        self.title_label.pack(side="left", padx=8)

        self.count_label = tk.Label(
            header, text="0 samples",
            font=("SF Pro", 12), fg=CLR_TEXT_SECONDARY, bg=CLR_BG,
        )
        self.count_label.pack(side="right", padx=20)

        tk.Frame(self, bg=CLR_BORDER, height=1).pack(fill="x")

        # ── Stage selector tabs ──────────────────────────────────
        tab_frame = tk.Frame(self, bg=CLR_BG_LIGHT, pady=8)
        tab_frame.pack(fill="x")

        self._tab_buttons = {}
        for stage in range(1, 5):
            icon = STAGE_ICONS.get(stage, "")
            name = STAGE_NAMES.get(stage, f"Stage {stage}")
            btn = tk.Button(
                tab_frame, text=f"{icon} {name}",
                font=("SF Pro", 11), fg=CLR_TEXT, bg=CLR_BG,
                relief="solid", bd=1, padx=12, pady=6, cursor="hand2",
                highlightbackground=CLR_BORDER,
                command=lambda s=stage: self.set_stage(s),
            )
            btn.pack(side="left", padx=4)
            self._tab_buttons[stage] = btn

        # ── Scrollable thumbnail grid ────────────────────────────
        grid_container = tk.Frame(self, bg=CLR_BG)
        grid_container.pack(fill="both", expand=True, padx=16, pady=8)

        self.grid_canvas = tk.Canvas(
            grid_container, bg=CLR_BG, highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            grid_container, orient="vertical", command=self.grid_canvas.yview,
        )
        self.grid_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.grid_canvas.pack(side="left", fill="both", expand=True)

        self.thumb_frame = tk.Frame(self.grid_canvas, bg=CLR_BG)
        self._canvas_window = self.grid_canvas.create_window(
            (0, 0), window=self.thumb_frame, anchor="nw",
        )

        self.thumb_frame.bind("<Configure>", self._on_frame_configure)
        self.grid_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling
        self.grid_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.grid_canvas.bind("<Button-4>", self._on_mousewheel)
        self.grid_canvas.bind("<Button-5>", self._on_mousewheel)

        # ── Bottom action bar ────────────────────────────────────
        action_bar = tk.Frame(self, bg=CLR_BG_LIGHT, pady=10)
        action_bar.pack(fill="x", side="bottom")

        self.edit_btn = tk.Button(
            action_bar, text="Edit Selected", font=("SF Pro", 11, "bold"),
            fg=CLR_BG, bg=CLR_PRIMARY, activebackground=CLR_PRIMARY_HOVER,
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=14, pady=6, cursor="hand2",
            highlightbackground=CLR_PRIMARY,
            state="disabled",
            command=self._edit_selected,
        )
        self.edit_btn.pack(side="left", padx=16)

        self.delete_btn = tk.Button(
            action_bar, text="Delete Selected", font=("SF Pro", 11, "bold"),
            fg=CLR_BG, bg=CLR_DANGER, activebackground="#DC2626",
            activeforeground=CLR_BG,
            relief="solid", bd=1, padx=14, pady=6, cursor="hand2",
            highlightbackground=CLR_DANGER,
            state="disabled",
            command=self._delete_selected,
        )
        self.delete_btn.pack(side="left")

        self.selection_label = tk.Label(
            action_bar, text="Click a sample to select it",
            font=("SF Pro", 10), fg=CLR_TEXT_MUTED, bg=CLR_BG_LIGHT,
        )
        self.selection_label.pack(side="right", padx=16)

    # ── Event handlers ────────────────────────────────────────────

    def _on_frame_configure(self, event=None):
        self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        self.grid_canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.grid_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.grid_canvas.yview_scroll(1, "units")
        else:
            delta = -1 if event.delta > 0 else 1
            self.grid_canvas.yview_scroll(delta, "units")

    # ── Public API ────────────────────────────────────────────────

    def set_stage(self, stage):
        """Load and display data for the given stage."""
        self.current_stage = stage
        self._selected_index = -1

        icon = STAGE_ICONS.get(stage, "")
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        self.title_label.config(text=f"Training Data: {icon} {name}")

        # Highlight active tab
        for s, btn in self._tab_buttons.items():
            if s == stage:
                btn.config(bg=CLR_PRIMARY, fg=CLR_BG,
                           highlightbackground=CLR_PRIMARY)
            else:
                btn.config(bg=CLR_BG, fg=CLR_TEXT,
                           highlightbackground=CLR_BORDER)

        self._load_data()
        self._render_thumbnails()
        self._update_selection_ui()

    def _load_data(self):
        """Load target (and base) CSV data for current stage."""
        self._targets = []
        self._bases = []
        stage = self.current_stage
        base_path, target_path, _ = STAGE_FILES[stage]

        if target_path and os.path.exists(target_path):
            try:
                with open(target_path, "r") as f:
                    for row in csv.reader(f):
                        if len(row) == 256:
                            self._targets.append([float(v) for v in row])
            except Exception:
                pass

        if stage > 1 and base_path and os.path.exists(base_path):
            try:
                with open(base_path, "r") as f:
                    for row in csv.reader(f):
                        if len(row) == 256:
                            self._bases.append([float(v) for v in row])
            except Exception:
                pass

        self.count_label.config(text=f"{len(self._targets)} samples")

    def _render_thumbnails(self):
        """Render all thumbnails into the scrollable grid."""
        # Clear existing
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        self._thumb_images = []

        if not self._targets:
            tk.Label(
                self.thumb_frame, text="No training data yet.\nDraw some samples first!",
                font=("SF Pro", 13), fg=CLR_TEXT_MUTED, bg=CLR_BG,
                justify="center",
            ).grid(row=0, column=0, columnspan=COLS, pady=40)
            return

        for i, target in enumerate(self._targets):
            row, col = divmod(i, COLS)
            cell = self._create_thumb_cell(i, target)
            cell.grid(row=row, column=col, padx=THUMB_PAD, pady=THUMB_PAD)

    def _create_thumb_cell(self, index, target_data):
        """Create a single clickable thumbnail cell."""
        cell = tk.Frame(self.thumb_frame, bg=CLR_BG, padx=2, pady=2,
                        highlightbackground=CLR_BORDER, highlightthickness=1)

        # Build thumbnail image
        img = tk.PhotoImage(width=THUMB_SIZE, height=THUMB_SIZE)
        grid = np.array(target_data).reshape(GRID_SIZE, GRID_SIZE)
        pixel_size = THUMB_SIZE // GRID_SIZE

        # If we have base data, show base in light blue, new in dark
        has_base = index < len(self._bases) and self.current_stage > 1
        if has_base:
            base_grid = np.array(self._bases[index]).reshape(GRID_SIZE, GRID_SIZE)
        else:
            base_grid = None

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[y][x] > 0.5:
                    if base_grid is not None and base_grid[y][x] > 0.5:
                        color = "#93C5FD"  # base pixel (blue)
                    else:
                        color = "#1F2937"  # drawn pixel (dark)
                else:
                    color = "#F9FAFB"  # empty
                for py in range(pixel_size):
                    for px in range(pixel_size):
                        img.put(color, (x * pixel_size + px, y * pixel_size + py))

        self._thumb_images.append(img)

        label = tk.Label(cell, image=img, bg=CLR_BG, cursor="hand2")
        label.pack()

        # Index label
        tk.Label(
            cell, text=f"#{index + 1}",
            font=("SF Pro", 9), fg=CLR_TEXT_MUTED, bg=CLR_BG,
        ).pack()

        # Click to select
        label.bind("<Button-1>", lambda e, idx=index: self._select_item(idx, cell))
        cell.bind("<Button-1>", lambda e, idx=index: self._select_item(idx, cell))

        # Double-click to edit
        label.bind("<Double-Button-1>", lambda e, idx=index: self._edit_item(idx))

        cell._index = index
        return cell

    def _select_item(self, index, cell):
        """Select a thumbnail item."""
        # Deselect previous
        for widget in self.thumb_frame.winfo_children():
            if hasattr(widget, '_index'):
                widget.config(highlightbackground=CLR_BORDER, highlightthickness=1)

        # Select new
        self._selected_index = index
        cell.config(highlightbackground=CLR_PRIMARY, highlightthickness=3)
        self._update_selection_ui()

    def _update_selection_ui(self):
        """Update button states based on selection."""
        has_sel = self._selected_index >= 0
        self.edit_btn.config(
            state="normal" if has_sel else "disabled",
            bg=CLR_PRIMARY if has_sel else CLR_BG_HOVER,
            fg=CLR_BG if has_sel else CLR_TEXT_MUTED,
        )
        self.delete_btn.config(
            state="normal" if has_sel else "disabled",
            bg=CLR_DANGER if has_sel else CLR_BG_HOVER,
            fg=CLR_BG if has_sel else CLR_TEXT_MUTED,
        )
        if has_sel:
            self.selection_label.config(
                text=f"Sample #{self._selected_index + 1} selected"
            )
        else:
            self.selection_label.config(text="Click a sample to select it")

    # ── Actions ───────────────────────────────────────────────────

    def _edit_selected(self):
        if self._selected_index < 0:
            return
        self._edit_item(self._selected_index)

    def _edit_item(self, index):
        """Open the drawer to edit an existing sample."""
        if index < 0 or index >= len(self._targets):
            return
        target = self._targets[index]
        base = self._bases[index] if index < len(self._bases) and self.current_stage > 1 else None
        self.controller.show_drawer_edit(self.current_stage, index, target, base)

    def _delete_selected(self):
        """Delete the selected sample from the CSV files."""
        if self._selected_index < 0:
            return

        index = self._selected_index
        confirm = messagebox.askyesno(
            "Delete Sample",
            f"Delete sample #{index + 1}? This cannot be undone.",
        )
        if not confirm:
            return

        stage = self.current_stage
        base_path, target_path, _ = STAGE_FILES[stage]

        # Remove from in-memory lists
        if index < len(self._targets):
            self._targets.pop(index)
        if stage > 1 and index < len(self._bases):
            self._bases.pop(index)

        # Rewrite CSV files
        try:
            self._rewrite_csv(target_path, self._targets)
            if stage > 1 and base_path:
                self._rewrite_csv(base_path, self._bases)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
            return

        self._selected_index = -1
        self._render_thumbnails()
        self._update_selection_ui()
        self.count_label.config(text=f"{len(self._targets)} samples")

    @staticmethod
    def _rewrite_csv(path, rows):
        """Rewrite a CSV file with the given rows."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
