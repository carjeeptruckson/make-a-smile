import tkinter as tk
from tkinter import ttk
import csv
from config import GRID_SIZE, CELL_SIZE, DATA_FILE, GOAL_IMAGES

class DrawerUI(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.grid_visible = True
        
        tk.Label(self, text="Drawing Studio", font=("Helvetica", 18, "bold")).pack(pady=10)
        tk.Label(self, text="Left-Click: Draw  |  Right-Click: Erase").pack()

        # Canvas Setup
        self.canvas = tk.Canvas(self, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack(pady=10)

        self.grid_data = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.last_x, self.last_y = None, None

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.rects[y][x] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#dddddd")

        # Binds
        self.canvas.bind("<Button-1>", lambda e: self.start_draw(e, 1))
        self.canvas.bind("<B1-Motion>", lambda e: self.drag_draw(e, 1))
        self.canvas.bind("<Button-3>", lambda e: self.start_draw(e, 0))
        self.canvas.bind("<B3-Motion>", lambda e: self.drag_draw(e, 0))
        self.canvas.bind("<Button-2>", lambda e: self.start_draw(e, 0)) 
        self.canvas.bind("<B2-Motion>", lambda e: self.drag_draw(e, 0))

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Save & Next", command=self.save_drawing).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_grid).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Toggle Grid", command=self.toggle_grid).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Main Menu", command=controller.show_menu).grid(row=0, column=3, padx=5)

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        outline_color = "#dddddd" if self.grid_visible else ""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.canvas.itemconfig(self.rects[y][x], outline=outline_color)

    def get_line_pixels(self, x0, y0, x1, y1):
        points =[]
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def start_draw(self, event, color_val):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        self.last_x, self.last_y = x, y
        self.color_pixel(x, y, color_val)

    def drag_draw(self, event, color_val):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if self.last_x is not None and self.last_y is not None:
            for px, py in self.get_line_pixels(self.last_x, self.last_y, x, y):
                self.color_pixel(px, py, color_val)
        self.last_x, self.last_y = x, y

    def color_pixel(self, x, y, color_val):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid_data[y][x] = color_val
            color = "#1a1a1a" if color_val == 1 else "white"
            self.canvas.itemconfig(self.rects[y][x], fill=color)

    def clear_grid(self):
        self.grid_data = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.canvas.itemconfig(self.rects[y][x], fill="white")

    def save_drawing(self):
        flat_data = [pixel for row in self.grid_data for pixel in row]
        if sum(flat_data) > 0:
            with open(DATA_FILE, "a", newline="") as f:
                csv.writer(f).writerow(flat_data)
        self.clear_grid()