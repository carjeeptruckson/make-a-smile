import tkinter as tk
from tkinter import ttk
import torch
import random
from config import GRID_SIZE, CELL_SIZE, MODEL_FILE, LATENT_DIM
from model import VAE

class GeneratorUI(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.grid_visible = True
        self.model = VAE()
        
        tk.Label(self, text="AI Generator", font=("Helvetica", 18, "bold")).pack(pady=10)

        self.canvas = tk.Canvas(self, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack(pady=10)

        self.rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.rects[y][x] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#dddddd")

        # Dynamic Sliders based on LATENT_DIM
        self.sliders =[]
        slider_frame = tk.Frame(self)
        slider_frame.pack(pady=5)
        
        # Split sliders into two columns for a cleaner UI
        for i in range(LATENT_DIM):
            row = i // 2
            col = (i % 2) * 2
            tk.Label(slider_frame, text=f"Trait {i+1}").grid(row=row, column=col, padx=5)
            scale = ttk.Scale(slider_frame, from_=-3.0, to=3.0, orient=tk.HORIZONTAL, length=120, command=self.generate_face)
            scale.grid(row=row, column=col+1, padx=5, pady=2)
            self.sliders.append(scale)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Surprise Me", command=self.randomize_sliders).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Morph (Animate)", command=self.morph_animation).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Main Menu", command=controller.show_menu).grid(row=0, column=2, padx=5)

    def load_model(self):
        self.model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        self.model.eval()
        self.randomize_sliders()

    def randomize_sliders(self):
        for slider in self.sliders:
            slider.set(random.uniform(-2.0, 2.0))
        self.generate_face()

    def generate_face(self, val=None):
        z_values =[slider.get() for slider in self.sliders]
        z_tensor = torch.tensor([z_values], dtype=torch.float32)

        with torch.no_grad():
            generated_img = self.model.decode(z_tensor).view(GRID_SIZE, GRID_SIZE).numpy()

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = "#1a1a1a" if generated_img[y][x] > 0.5 else "white"
                self.canvas.itemconfig(self.rects[y][x], fill=color)

    def morph_animation(self):
        # Captures current sliders, picks a random target, and animates the transition
        start_z =[s.get() for s in self.sliders]
        target_z =[random.uniform(-2.0, 2.0) for _ in range(LATENT_DIM)]
        steps = 15
        
        def step_morph(current_step):
            alpha = current_step / float(steps)
            for i, slider in enumerate(self.sliders):
                new_val = start_z[i] * (1 - alpha) + target_z[i] * alpha
                slider.set(new_val)
            self.generate_face()
            
            if current_step < steps:
                self.after(50, lambda: step_morph(current_step + 1))
                
        step_morph(0)