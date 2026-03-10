import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.optim as optim
import csv
from config import GRID_SIZE, CELL_SIZE, DATA_FILE, MODEL_FILE, LATENT_DIM
from model import VAE, loss_function

class RefineUI(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.model = VAE()
        
        tk.Label(self, text="Guided Refinement Studio", font=("Helvetica", 18, "bold")).pack(pady=10)
        
        self.instruction_label = tk.Label(self, text="Step 1: Rate this generation", font=("Helvetica", 12, "italic"))
        self.instruction_label.pack(pady=5)

        # The Canvas
        self.canvas = tk.Canvas(self, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg="white")
        self.canvas.pack(pady=5)

        self.grid_data = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.rects[y][x] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#dddddd")

        # --- CONTROLS ---
        self.current_weight = 1.0
        self.can_draw = False

        # Frame 1: Rating Buttons (Visible initially)
        self.rating_frame = tk.Frame(self)
        self.rating_frame.pack(pady=10)
        
        tk.Button(self.rating_frame, text="Perfect\n(None)", bg="#a8e6cf", width=10, command=lambda: self.set_rating(0.5, "perfect")).grid(row=0, column=0, padx=5)
        tk.Button(self.rating_frame, text="A Little\nBad", bg="#ffd3b6", width=10, command=lambda: self.set_rating(1.0, "little")).grid(row=0, column=1, padx=5)
        tk.Button(self.rating_frame, text="A Lot\nBad", bg="#ffaaa5", width=10, command=lambda: self.set_rating(2.5, "lot")).grid(row=0, column=2, padx=5)
        tk.Button(self.rating_frame, text="Horrible!", bg="#ff8b94", font=("Helvetica", 10, "bold"), width=10, command=lambda: self.set_rating(4.0, "horrible")).grid(row=0, column=3, padx=5)

        # Frame 2: Editing Buttons (Hidden initially)
        self.edit_frame = tk.Frame(self)
        ttk.Button(self.edit_frame, text="Submit Correction & Train", command=self.submit_correction).grid(row=0, column=0, padx=5)
        ttk.Button(self.edit_frame, text="Cancel (Skip)", command=self.generate_base_face).grid(row=0, column=1, padx=5)

        ttk.Button(self, text="Main Menu", command=controller.show_menu).pack(pady=20)

        # Binds
        self.canvas.bind("<B1-Motion>", lambda e: self.paint(e, 1))
        self.canvas.bind("<Button-1>", lambda e: self.paint(e, 1))
        self.canvas.bind("<B3-Motion>", lambda e: self.paint(e, 0))
        self.canvas.bind("<Button-3>", lambda e: self.paint(e, 0))

    def load_model(self):
        self.model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        self.model.eval()
        self.generate_base_face()

    def generate_base_face(self):
        # Reset UI State
        self.can_draw = False
        self.instruction_label.config(text="Step 1: Rate this generation", fg="black")
        self.edit_frame.pack_forget()
        self.rating_frame.pack(pady=10)

        # Generate Image
        z = torch.randn(1, LATENT_DIM)
        with torch.no_grad():
            img = self.model.decode(z).view(GRID_SIZE, GRID_SIZE).numpy()
            
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                val = 1 if img[y][x] > 0.5 else 0
                self.grid_data[y][x] = val
                color = "#1a1a1a" if val == 1 else "white"
                self.canvas.itemconfig(self.rects[y][x], fill=color)

    def set_rating(self, weight, severity):
        self.current_weight = weight
        
        if severity == "perfect":
            # If it's perfect, no need to edit. Train immediately.
            self.submit_correction()
        else:
            # Unlock the canvas for drawing
            self.can_draw = True
            self.rating_frame.pack_forget()
            self.edit_frame.pack(pady=10)
            self.instruction_label.config(text="Step 2: Fix the mistakes on the canvas, then submit.", fg="#d90429")

    def paint(self, event, val):
        if not self.can_draw: return
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid_data[y][x] = val
            color = "#1a1a1a" if val == 1 else "white"
            self.canvas.itemconfig(self.rects[y][x], fill=color)

    def submit_correction(self):
        flat_data =[pixel for row in self.grid_data for pixel in row]
        
        # 1. Save the corrected image to your permanent dataset
        with open(DATA_FILE, "a", newline="") as f:
            csv.writer(f).writerow(flat_data)

        # 2. INSTANT WEIGHTED TRAINING STEP
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        tensor_img = torch.tensor(flat_data, dtype=torch.float32)
        
        optimizer.zero_grad()
        recon, mu, logvar = self.model(tensor_img)
        loss = loss_function(recon, tensor_img, mu, logvar)
        
        # Multiply the loss by the severity weight!
        weighted_loss = loss * self.current_weight
        weighted_loss.backward()
        
        optimizer.step()
        torch.save(self.model.state_dict(), MODEL_FILE)
        self.model.eval()

        # Generate the next face to keep the loop going
        self.generate_base_face()