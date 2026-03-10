import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import torch
import torch.optim as optim
import numpy as np
from config import DATA_FILE, MODEL_FILE, GOAL_IMAGES, GRID_SIZE
from model import VAE, loss_function, add_noise

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#2b2d42")
        self.controller = controller

        tk.Label(self, text="AI Face Studio", font=("Helvetica", 28, "bold"), bg="#2b2d42", fg="#edf2f4").pack(pady=(40, 10))
        tk.Label(self, text="Create, Train, and Generate", font=("Helvetica", 14, "italic"), bg="#2b2d42", fg="#8d99ae").pack(pady=(0, 20))

        stats_frame = tk.Frame(self, bg="#8d99ae", padx=20, pady=10)
        stats_frame.pack(pady=10)
        self.count_label = tk.Label(stats_frame, text="", font=("Helvetica", 12, "bold"), bg="#8d99ae", fg="#2b2d42")
        self.count_label.pack()
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(stats_frame, variable=self.progress_var, maximum=GOAL_IMAGES, length=250)
        self.progress_bar.pack(pady=(5, 0))

        btn_frame = tk.Frame(self, bg="#2b2d42")
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="✏️ 1. Drawing Studio", command=controller.show_drawer, width=25).pack(pady=5)
        
        self.refine_btn = ttk.Button(btn_frame, text="🛠️ 2. Refine Studio (RLHF)", command=controller.show_refine, width=25)
        self.refine_btn.pack(pady=5)

        ttk.Button(btn_frame, text="🧠 3. Train AI Model", command=self.train_ai, width=25).pack(pady=5)
        
        self.gen_btn = ttk.Button(btn_frame, text="✨ 4. AI Generator", command=controller.show_generator, width=25)
        self.gen_btn.pack(pady=5)

        self.update_stats()

    def update_stats(self):
        count = self.get_dataset_count()
        self.progress_var.set(count)
        self.count_label.config(text=f"Dataset: {count} / {GOAL_IMAGES} Images")
        
        state = ['!disabled'] if os.path.exists(MODEL_FILE) else['disabled']
        self.gen_btn.state(state)
        self.refine_btn.state(state) # Needs a trained model to refine!

    def get_dataset_count(self):
        if not os.path.exists(DATA_FILE): return 0
        with open(DATA_FILE, "r") as f: return sum(1 for row in f)

    def train_ai(self):
        if self.get_dataset_count() < 50:
            messagebox.showwarning("Data", "Get at least 50 drawings before training!")
            return

        train_win = tk.Toplevel(self)
        train_win.title("Training AI")
        train_win.geometry("400x300")
        train_win.configure(bg="#2b2d42")
        tk.Label(train_win, text="Training in progress...", font=("Helvetica", 14), bg="#2b2d42", fg="white").pack(pady=10)
        log_text = tk.Text(train_win, height=10, width=40)
        log_text.pack(pady=10)
        self.update()

        data =[]
        with open(DATA_FILE, "r") as f:
            for row in csv.reader(f):
                float_row = [float(x) for x in row]
                data.append(float_row)
                # DATA AUGMENTATION: Automatically flip images horizontally
                grid = np.array(float_row).reshape(GRID_SIZE, GRID_SIZE)
                flipped = np.fliplr(grid).flatten().tolist()
                data.append(flipped)
        
        clean_tensor = torch.tensor(data, dtype=torch.float32)
        model = VAE()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        epochs = 600
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            # DENOISING: Add noise to input, but calculate loss against CLEAN data
            noisy_tensor = add_noise(clean_tensor, noise_factor=0.1)
            recon_batch, mu, logvar = model(noisy_tensor)
            
            loss = loss_function(recon_batch, clean_tensor, mu, logvar)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 or epoch == 1:
                log_text.insert(tk.END, f"Epoch {epoch}/{epochs} | Loss: {loss.item() / len(data):.4f}\n")
                log_text.see(tk.END)
                train_win.update()

        torch.save(model.state_dict(), MODEL_FILE)
        train_win.destroy()
        messagebox.showinfo("Success", f"Trained on {len(data)} augmented images!")
        self.update_stats()