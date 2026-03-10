import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "my_faces.csv")
MODEL_FILE = os.path.join(DATA_DIR, "vae_model.pth")

GRID_SIZE = 16
CELL_SIZE = 24
LATENT_DIM = 6  # Upgraded from 2 to 6 for much higher detail!
GOAL_IMAGES = 200
