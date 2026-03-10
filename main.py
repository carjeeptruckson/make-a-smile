import tkinter as tk
from tkinter import ttk
from ui.menu import MainMenu
from ui.drawer import DrawerUI
from ui.generator import GeneratorUI
from ui.refine import RefineUI  # NEW IMPORT

class AppController(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Face Maker Studio")
        self.geometry("550x750")
        
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=8, background="#edf2f4", foreground="#2b2d42")
        style.map('TButton', background=[('active', '#8d99ae')])

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {
            "Menu": MainMenu(self.container, self),
            "Drawer": DrawerUI(self.container, self),
            "Generator": GeneratorUI(self.container, self),
            "Refine": RefineUI(self.container, self) # NEW UI REGISTERED
        }
        self.show_menu()

    def switch_frame(self, frame_name):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[frame_name].pack(fill="both", expand=True)

    def show_menu(self):
        self.frames["Menu"].update_stats()
        self.switch_frame("Menu")

    def show_drawer(self):
        self.switch_frame("Drawer")

    def show_refine(self):
        self.frames["Refine"].load_model()
        self.switch_frame("Refine")

    def show_generator(self):
        self.frames["Generator"].load_model()
        self.switch_frame("Generator")

if __name__ == "__main__":
    app = AppController()
    app.mainloop()