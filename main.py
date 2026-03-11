import tkinter as tk
from tkinter import ttk
from ui.menu import MainMenu
from ui.drawer import DrawerUI
from ui.generator import GeneratorUI
from ui.refine import RefineUI


class AppController(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Studio")
        self.geometry("900x900")
        self.minsize(800, 700)
        self.configure(bg="#FFFFFF")

        # Global styling
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure('TButton',
                        font=('SF Pro', 11),
                        padding=10,
                        background='#F3F4F6',
                        foreground='#111827')
        style.map('TButton',
                   background=[('active', '#E5E7EB')])
        style.configure('TLabel',
                        font=('SF Pro', 11),
                        background='#FFFFFF',
                        foreground='#111827')
        style.configure('TProgressbar',
                        troughcolor='#E5E7EB',
                        background='#3B82F6',
                        thickness=8)

        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.pack(fill="both", expand=True)

        self.frames = {
            "Menu": MainMenu(self.container, self),
            "Drawer": DrawerUI(self.container, self),
            "Generator": GeneratorUI(self.container, self),
            "Refine": RefineUI(self.container, self),
        }
        self.show_menu()

    def switch_frame(self, frame_name):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[frame_name].pack(fill="both", expand=True)

    def show_menu(self):
        self.frames["Menu"].update_stats()
        self.switch_frame("Menu")

    def show_drawer(self, stage=1):
        self.frames["Drawer"].set_stage(stage)
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