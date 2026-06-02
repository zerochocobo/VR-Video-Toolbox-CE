import tkinter as tk
import sys
import os

# Ensure we can import sibling modules if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import gui

if __name__ == "__main__":
    root = tk.Tk()
    app = gui.VRVideoToolsApp(root)
    root.mainloop()
