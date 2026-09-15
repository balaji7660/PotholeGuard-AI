"""
main_gui.py
Entry point for Tkinter Pothole Detection & Avoidance GUI.

Run with:
    python main_gui.py
"""
import tkinter as tk
from gui.app import PotholeDetectionApp

def main():
    root = tk.Tk()
    app = PotholeDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
