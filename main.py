"""
main.py
Main entry point for Pothole Detection & Avoidance GUI.

Run with:
    python main.py
"""
import tkinter as tk
from gui.app import PotholeDetectionApp

def main():
    root = tk.Tk()
    app = PotholeDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
