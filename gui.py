from tkinter import Tk, Canvas
from PIL import Image, ImageTk, ImageFilter
from pathlib import Path
import subprocess
import sys
import tkinter as tk
import wave
import pyaudio
from scipy.io.wavfile import read
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import pickle
from scipy import signal
import time 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from pygame import mixer 




OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = r"C:\Users\samar\OneDrive\Desktop\GradProject2\frame0"  # Adjust this path accordingly

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)



def draw_gui():
    window = Tk()
    window.geometry("362x595")
    window.configure(bg="#00394D")

    canvas = Canvas(
        window,
        bg="#00394D",
        height=595,
        width=362,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.place(x=0, y=0)

    # Load and display the image 
    image_path1 = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\curve.png") 
    image_path = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\logo.png")  # Replace with your image path
    try:
        img = Image.open(image_path)
        img = img.resize((244, 292), Image.BOX)
        img = ImageTk.PhotoImage(img)

        img1 = Image.open(image_path1)
        img1 = img1.resize((300, 200), Image.BOX)
        img1 = ImageTk.PhotoImage(img1)

        logo= canvas.create_image(
            180, 320,
            anchor="center",
            image=img
        )
        Curve = canvas.create_image(
            150, 100,
            anchor="center",
            image=img1
        )
    except FileNotFoundError as e:
        print("File not found:", e)
    except Exception as e:
        print("Error loading image:", e)
    
    welcome_text =canvas.create_text(
        30.0,
        90.0,
        anchor="nw",
        text="Welcome",
        fill="#FFFFFF",
        font=("Chewy", 40)  
    )


    # Function to handle button click
    def button_click():
        #window.withdraw()
        python_path = sys.executable  # Get the path to the current Python interpreter
        second_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui1.py"  # Replace with your second script's absolute path
        subprocess.Popen([python_path, second_script_path])
       #button_text = "Tab to Start"

    
    button = tk.Button(
        window,
        text="Tab to Start",
        command=button_click,
        bg="#2CB1C4",       # Background color
        fg="white",      # Foreground color (text color)
        font=("chewy", 20),  # Font and font size
        relief=tk.RAISED,    # Border relief style
        width= 15 ,        # Button width
        height=1,        # Button height
        borderwidth=3)
        

    # Bind the button click event to the button_click function
    button.pack()
    button.place(x=80,y=460)
  

    
    window.resizable(False, False)
    
    window.mainloop()


if __name__ == "__main__":
    draw_gui()
    




