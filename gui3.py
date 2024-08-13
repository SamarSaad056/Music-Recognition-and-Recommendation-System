import tkinter as tk
from PIL import Image, ImageTk, ImageSequence,ImageFilter 
from pathlib import Path
import subprocess
import sys

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\samar\OneDrive\Desktop\GradProject2\frame3")

def button_click():
    python_path = sys.executable  # Get the path to the current Python interpreter
    second_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui1.py"  # Replace with your second script's absolute path
    subprocess.Popen([python_path, second_script_path])
    
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def on_button_click():
    python_path = sys.executable  # Get the path to the current Python interpreter
    second_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui.py"  # Replace with your second script's absolute path
    subprocess.Popen([python_path, second_script_path])

def draw_gui():
    global window, canvas, gif_frames, loaded_gif, image_references

    window = tk.Tk()
    window.geometry("362x595")
    window.configure(bg="#00394D")

    canvas = tk.Canvas(
        window,
        bg="#00394D",
        height=595,
        width=362,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.place(x=0, y=0)

    image_path = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\Exclamation_mark.png") 
    image_path1 = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\home.png") 

    try:
        img = Image.open(image_path)
        img = img.resize((200, 200), Image.BOX)
        img = ImageTk.PhotoImage(img)

        img1 = Image.open(image_path1)
        img1 = img1.resize((30, 30), Image.BOX)
        img1 = ImageTk.PhotoImage(img1)
        
    except FileNotFoundError as e:
        print("File not found:", e)
    except Exception as e:
        print("Error loading image:", e)


    Exclamation_mark = canvas.create_image(
            190, 190,
            anchor="center",
            image=img
        )

    text1 = canvas.create_text(
        115.0,
        370.0,
        anchor="nw",
        text="We didn't quite catch that",
        fill="#FFFFFF",
        font=("Chewy", 10)  
    )
    text2 = canvas.create_text(
        105.0,
        300.0,
        anchor="nw",
        text="No Result.",
        fill="#FFFFFF",
        font=("Chewy", 30)  
    )
    button = tk.Button(
        window,
        text="Try Again",
        command=button_click,
        bg="#2CB1C4",       # Background color
        fg="white",      # Foreground color (text color)
        font=("chewy", 15),  # Font and font size
        relief=tk.RAISED,    # Border relief style
        width= 15,        # Button width
        height=1,        # Button height
        borderwidth=5
    )

    Home_button = tk.Button(
            window,
            image=img1,
            command=on_button_click,
            relief=tk.FLAT,
            bg="#00394D", 
            highlightthickness=0
        )
    


    button.pack()
    button.place(x=100,y=430)

    Home_button.pack()
    Home_button.place(x=325,y=10)

    window.resizable(False, False)
    window.mainloop()

if __name__ == "__main__":
    draw_gui()