import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageTk, ImageFilter
from scipy.io.wavfile import read
import librosa
from pydub import AudioSegment
from tqdm import tqdm
import pickle
from scipy import signal
import time 
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from pygame import mixer 
from tkinter import Tk, Canvas, PhotoImage
import wave
import pyaudio
from tkinter import Tk, Canvas, PhotoImage
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter
from pathlib import Path
import subprocess
import sys  
import threading 
import pygame

database = pickle.load(open('database1.pickle', 'rb'))
song_name_index = pickle.load(open("song_index1.pickle", "rb"))

global  window,track_title_Test ,id,artist_name_Test,track_title , artist_name ,track_title1,id1 , artist_name1 ,track_title2 ,id2, artist_name2 ,track_title3,id3 , artist_name3 
is_playing = False
track_title_Test=None
artist_name_Test=None
track_title1=None
artist_name1 =None
track_title2=None
artist_name2=None
track_title3=None
artist_name3=None
id=None
id1=None
id2=None
id3=None



window = tk.Tk()

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\samar\OneDrive\Desktop\GradProject2\frame1")

button_clicked = False 
buttonStart_clicked = False 


def record_audio(duration):
 
 file="input.wav"

 CHUNK = 1024
 FORMAT = pyaudio.paInt16
 CHANNELS = 1
 RATE = 44100
 RECORD_SECONDS = duration
 p = pyaudio.PyAudio()
 stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
 
 print("Waiting for 3 seconds...")
 time.sleep(3)  # Wait for 10 seconds

 print("Recording...")

 frames = []

 for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

 print("Recording finished.")

 stream.stop_stream()
 stream.close()
 p.terminate()

 audio_frames = b''.join(frames)  # Convert frames to bytes
 wave_file = wave.open(file, 'wb')
 wave_file.setnchannels(1)
 wave_file.setsampwidth(2)
 wave_file.setframerate(44100)
 wave_file.writeframes(audio_frames)
 wave_file.close()
 is_music(file)


def fingerprintMap5(audio):

    audio, Fs = librosa.load(audio)
    # Parameters
    window_length_seconds = 2
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 6
    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples #Pad the audio signal with zeros to achieve the desired window length
    song_input = np.pad(audio, (0, amount_to_pad))
    # Perform a short time fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )
    constellation_map = []
    for time_idx, window in enumerate(stft.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = signal.find_peaks(spectrum, prominence=0.0001, distance=200)
        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
        n_peaks = min(num_peaks, len(peaks))
        # Get the n_peaks largest peaks from the prominences
        # This is an argpartition
        # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    #print(len(constellation_map))  
      
    return constellation_map

def create_hashes5(constellation_map, song_id=None):
    hashes = {}
    # Use this for binning - 23_000 is slighlty higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10
    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 500]: 
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 10:
                continue
            # Place the frequencies (in Hz) into a 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)
            # Produce a 32 bit hash
            # Use bit shifting to move the bits to the correct location
            hash = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    #print(hashes)
    return hashes


def score_hashes_against_database(hashes):

    
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        #print(hash)
        if hash in database:
            #print(hash)
            matching_occurences = database[hash]
            for source_time, song_index in matching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = []
                matches_per_song[song_index].append((hash, sample_time, source_time))
            


    scores = {}
    for song_index, matches in matches_per_song.items():
        song_scores_by_offset = {}
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            song_scores_by_offset[delta] += 18

        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1]:
                max = (offset, score)
        
        scores[song_index] = max

    # Sort the scores for the user
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    #print(scores)
    return scores





def print_top_five():
        
    
           
        constellation = fingerprintMap5("input.wav")
        hashes = create_hashes5(constellation, None)
        
        scores = score_hashes_against_database(hashes)[:1]
         
        df = pd.read_csv('D_Clustered.csv')
        with open("song_index1.pickle", 'rb') as songs_file:
            song_name_index = pickle.load(songs_file)
        for song_id, score in scores:
            for index, row in df.iterrows():
                #print("h")
                if row['track_id'] == int(song_name_index[song_id]):
                    #print("hhh")
                    #print(int(song_name_index[song_id]))
                    track_title_Test=row['track_name']
                    artist_name_Test = row['artist_name']
                    id=row['track_id']
                    #print(sysmu.get_artist_name())
        
            
        

        #def recommend():
        df2 = pd.read_csv('D_Clustered.csv')

        user_input_song = track_title_Test

        entered_song = df2[df2['track_name'] == user_input_song].iloc[0]
        
        audio_features = entered_song[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']]
        cluster_label = entered_song['cluster_label']

        same_cluster_songs = df2[df2['cluster_label'] == cluster_label]

        similarity_scores = cosine_similarity(audio_features.values.reshape(1, -1), same_cluster_songs[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']])

         #same_cluster_songs['similarity'] = similarity_scores.flatten()
        same_cluster_songs = same_cluster_songs.copy()
        same_cluster_songs.loc[:, 'similarity'] = similarity_scores.flatten()

        same_cluster_songs = same_cluster_songs[same_cluster_songs['track_name'] != user_input_song]

        recommendations = same_cluster_songs.sort_values(by='similarity', ascending=False).head(3)
        
        song_names = []
        artist=[]
        ids=[]
        
        for index, row in recommendations.iterrows():
           song_names.append(row['track_name'])
           artist.append(row['artist_name'])
           ids.append(row['track_id'])
           #ids.append()

        track_title1 = song_names[0]
        track_title2 = song_names[1]
        track_title3 = song_names[2]

        artist_name1=artist[0]
        artist_name2=artist[1]
        artist_name3 =artist[2]

        
        id1=ids[0]
        id2=ids[1]
        id3=ids[2]


        if track_title_Test is not None and score[1] > 1000:
              print(f"Song ID: {song_name_index[song_id]}, Score: {score[1]}, Song Name: {track_title_Test}, Artist Name: {artist_name_Test}")
              print(track_title1)
              transition_to_gui2(track_title_Test,artist_name_Test,track_title1,artist_name1,track_title2,artist_name2,track_title3, artist_name3,id,id1,id2,id3) # Transition to gui2.py after 10 seconds
              return
        else:
              transition_to_gui3()
              return
     


def is_music(sound_file):

    audio_data, _ = librosa.load(sound_file)

    # Analyze the frequency content
    spectrum = librosa.stft(y=audio_data)
    average_mag = abs(spectrum).mean()

    #  spectral and temporal characteristics
    amplitude_changes = librosa.feature.rms(y=audio_data)
    amplitude_changes = amplitude_changes[0]

     # Additional analysis
    spectral_flux = librosa.onset.onset_strength(y=audio_data)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data)[0]
   
    pitch = librosa.yin(audio_data, fmin=50, fmax=2000)
    #print_top_five()
   
    
    if average_mag > 0.01 and max(amplitude_changes) > 0.01 and max(spectral_flux) > 0.1 and max(spectral_centroid) > 1500  and sum(pitch) > 0 :
        print_top_five()
        return
    else:
       transition_to_gui3()
       return

# function to back to the home page.
def on_button_click():
    global button_clicked
    button_clicked = True
    python_path = sys.executable  # Get the path to the current Python interpreter
    second_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui.py"  # Replace with your second script's absolute path
    subprocess.Popen([python_path, second_script_path])
    
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def update_frame(counter):
    global gif_frames, loaded_gif, image_references
    try:
        updated_frame = image_references[counter]
        canvas.itemconfig(loaded_gif, image=updated_frame)
        window.after(20, update_frame, (counter + 1) % len(gif_frames))  # Update delay set to 20ms
    except Exception as e:
        print(f"Error updating frame: {e}")

def transition_to_gui2(track_title_Test , artist_name_Test,track_title1,artist_name1,track_title2,artist_name2,track_title3, artist_name3,id,id1,id2,id3):
  
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\samar\OneDrive\Desktop\GradProject2\frame2")


    def relative_to_assets(path: str) -> Path:
     return ASSETS_PATH / Path(path)

# function to back to the home page.
    def on_button_click():
     python_path = sys.executable  # Get the path to the current Python interpreter
     second_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui.py"  # Replace with your second script's absolute path
     subprocess.Popen([python_path, second_script_path])

    
    def Start_button_click1(idd):
        global is_playing
        
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False
        else:
            #dataframe = pd.read_excel('trackdfsmall 2 (1).xlsx')
            #rack_id = dataframe[dataframe['track_id'] == idd]['track_id'].values[0]
            file_path = r'C:\Users\samar\OneDrive\Desktop\GradProject2\RecogSongs\\' + str(idd).zfill(6) + '.mp3'

            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            is_playing = True
    
    def Start_button_click2(idd):
        global is_playing
        
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False
        else:
            
            file_path = r'C:\Users\samar\OneDrive\Desktop\GradProject2\Recommendation\\' + str(idd).zfill(6) + '.mp3'

            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            is_playing = True



    window.geometry("362x595")
    window.configure(bg = "#00394D")


    canvas = Canvas(
        window,
        bg = "#00394D",
        height = 595,
        width = 362,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    image_path2 = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\start.png") 
    image_path1 = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\home.png") 
    image_path = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\logo.png")  # Replace with your image path
    try:
        img = Image.open(image_path)
        img = img.resize((90, 90), Image.BOX)
        img = ImageTk.PhotoImage(img)

        img1 = Image.open(image_path)
        img1 = img1.resize((110, 110), Image.BOX)
        img1 = ImageTk.PhotoImage(img1)
        
        img2 = Image.open(image_path1)
        img2 = img2.resize((25, 25), Image.BOX)
        img2 = ImageTk.PhotoImage(img2)

        img3 = Image.open(image_path2)
        img3 = img3.resize((30, 30), Image.BOX)
        img3 = ImageTk.PhotoImage(img3)

        img4 = Image.open(image_path2)
        img4 = img4.resize((40, 40), Image.BOX)
        img4 = ImageTk.PhotoImage(img4)


    except FileNotFoundError as e:
        print("File not found:", e)
    except Exception as e:
        print("Error loading image:", e)

    Top_song = canvas.create_text(
        30.0,
        180.0,
        anchor="nw",
        text="Top Songs:",
        fill="#FFFFFF",
        font=("Chewy", 35 * -1)
    )

    rectangle_4 = canvas.create_rectangle(
        30.0,
        467.0,
        330,
        568.0,
         fill="#3E6472",
     outline="")

    Song_Name_4 =canvas.create_text(
         135,
         486,
        anchor="nw",
        text=track_title3,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Artist_Name_4 = canvas.create_text(
         135,
         516,
        anchor="nw",
        text=artist_name3,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    small_logo_4 = canvas.create_image(
            80, 517,
            anchor="center",
            image=img
        )

    start_button_4 = canvas.create_image(
            80, 517,
            anchor="center",
            image=img3
        )


    rectangle_3 = canvas.create_rectangle(
        30.0,
        357.0,
        330,
        458.0,
        fill="#3E6472",
        outline="")

    Song_Name_3 = canvas.create_text(
        135,
        376,
        anchor="nw",
        text=track_title2,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Artist_Name_3 = canvas.create_text(
        135,
        406,
        anchor="nw",
        text=artist_name2,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    small_logo_3 = canvas.create_image(
                80, 407,
                anchor="center",
                image=img
            )

    start_button_3 = canvas.create_image(
                80, 407,
                anchor="center",
                image=img3
            )


    rectangle_2 = canvas.create_rectangle(
        30.0,
        250.0,
        330.0,
        350.0,
        fill="#3E6472",
        outline="")

    Song_Name_2 = canvas.create_text(
        135,
        266,
        anchor="nw",
        text=track_title1,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Artist_Name_2 = canvas.create_text(
        135,
        296,
        anchor="nw",
        text=artist_name1,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Small_logo_2= canvas.create_image(
                80, 300,
                anchor="center",
                image=img
            )

    start_button_2= canvas.create_image(
                80, 300,
                anchor="center",
                image=img3
            )


    rectangle_1 = canvas.create_rectangle(
        20.0,
        45.0,
        335.0,
        172.0,
        fill="#3E6472",
        outline="")

    Song_Name_1 = canvas.create_text(
        145,
        66,
        anchor="nw",
        text=track_title_Test,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Artist_Name_1 = canvas.create_text(
        145,
        105,
        anchor="nw",
        text=artist_name_Test,
        fill="#FFFFFF",
        font=("Chewy", 15 * -1)
    )

    Big_logo_1 = canvas.create_image(
                85, 110,
                anchor="center",
                image=img1
            )
    start_button_1= canvas.create_image(
                85, 110,
                anchor="center",
                image=img4
            )


    Home_button = tk.Button(
                window,
                image=img2,
                command=on_button_click,
                relief=tk.FLAT,
                bg="#00394D", 

                highlightthickness=0
            )


    Home_button.pack()
    Home_button.place(x=320,y=10)

    #canvas.tag_bind(start_button_1, '<Button-1>', Start_button_click(track_title=track_title_Test))
    canvas.tag_bind(start_button_1, '<Button-1>', lambda event: Start_button_click1(idd=id))
    canvas.tag_bind(start_button_2, '<Button-1>', lambda event: Start_button_click2(idd=id1))
    canvas.tag_bind(start_button_3, '<Button-1>', lambda event: Start_button_click2(idd=id2))
    canvas.tag_bind(start_button_4, '<Button-1>', lambda event: Start_button_click2(idd=id3))
  

    window.resizable(False, False)
    window.mainloop()


def transition_to_gui3():
    if not button_clicked:
        #window.withdraw()
    # Replace with the correct path to your Python interpreter and gui2.py script
        python_path = sys.executable  # Get the path to the current Python interpreter
        gui2_script_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\gui3.py"  # Replace with the path to your gui2.py script
        subprocess.Popen([python_path, gui2_script_path])


def on_window_open(window):
    window.after(0, draw_gui)  # Open the window first
    window.after(500, record_audio, 10)  # Start recording audio after a delay of 500ms


def start_button_click():
    #record_audio(10)
    audio_thread = threading.Thread(target=record_audio(10))
    audio_thread.start()


def draw_gui():
    
    global canvas, gif_frames, loaded_gif, image_references
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

    image_path = relative_to_assets(r"C:\Users\samar\OneDrive\Desktop\GradProject2\home.png") 

    try:
        img = Image.open(image_path)
        img = img.resize((30, 30), Image.BOX)
        img = ImageTk.PhotoImage(img)
        
    except FileNotFoundError as e:
        print("File not found:", e)
    except Exception as e:
        print("Error loading image:", e)


    Home_button = tk.Button(
            window,
            image=img,
            command=on_button_click,
            relief=tk.FLAT,
            bg="#00394D", 
            highlightthickness=0
        )
   

    
# Create a function to handle the button click event
 
    Rercord_button = tk.Button(
        window,
        text="Start",
        command=start_button_click,
        bg="#2CB1C4",       # Background color
        fg="white",      # Foreground color (text color)
        font=("chewy", 12),  # Font and font size
        relief=tk.RAISED,    # Border relief style
        width=15,        # Button width
        height=2,        # Button height
        borderwidth=3 
        )

    Rercord_button.pack()
    Rercord_button.place(x=100,y=450)   
    Home_button.pack()
    Home_button.place(x=325,y=10)
    


    #re=tk.Button(window , text="Start" , command=lambda: record_audio(10))
   # re.pack()
    

    try:
        # Load your GIF image
        gif_path = relative_to_assets("ezgif.com-optimize.gif")
        print(f"GIF path: {gif_path}")  # Print the path for debugging
        gif = Image.open(gif_path)

        # Split GIF into frames
        gif_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

        # Convert frames to ImageTk format and preload
        image_references = [ImageTk.PhotoImage(frame) for frame in gif_frames]

        # Create an image on the canvas using the first frame of the GIF
        loaded_gif = canvas.create_image(
            170, 270,  # Adjust the coordinates to position the image where you want
            anchor="center",
            image=image_references[0]
        )

        Recording_text = canvas.create_text(
            115.0,
            250.0,
            anchor="nw",
            text="Recording",
            fill="#FFFFFF",
            font=("Chewy", 20)
        )


        #window.after(10000, transition_to_gui2)  # Transition to gui2.py after 10 seconds
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", window.quit)
        window.after(0, update_frame, 0)
        #window.after(0, on_window_open)
        window.mainloop()
        

    except Exception as e:
        print(f"Error drawing GUI: {e}")


if __name__ == "__main__":
   draw_gui()
   #record_audio(10)
   
  
  
    
   
