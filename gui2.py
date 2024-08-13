import os

folder_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\ma_small\000New"  # Path to the folder "000New"

# Get all the file names in the folder
file_names = os.listdir(folder_path)

# Iterate through each file in the folder
for file_name in file_names:
    if file_name.endswith(".mp3"):  # Check if the file is an MP3 file
        new_file_name = file_name.lstrip("0")  # Remove leading zeros from the file name
        old_file_path = os.path.join(folder_path, file_name)  # Path to the old file
        new_file_path = os.path.join(folder_path, new_file_name)  # Path to the new file
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

print("MP3 files renamed successfully!")