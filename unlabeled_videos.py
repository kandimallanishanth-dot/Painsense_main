import os
import glob
import shutil

# Original SynPain videos
SOURCE_DIR = "data/videos/raw/SynPain/Videos"

# New unlabeled dataset
DEST_DIR = "data/videos/unlabeled/videos"

# Create destination folder
os.makedirs(DEST_DIR, exist_ok=True)

# Find only MP4 videos
video_files = glob.glob(os.path.join(SOURCE_DIR, "*.mp4"))

# Sort them so numbering is consistent
video_files.sort()

print(f"Found {len(video_files)} videos.")

for index, video_path in enumerate(video_files, start=1):

    # New filename contains NO label or demographic information
    new_name = f"video_{index:03d}.mp4"

    destination = os.path.join(DEST_DIR, new_name)

    # Copy original video
    shutil.copy2(video_path, destination)

    print(f"{os.path.basename(video_path)}  ->  {new_name}")

print("\n================================")
print("Unlabeled dataset created!")
print(f"Total videos: {len(video_files)}")
print("================================")