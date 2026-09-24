import cv2
import os
import glob

# Folder containing the 40 unlabeled videos
VIDEO_DIR = "data/videos/unlabeled/videos"

# Folder where frames will be stored
FRAME_DIR = "data/videos/frames"

# Create the main frames directory
os.makedirs(FRAME_DIR, exist_ok=True)

# Find all MP4 videos
video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
video_files.sort()

print(f"Found {len(video_files)} videos.")
print("=" * 50)

for video_path in video_files:

    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a folder for this video's frames
    output_folder = os.path.join(FRAME_DIR, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_name}")
        continue

    frame_number = 0

    while True:

        ret, frame = cap.read()

        # Stop when video ends
        if not ret:
            break

        # Create frame filename
        frame_filename = os.path.join(
            output_folder,
            f"frame_{frame_number:04d}.jpg"
        )

        # Save frame
        cv2.imwrite(frame_filename, frame)

        frame_number += 1

    # Release video
    cap.release()

    print(f"{video_name}: {frame_number} frames extracted")

print("=" * 50)
print("Frame extraction completed!")