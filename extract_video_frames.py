import cv2
from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "raw" / "SynPain" / "Videos"
FRAME_DIR = PROJECT_ROOT / "data" / "videos" / "frames"

# Create output folder
FRAME_DIR.mkdir(parents=True, exist_ok=True)

# Find all MP4 videos
videos = sorted(VIDEO_DIR.glob("*.mp4"))

print(f"Found {len(videos)} videos")

if not videos:
    print("ERROR: No MP4 videos found.")
    print(f"Expected folder: {VIDEO_DIR}")
    exit()

for video_path in videos:

    # Folder name = video filename without .mp4
    video_name = video_path.stem

    output_dir = FRAME_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Processing: {video_path.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path.name}")
        continue

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Save frame
        frame_name = f"frame_{frame_number:06d}.jpg"
        frame_path = output_dir / frame_name

        cv2.imwrite(str(frame_path), frame)

    cap.release()

    print(f"  Extracted {frame_number} frames")

print()
print("====================================")
print("VIDEO FRAME EXTRACTION COMPLETE")
print("====================================")
print(f"Frames saved to: {FRAME_DIR}")