import cv2
import config
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Input and output folders
INPUT_DIR = PROJECT_ROOT / "data" / "videos" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "data" / "videos" / "face_frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OpenCV's built-in face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Could not load face detector.")
    exit()

video_folders = sorted(
    [folder for folder in INPUT_DIR.iterdir() if folder.is_dir()]
)

print(f"Found {len(video_folders)} video frame folders")

total_processed = 0
total_faces = 0
total_failed = 0

for video_folder in video_folders:

    output_folder = OUTPUT_DIR / video_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(video_folder.glob("*.jpg"))

    print()
    print(f"Processing: {video_folder.name}")
    print(f"Frames found: {len(frame_files)}")

    for frame_path in frame_files:

        image = cv2.imread(str(frame_path))

        if image is None:
            total_failed += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        total_processed += 1

        if len(faces) == 0:
            total_failed += 1
            continue

        # Select the largest detected face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        # Add a small margin around the face
        margin = int(0.15 * max(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        face = image[y1:y2, x1:x2]

        # Resize to the same size expected by the CNN
        face = cv2.resize(face,(config.IMG_SIZE, config.IMG_SIZE))

        output_path = output_folder / frame_path.name

        cv2.imwrite(str(output_path), face)

        total_faces += 1

print()
print("======================================")
print("FACE CROPPING COMPLETE")
print("======================================")
print(f"Frames processed : {total_processed}")
print(f"Faces extracted  : {total_faces}")
print(f"Frames skipped   : {total_failed}")
print(f"Output folder    : {OUTPUT_DIR}")