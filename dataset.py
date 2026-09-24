"""
PainSense - PyTorch datasets using REAL SynPAIN video frames.

FrameDataset:
    Reads real cropped face frames from:
    data/videos/face_frames/<video_name>/*.jpg

SequenceDataset:
    Builds real consecutive 8-frame sequences from the same video.
    Each sequence receives the video's real Pain / NoPain label.

Labels:
    0 = NoPain
    1 = Pain
"""

import os
import glob
import re
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# Image transforms
# ---------------------------------------------------------

frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


train_frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_video_label(video_name):
    """
    Get the real SynPAIN label from the video folder name.

    Examples:
        1000026870_NoPain_man_Young -> 0
        1100031270_Pain_man_Young   -> 1
    """

    name = video_name.lower()

    if "_nopain_" in name:
        return 0

    if "_pain_" in name:
        return 1

    raise ValueError(
        f"Could not determine Pain/NoPain label from video name: {video_name}"
    )


def get_frame_number(path):
    """
    Extract frame number from names such as:
        frame_000001.jpg
        frame_000125.jpg
    """

    match = re.search(r"frame_(\d+)", os.path.basename(path))

    if match:
        return int(match.group(1))

    return -1


def list_labeled_videos(frames_dir=None):
    """
    SynPAIN videos whose folder name contains Pain or NoPain.

    Unlabeled folders such as video_001 are skipped.
    """

    if frames_dir is None:
        frames_dir = config.VIDEO_FACE_DIR

    labeled = []

    for folder in sorted(glob.glob(os.path.join(frames_dir, "*"))):
        if not os.path.isdir(folder):
            continue

        video_name = os.path.basename(folder)

        try:
            label = get_video_label(video_name)
        except ValueError:
            continue

        labeled.append((video_name, label, folder))

    return labeled


def frame_intensity(label, frame_number, frame_min, frame_max):
    """
    Generated intensity for one frame.

    NoPain videos stay at 0. Those clips transition into a
    non-pain expression, not a pain face.

    Pain videos map the first frame to 0 and the last frame
    to PAIN_SCORE_MAX. SynPAIN built these clips as a morph
    from neutral to the pain expression.
    """

    if label == 0:
        return 0.0

    span = max(frame_max - frame_min, 1)
    alpha = (frame_number - frame_min) / span
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha * float(config.PAIN_SCORE_MAX)


def export_intensity_manifest(frames_dir=None, manifest_path=None):
    """
    Write one row per labeled face frame:

        video, frame, frame_number, intensity, label_source
    """

    if frames_dir is None:
        frames_dir = config.VIDEO_FACE_DIR

    if manifest_path is None:
        manifest_path = config.INTENSITY_MANIFEST

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    rows = []

    for video_name, label, folder in list_labeled_videos(frames_dir):
        frame_files = sorted(
            glob.glob(os.path.join(folder, "*.jpg")),
            key=get_frame_number
        )

        if not frame_files:
            continue

        numbers = [get_frame_number(path) for path in frame_files]
        frame_min = numbers[0]
        frame_max = numbers[-1]
        source = (
            "non_pain_expression"
            if label == 0
            else "neutral_to_pain_progress"
        )

        for path, number in zip(frame_files, numbers):
            intensity = frame_intensity(
                label,
                number,
                frame_min,
                frame_max
            )
            rows.append((
                video_name,
                os.path.basename(path),
                number,
                f"{intensity:.4f}",
                source
            ))

    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            "video,frame,frame_number,intensity,label_source\n"
        )
        for row in rows:
            handle.write(",".join(str(part) for part in row) + "\n")

    print(
        f"Intensity manifest: {len(rows)} frames -> {manifest_path}"
    )

    return manifest_path


# ---------------------------------------------------------
# Frame Dataset
# ---------------------------------------------------------

class FrameDataset(Dataset):
    """
    Reads real cropped SynPAIN face frames.

    Expected structure:

    data/videos/face_frames/
        1000026870_NoPain_man_Young/
            frame_000001.jpg
            frame_000002.jpg
            ...

        1100031270_Pain_man_Young/
            frame_000001.jpg
            frame_000002.jpg
            ...
    """

    def __init__(
        self,
        frames_dir=None,
        train=True
    ):

        if frames_dir is None:
            frames_dir = config.VIDEO_FACE_DIR

        self.samples = []
        self.groups = []

        video_folders = sorted([
            folder
            for folder in glob.glob(os.path.join(frames_dir, "*"))
            if os.path.isdir(folder)
        ])

        if not video_folders:
            raise RuntimeError(
                f"No video face folders found in {frames_dir}"
            )

        for video_folder in video_folders:

            video_name = os.path.basename(video_folder)

            label = get_video_label(video_name)

            frame_files = sorted(
                glob.glob(os.path.join(video_folder, "*.jpg")),
                key=get_frame_number
            )

            for frame_path in frame_files:

                self.samples.append(
                    (frame_path, label)
                )

                # Store the video name.
                # This is important because validation must be
                # separated by video rather than by individual frame.
                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No face frames found in {frames_dir}"
            )

        self.transform = (
            train_frame_transform
            if train
            else frame_transform
        )

        print(
            f"FrameDataset: {len(self.samples)} real face frames "
            f"from {len(set(self.groups))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        return (
            self.transform(img),
            torch.tensor(label, dtype=torch.long)
        )


# ---------------------------------------------------------
# Sequence Dataset
# ---------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Builds REAL temporal sequences directly from SynPAIN
    video face frames.

    Example:

        frame_000001
        frame_000002
        frame_000003
        ...
        frame_000008

        becomes one sequence.

    Sequence shape:

        (T, C, H, W)

    Target shape:

        (T,)

    Since SynPAIN provides a video-level Pain/NoPain label,
    the label is repeated for every timestep.

    Example for a Pain video:

        target = [1,1,1,1,1,1,1,1]

    Example for a NoPain video:

        target = [0,0,0,0,0,0,0,0]
    """

    def __init__(
        self,
        seq_dir=None,
        train=True
    ):

        if seq_dir is None:
            seq_dir = config.VIDEO_FACE_DIR

        self.samples = []
        self.groups = []

        video_folders = sorted([
            folder
            for folder in glob.glob(os.path.join(seq_dir, "*"))
            if os.path.isdir(folder)
        ])

        if not video_folders:
            raise RuntimeError(
                f"No video face folders found in {seq_dir}"
            )

        sequence_length = config.SEQ_LEN

        for video_folder in video_folders:

            video_name = os.path.basename(video_folder)

            label = get_video_label(video_name)

            frame_files = sorted(
                glob.glob(os.path.join(video_folder, "*.jpg")),
                key=get_frame_number
            )

            # Build sequences ONLY from truly consecutive frame numbers.
            #
            # This prevents us from joining frame 20 and frame 22
            # when frame 21 was skipped by face detection.
            for start in range(
                0,
                len(frame_files) - sequence_length + 1
            ):

                candidate = frame_files[
                    start:start + sequence_length
                ]

                numbers = [
                    get_frame_number(path)
                    for path in candidate
                ]

                expected = list(
                    range(
                        numbers[0],
                        numbers[0] + sequence_length
                    )
                )

                if numbers != expected:
                    continue

                self.samples.append(
                    candidate
                )

                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No real {sequence_length}-frame sequences could "
                f"be created from {seq_dir}"
            )

        self.transform = frame_transform

        print(
            f"SequenceDataset: {len(self.samples)} real "
            f"{sequence_length}-frame sequences "
            f"from {len(set(self.groups))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_paths = self.samples[idx]

        video_name = self.groups[idx]

        label = get_video_label(video_name)

        tensors = []

        for frame_path in frame_paths:

            img = Image.open(frame_path).convert("RGB")

            tensors.append(
                self.transform(img)
            )

        seq_tensor = torch.stack(tensors)

        # Binary Pain/NoPain target repeated for each frame.
        target_tensor = torch.full(
            (len(frame_paths),),
            float(label),
            dtype=torch.float32
        )

        return seq_tensor, target_tensor


# ---------------------------------------------------------
# Intensity sequence dataset
# ---------------------------------------------------------

class IntensitySequenceDataset(Dataset):
    """
    Consecutive face frames with a generated 0-10 intensity target.

    Pain video: target rises from 0 at the first frame to 10 at the last.
    NoPain video: target is 0 on every frame.
    """

    def __init__(
        self,
        frames_dir=None,
        sequences_per_video=6,
        train=True
    ):
        if frames_dir is None:
            frames_dir = config.VIDEO_FACE_DIR

        self.samples = []
        self.groups = []
        sequence_length = config.SEQ_LEN

        for video_name, label, folder in list_labeled_videos(frames_dir):
            frame_files = sorted(
                glob.glob(os.path.join(folder, "*.jpg")),
                key=get_frame_number
            )

            if len(frame_files) < sequence_length:
                continue

            numbers = [get_frame_number(path) for path in frame_files]
            frame_min = numbers[0]
            frame_max = numbers[-1]

            valid_starts = []

            for start in range(0, len(frame_files) - sequence_length + 1):
                candidate = frame_files[start:start + sequence_length]
                window_numbers = [
                    get_frame_number(path)
                    for path in candidate
                ]
                expected = list(range(
                    window_numbers[0],
                    window_numbers[0] + sequence_length
                ))

                if window_numbers == expected:
                    valid_starts.append(start)

            if not valid_starts:
                continue

            pick_count = min(sequences_per_video, len(valid_starts))
            chosen = []

            if pick_count == 1:
                chosen = [valid_starts[len(valid_starts) // 2]]
            else:
                for index in range(pick_count):
                    position = round(
                        index * (len(valid_starts) - 1) / (pick_count - 1)
                    )
                    chosen.append(valid_starts[position])

            for start in sorted(set(chosen)):
                candidate = frame_files[start:start + sequence_length]
                intensities = [
                    frame_intensity(
                        label,
                        get_frame_number(path),
                        frame_min,
                        frame_max
                    )
                    for path in candidate
                ]
                self.samples.append((candidate, intensities))
                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No intensity sequences could be built from {frames_dir}"
            )

        self.transform = frame_transform

        print(
            f"IntensitySequenceDataset: {len(self.samples)} sequences "
            f"from {len(set(self.groups))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, intensities = self.samples[idx]
        tensors = []

        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            tensors.append(self.transform(img))

        seq_tensor = torch.stack(tensors)
        target_tensor = torch.tensor(intensities, dtype=torch.float32)
        return seq_tensor, target_tensor


# ---------------------------------------------------------
# Video-level train/validation split
# ---------------------------------------------------------

def split_dataset(
    dataset,
    val_split=config.VAL_SPLIT,
    seed=42
):
    """
    Split the dataset by VIDEO.

    Training data keeps the training augmentation.

    Validation data uses the normal validation transform
    without RandomHorizontalFlip or ColorJitter.
    """

    import copy

    if not hasattr(dataset, "groups"):
        raise RuntimeError(
            "Dataset does not contain video group information."
        )

    groups = sorted(set(dataset.groups))

    group_labels = {}

    for group in groups:
        group_labels[group] = get_video_label(group)

    pain_groups = [
        g for g in groups
        if group_labels[g] == 1
    ]

    nopain_groups = [
        g for g in groups
        if group_labels[g] == 0
    ]

    import random

    rng = random.Random(seed)

    rng.shuffle(pain_groups)
    rng.shuffle(nopain_groups)

    def split_groups(group_list):

        if len(group_list) <= 1:
            return group_list, []

        n_val = max(
            1,
            int(len(group_list) * val_split)
        )

        n_val = min(
            n_val,
            len(group_list) - 1
        )

        return (
            group_list[n_val:],
            group_list[:n_val]
        )

    train_pain, val_pain = split_groups(
        pain_groups
    )

    train_nopain, val_nopain = split_groups(
        nopain_groups
    )

    train_groups = set(
        train_pain + train_nopain
    )

    val_groups = set(
        val_pain + val_nopain
    )

    train_indices = [
        i
        for i, group in enumerate(dataset.groups)
        if group in train_groups
    ]

    val_indices = [
        i
        for i, group in enumerate(dataset.groups)
        if group in val_groups
    ]

    # Keep the original dataset for training.
    train_dataset = dataset

    # Make a separate copy for validation so that
    # validation does NOT use training augmentation.
    val_dataset = copy.copy(dataset)

    val_dataset.transform = frame_transform

    train_subset = torch.utils.data.Subset(
        train_dataset,
        train_indices
    )

    val_subset = torch.utils.data.Subset(
        val_dataset,
        val_indices
    )

    print(
        f"Video split: "
        f"{len(train_groups)} train videos, "
        f"{len(val_groups)} validation videos"
    )

    print(
        f"Samples: "
        f"{len(train_subset)} train, "
        f"{len(val_subset)} validation"
    )

    return train_subset, val_subset

    """
    Split by VIDEO, not by individual frame/sequence.

    This prevents frames from the same SynPAIN video appearing
    in both training and validation.
    """

    if not hasattr(dataset, "groups"):
        raise RuntimeError(
            "Dataset does not contain video group information."
        )

    groups = sorted(set(dataset.groups))

    # Determine label for each video.
    group_labels = {}

    for group in groups:
        group_labels[group] = get_video_label(group)

    pain_groups = [
        g for g in groups
        if group_labels[g] == 1
    ]

    nopain_groups = [
        g for g in groups
        if group_labels[g] == 0
    ]

    rng = random.Random(seed)

    rng.shuffle(pain_groups)
    rng.shuffle(nopain_groups)

    def split_groups(group_list):

        if len(group_list) <= 1:
            return group_list, []

        n_val = max(
            1,
            int(len(group_list) * val_split)
        )

        n_val = min(
            n_val,
            len(group_list) - 1
        )

        return (
            group_list[n_val:],
            group_list[:n_val]
        )

    train_pain, val_pain = split_groups(pain_groups)
    train_nopain, val_nopain = split_groups(nopain_groups)

    train_groups = set(train_pain + train_nopain)
    val_groups = set(val_pain + val_nopain)

    train_indices = [
        i
        for i, group in enumerate(dataset.groups)
        if group in train_groups
    ]

    val_indices = [
        i
        for i, group in enumerate(dataset.groups)
        if group in val_groups
    ]

    train_subset = torch.utils.data.Subset(
        dataset,
        train_indices
    )

    val_subset = torch.utils.data.Subset(
        dataset,
        val_indices
    )

    print(
        f"Video split: "
        f"{len(train_groups)} train videos, "
        f"{len(val_groups)} validation videos"
    )

    print(
        f"Samples: "
        f"{len(train_subset)} train, "
        f"{len(val_subset)} validation"
    )

    return train_subset, val_subset