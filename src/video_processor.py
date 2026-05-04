"""Video processing: frame sampling, clip extraction, representative frame extraction."""

import os
import subprocess

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Site name normalization
# Maps variant spellings/casing to canonical site names.
# Canonical names should match the plant layout / camera_zones.yaml.
# ---------------------------------------------------------------------------
_SITE_NORMALIZATION = {
    # SMS variants
    "sms": "SMS",
    "sms 1": "SMS 1",
    "SMS - 1": "SMS 1",
    "SMS1": "SMS 1",
    # CRM variants
    "CRM - 1&2": "CRM 1&2",
    # Sinter Plant variants
    "SinterPlant": "Sinter Plant",
    # RMHP trailing space
    "RMHP ": "RMHP",
    # BF variants
    "BF Converyor": "BF Conveyor",
}


def normalize_site_name(site: str) -> str:
    """Normalize site name to canonical form."""
    site = site.strip()
    if site in _SITE_NORMALIZATION:
        return _SITE_NORMALIZATION[site]
    return site


# Work area normalization — maps variant spellings to canonical forms
_WORK_AREA_NORMALIZATION = {
    # Case variants
    "WELD MC": "WELD Mc",
    "wT3LOAD-SIDE": "WT3 Load Side",
    "WT3_EMPTY-SIDE": "WT3 Empty Side",
    "WT4_LOAD-SIDE": "WT4 Load Side",
    "WT4 EMPTY SIDE": "WT4 Empty Side",
    "WT2-LOAD_SIDE": "WT2 Load Side",
    "WT-1 Load Side": "WT1 Load Side",
    # Spacing/formatting
    "MOTER STORE": "Motor Store",
    "WORK SHOP": "Work Shop",
    "FBD-1 BAGGING AREA": "FBD-1 Bagging Area",
    "FBD -2 Bagging Area": "FBD-2 Bagging Area",
    "Go Down- South Side": "Go Down South Side",
    "HPN Loading platform": "HPN Loading Platform",
    "PCM LODDING STATION": "PCM Loading Station",
    "Crane Lavel HPTC": "Crane Level HPTC",
    "19.1-19.2TEND": "19.1-19.2 Tend",
    "KP 4-1 _4-2 Head End": "KP 4-1/4-2 Head End",
    "J3- C1 Tail End": "J3-C1 Tail End",
    "WLP- PTZ": "WLP PTZ",
    "Ground CC2": "Ground CC-2",
    "DC-Floor_left": "DC Floor Left",
    "PKP 20-1 MCC5": "PKP 20-1 MCC-5",
}


def normalize_work_area(work_area: str) -> str:
    """Normalize work area name to canonical form."""
    work_area = work_area.strip()
    if work_area in _WORK_AREA_NORMALIZATION:
        return _WORK_AREA_NORMALIZATION[work_area]
    return work_area


def get_video_info(video_path):
    """Get video metadata using OpenCV.

    Returns dict with fps, total_frames, duration_sec, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def sample_frames(video_path, sample_every_n=10, motion_threshold=5.0,
                   open_timeout_ms=30000, read_timeout_ms=60000,
                   sample_interval_sec=None):
    """Yield (frame_index, timestamp_sec, frame) for sampled frames.

    Two sampling modes:
    - sample_interval_sec (preferred): seek-based, samples 1 frame every N
      seconds by seeking directly. Much faster for sparse sampling (e.g., 1fps)
      as it skips decoding intermediate frames entirely.
    - sample_every_n: sequential read, process every Nth frame. Decodes all
      frames which is slow for large intervals.

    Args:
        video_path: path to video file
        sample_every_n: process every Nth frame (legacy, used if sample_interval_sec is None)
        motion_threshold: mean pixel difference below this = no motion (skip).
                          Set to 0 to disable. Only used in sequential mode.
        open_timeout_ms: timeout for opening the video stream (ms)
        read_timeout_ms: timeout for reading frames (ms)
        sample_interval_sec: if set, sample one frame every N seconds using seek
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if sample_interval_sec and sample_interval_sec > 0:
        # Seek-based sampling: jump directly to target timestamps
        # Much faster than sequential read for sparse sampling
        t = 0.0
        while t < duration_sec:
            target_frame = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break
            yield target_frame, round(t, 3), frame
            t += sample_interval_sec
    else:
        # Sequential read mode (legacy)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n == 0:
                timestamp_sec = frame_idx / fps if fps > 0 else 0
                yield frame_idx, round(timestamp_sec, 3), frame
            frame_idx += 1

    cap.release()


def extract_clip_ffmpeg(video_path, output_path, start_sec, duration_sec,
                        codec="libx264", fps=25):
    """Extract a clip from video using ffmpeg.

    Uses -ss before -i for fast seeking.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration_sec),
        "-c:v", codec,
        "-preset", "fast",
        "-crf", "23",
        "-an",  # no audio
        "-vsync", "cfr",  # -fps_mode requires ffmpeg 5.1+; -vsync works on 4.x+
        "-r", str(fps),
        "-y",  # overwrite
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    return output_path


def extract_representative_frames(video_path, output_dir, start_sec, clip_duration_sec,
                                  num_frames=8, quality=95):
    """Extract representative frames from a clip region of the source video.

    Frames are sampled evenly across clip duration (default 8 frames).
    Saved as frame_0.jpg through frame_{num_frames-1}.jpg.

    Returns list of saved frame paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute timestamps: 0%, 33%, 66%, ~100% of clip
    if num_frames == 4:
        offsets = [0.0, clip_duration_sec * 0.33, clip_duration_sec * 0.66,
                   clip_duration_sec - 0.1]
    else:
        offsets = [clip_duration_sec * i / (num_frames - 1) for i in range(num_frames)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    saved_paths = []

    for i, offset in enumerate(offsets):
        target_sec = start_sec + offset
        target_frame = int(target_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved_paths.append(frame_path)

    cap.release()
    return saved_paths


def read_frames_at_paths(frame_paths):
    """Read frames from saved JPEG paths. Returns list of numpy arrays."""
    frames = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    return frames


def parse_video_filename(filename, parent_dir=None):
    """Parse site, work area, and timestamp from video filename.

    Supports two naming conventions:
    1. Standard: {Site}_{Work Area}_main_{YYYYMMDDHHMMSS}.mp4
    2. Legacy:   {YYYYMMDD}_{start}_{end}_{seq}.mp4 (site info in parent folder)

    For legacy format, pass parent_dir to extract site/work_area from the
    folder name (e.g., 'SinterPlant_StockBin').

    Returns dict with site, work_area, camera_id, date, start_timestamp.
    """
    name = os.path.splitext(filename)[0]

    # Split on '_main_' to separate the identifier from the timestamp
    if "_main_" in name:
        prefix, timestamp = name.rsplit("_main_", 1)
        # Split prefix into site and work_area on the first underscore
        if "_" in prefix:
            site, work_area = prefix.split("_", 1)
        else:
            site = prefix
            work_area = ""
    else:
        # Legacy format: filename is timestamps only, site info in parent folder
        parts = name.split("_")
        timestamps = [p for p in parts if len(p) >= 8 and p[:8].isdigit()]
        timestamp = timestamps[0] if timestamps else ""

        # Extract site/work_area from parent directory name
        site = ""
        work_area = ""
        if parent_dir:
            dir_name = os.path.basename(parent_dir)
            # Skip date-only folder names (e.g., '18112025')
            if not (len(dir_name) == 8 and dir_name.isdigit()):
                # Remove @duration suffix if present (e.g., 'RERS_RERS-1_20260303@10')
                dir_clean = dir_name.split("@")[0] if "@" in dir_name else dir_name
                # Remove trailing timestamp if present
                dir_parts = dir_clean.rsplit("_", 1)
                if len(dir_parts) == 2 and len(dir_parts[1]) >= 8 and dir_parts[1][:8].isdigit():
                    dir_clean = dir_parts[0]
                # Split into site and work_area
                if "_" in dir_clean:
                    site, work_area = dir_clean.split("_", 1)
                else:
                    site = dir_clean

    # Build camera_id: spaces replaced with underscores for paths/clip IDs
    if site:
        camera_id = f"{site}_{work_area}".replace(" ", "_") if work_area else site.replace(" ", "_")
    else:
        camera_id = name.replace(" ", "_")

    date = ""
    if len(timestamp) >= 8:
        date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

    # Normalize site and work area names
    site = normalize_site_name(site)
    work_area = normalize_work_area(work_area)

    # Rebuild camera_id after normalization
    if site:
        camera_id = f"{site}_{work_area}".replace(" ", "_") if work_area else site.replace(" ", "_")

    return {
        "site": site,
        "work_area": work_area,
        "camera_id": camera_id,
        "date": date,
        "start_timestamp": timestamp,
    }
