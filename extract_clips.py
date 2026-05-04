#!/usr/bin/env python3
"""Extract person-visible clips from CCTV footage for SteelBench dataset.

Supports parallel video decoding with shared GPU inference for speed.

Usage:
    python extract_clips.py --input-dir /<RAW_VIDEO_DIR>/
    python extract_clips.py --input-dir /<RAW_VIDEO_DIR>/ --workers 6
"""

import argparse
import glob
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yaml
from tqdm import tqdm

from src.checkpoint import CheckpointManager
from src.clip_tracker import build_intervals, slice_into_clips
from src.detector import PersonDetector
from src.metadata import MetadataWriter, compute_clip_summary
from src.quality import QualityScorer
from src.video_processor import (
    extract_clip_ffmpeg,
    extract_representative_frames,
    get_video_info,
    parse_video_filename,
    read_frames_at_paths,
    sample_frames,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_videos(input_dir):
    pattern = os.path.join(input_dir, "**", "*.mp4")
    return sorted(glob.glob(pattern, recursive=True))


def generate_clip_id(camera_id, date, start_sec, seq_num):
    hours = int(start_sec // 3600)
    minutes = int((start_sec % 3600) // 60)
    seconds = int(start_sec % 60)
    time_str = f"{hours:02d}{minutes:02d}{seconds:02d}"
    date_str = date.replace("-", "")
    return f"clip_{camera_id}_{date_str}_{time_str}_{seq_num:04d}"


def decode_video_frames(video_path, config):
    """CPU-bound: decode video and collect sampled frames.

    Uses sequential read at 1 frame/sec (sample_every_n=25 at 25fps).
    Sequential is faster than seek-based on HEVC videos because seeking
    requires re-decoding from the nearest keyframe.

    Returns list of (frame_idx, timestamp, frame_bgr) tuples.
    """
    collected = []
    sample_n = config.get("sample_every_n_frames", 25)
    motion_thresh = config.get("motion_threshold", 0)
    for frame_idx, ts, frame in sample_frames(
        video_path, sample_every_n=sample_n, motion_threshold=motion_thresh
    ):
        collected.append((frame_idx, ts, frame))
    return collected


def detect_persons_streaming(video_path, config, detector, gpu_batch_size=32):
    """Streaming approach: decode frames in chunks, run YOLO per chunk, free memory.

    Memory usage stays bounded to ~chunk_size frames at a time, regardless of
    video length. Critical for long videos where decoding everything first
    would use 10+ GB of RAM.

    Returns list of detection dicts (one per sampled frame), no raw frames retained.
    """
    sample_n = config.get("sample_every_n_frames", 25)
    motion_thresh = config.get("motion_threshold", 0)
    chunk_size = gpu_batch_size * 4  # 128 frames per chunk by default

    results = []
    chunk_frames = []
    chunk_meta = []

    def flush_chunk():
        if not chunk_frames:
            return
        # Compute mean intensity per frame (CPU-fast)
        intensities = []
        for frame in chunk_frames:
            if len(frame.shape) == 2:
                intensities.append(float(frame.mean()))
            else:
                intensity = (frame[:, :, 0].mean() * 0.114 +
                             frame[:, :, 1].mean() * 0.587 +
                             frame[:, :, 2].mean() * 0.299)
                intensities.append(float(intensity))
        # YOLO batched inference
        detections = detector.detect_batch(chunk_frames, batch_size=gpu_batch_size)
        for i, (frame_idx, ts) in enumerate(chunk_meta):
            persons = detections[i] if i < len(detections) else []
            results.append({
                "frame_index": frame_idx,
                "timestamp_sec": ts,
                "num_persons": len(persons),
                "mean_intensity": round(intensities[i], 2),
                "persons": persons,
            })
        # Free memory
        chunk_frames.clear()
        chunk_meta.clear()

    for frame_idx, ts, frame in sample_frames(
        video_path, sample_every_n=sample_n, motion_threshold=motion_thresh
    ):
        chunk_frames.append(frame)
        chunk_meta.append((frame_idx, ts))
        if len(chunk_frames) >= chunk_size:
            flush_chunk()

    flush_chunk()  # final partial chunk
    return results


def detect_persons_batch(frames_list, detector, gpu_batch_size=32):
    """GPU-bound: run YOLO on frames using batched inference.

    Processes frames in GPU batches of gpu_batch_size for maximum throughput
    on the A6000. FP16 is enabled by default in the detector.
    """
    import numpy as np

    if not frames_list:
        return []

    # Extract raw frames for batched detection
    raw_frames = [frame for _, _, frame in frames_list]

    # Compute mean intensity for all frames (CPU, fast)
    intensities = []
    for frame in raw_frames:
        if len(frame.shape) == 2:
            intensities.append(float(frame.mean()))
        else:
            intensity = (frame[:, :, 0].mean() * 0.114 +
                         frame[:, :, 1].mean() * 0.587 +
                         frame[:, :, 2].mean() * 0.299)
            intensities.append(float(intensity))

    # Batched YOLO inference on GPU
    all_detections = detector.detect_batch(raw_frames, batch_size=gpu_batch_size)

    # Combine results
    results = []
    for i, (frame_idx, ts, _) in enumerate(frames_list):
        persons = all_detections[i] if i < len(all_detections) else []
        results.append({
            "frame_index": frame_idx,
            "timestamp_sec": ts,
            "num_persons": len(persons),
            "mean_intensity": round(intensities[i], 2),
            "persons": persons,
        })

    return results


def extract_clips_from_detections(video_path, config, frame_detections_list):
    """Build intervals, slice clips, extract via ffmpeg, compute quality."""
    video_info = get_video_info(video_path)
    file_meta = parse_video_filename(
        os.path.basename(video_path),
        parent_dir=os.path.dirname(video_path),
    )
    camera_id = file_meta["camera_id"]
    site = file_meta.get("site", "")
    work_area = file_meta.get("work_area", "")
    date = file_meta["date"]
    output_dir = config["output_dir"]

    # Build timeline
    detections_timeline = [
        (d["timestamp_sec"], d["num_persons"] > 0) for d in frame_detections_list
    ]
    frame_detections = {d["timestamp_sec"]: d for d in frame_detections_list}

    if not detections_timeline:
        return []

    intervals = build_intervals(
        detections_timeline,
        merge_gap_sec=config["merge_gap_sec"],
        padding_sec=config["padding_sec"],
        video_duration_sec=video_info["duration_sec"],
    )
    clips = slice_into_clips(
        intervals,
        clip_duration_sec=config["clip_duration_sec"],
        min_duration_sec=config["clip_duration_sec"] * 0.33,
    )
    if not clips:
        return []

    quality_scorer = QualityScorer(
        low_light_threshold=config["low_light_intensity_threshold"]
    )

    results = []
    for seq_num, (clip_start, clip_end) in enumerate(clips):
        clip_duration = clip_end - clip_start
        clip_id = generate_clip_id(camera_id, date, clip_start, seq_num)

        date_dir = date.replace("-", "")
        clip_rel_path = os.path.join("clips", date_dir, camera_id, f"{clip_id}.mp4")
        clip_abs_path = os.path.join(output_dir, clip_rel_path)
        frames_rel_dir = os.path.join("frames", clip_id)
        frames_abs_dir = os.path.join(output_dir, frames_rel_dir)

        try:
            extract_clip_ffmpeg(
                video_path, clip_abs_path, clip_start, clip_duration,
                codec=config["output_codec"], fps=config["output_fps"],
            )
        except RuntimeError:
            continue

        try:
            frame_paths = extract_representative_frames(
                video_path, frames_abs_dir, clip_start, clip_duration,
                num_frames=config["num_representative_frames"],
                quality=config["frame_quality"],
            )
        except RuntimeError:
            frame_paths = []

        if frame_paths:
            rep_frames = read_frames_at_paths(frame_paths)
            quality_metrics = quality_scorer.score_frames(rep_frames)
        else:
            quality_metrics = {
                "brisque_scores": [], "mean_brisque": -1.0,
                "mean_pixel_intensity": -1.0, "visual_condition_auto": "unknown",
            }

        clip_detections = [
            frame_detections[ts] for ts in sorted(frame_detections.keys())
            if clip_start <= ts <= clip_end
        ]
        if not clip_detections:
            nearest = min(frame_detections.keys(),
                          key=lambda t: min(abs(t - clip_start), abs(t - clip_end)))
            clip_detections.append(frame_detections[nearest])

        manifest_row, detection_json = compute_clip_summary(
            clip_id=clip_id, clip_path=clip_rel_path, frames_dir=frames_rel_dir,
            source_video=os.path.abspath(video_path),
            source_dir=os.path.abspath(config["_input_dir"]),
            camera_id=camera_id, date=date,
            start_sec=clip_start, end_sec=clip_end,
            sampled_detections=clip_detections,
            quality_metrics=quality_metrics,
            extraction_timestamp=datetime.now().isoformat(),
            site=site, work_area=work_area,
        )
        results.append((manifest_row, detection_json, clip_id))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract person-visible clips from CCTV footage for SteelBench"
    )
    parser.add_argument("--input-dir", type=str,
                        help="Path to directory containing raw video files")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel video decode threads (default: 12)")
    parser.add_argument("--gpu-batch-size", type=int, default=32,
                        help="YOLO GPU batch size (default: 32, A6000 can handle 64)")
    parser.add_argument("--skip-legacy", action="store_true",
                        help="Skip legacy-format videos (no _main_ in filename)")
    parser.add_argument("--max-size-mb", type=int, default=0,
                        help="Only process videos under this size in MB (0=no limit)")
    parser.add_argument("--min-size-mb", type=int, default=0,
                        help="Only process videos above this size in MB (0=no limit)")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream frames through YOLO in chunks (low memory, for long videos)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    config = load_config(config_path)

    if args.input_dir:
        config["raw_video_dir"] = args.input_dir
    config["_input_dir"] = config["raw_video_dir"]

    input_dir = config["raw_video_dir"]
    output_dir = config["output_dir"]

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")

    videos = discover_videos(input_dir)
    print(f"Found {len(videos)} videos")

    if args.skip_legacy:
        before = len(videos)
        videos = [v for v in videos if "_main_" in os.path.basename(v)]
        print(f"Skipping {before - len(videos)} legacy videos (no _main_ in filename)")

    if args.max_size_mb > 0:
        before = len(videos)
        videos = [v for v in videos
                  if os.path.getsize(v) / (1024 * 1024) <= args.max_size_mb]
        print(f"Filtering to videos <= {args.max_size_mb}MB: {before - len(videos)} skipped")

    if args.min_size_mb > 0:
        before = len(videos)
        videos = [v for v in videos
                  if os.path.getsize(v) / (1024 * 1024) >= args.min_size_mb]
        print(f"Filtering to videos >= {args.min_size_mb}MB: {before - len(videos)} skipped")

    print(f"Processing {len(videos)} videos")

    if not videos:
        sys.exit(0)

    checkpoint_path = os.path.join(output_dir, "checkpoints", "progress.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path)
    metadata_writer = MetadataWriter(output_dir)

    remaining = [v for v in videos if not checkpoint_mgr.is_completed(v)]
    print(f"Already done: {len(videos) - len(remaining)}, Remaining: {len(remaining)}")

    if args.limit > 0:
        remaining = remaining[:args.limit]

    if not remaining:
        print("All done.")
        print(checkpoint_mgr.get_stats())
        sys.exit(0)

    # Load single shared YOLO model on GPU
    print(f"Loading YOLOv8 model: {config['yolo_model']}...")
    detector = PersonDetector(
        model_path=config["yolo_model"],
        confidence_threshold=config["confidence_threshold"],
        person_class_id=config["person_class_id"],
        device=config["device"],
    )

    num_workers = max(1, args.workers)
    print(f"Processing {len(remaining)} videos with {num_workers} decode threads...")
    motion_thresh = config.get("motion_threshold", 0)
    print(f"GPU batch size: {args.gpu_batch_size} (FP16 enabled)")
    print(f"Motion pre-filter: {'disabled' if motion_thresh == 0 else f'enabled (threshold={motion_thresh})'}")
    print("-" * 60)

    total_clips = 0
    _write_lock = threading.Lock()

    def process_one_video(video_path):
        """Process a single video end-to-end. Returns (video_path, num_clips)."""
        if args.streaming:
            # Streaming: bounded memory, decode + YOLO interleaved
            detections = detect_persons_streaming(
                video_path, config, detector, gpu_batch_size=args.gpu_batch_size
            )
        else:
            # Original: decode all, then YOLO all (high memory for long videos)
            frames = decode_video_frames(video_path, config)
            detections = detect_persons_batch(frames, detector, gpu_batch_size=args.gpu_batch_size)
            del frames  # Free memory immediately
        results = extract_clips_from_detections(video_path, config, detections)
        del detections

        num_clips = len(results)
        for manifest_row, detection_json, clip_id in results:
            metadata_writer.append_to_manifest(manifest_row)
            metadata_writer.write_clip_detection_json(clip_id, detection_json)

        checkpoint_mgr.mark_completed(
            video_path, num_clips, num_clips * config["clip_duration_sec"]
        )
        return video_path, num_clips

    if num_workers <= 1:
        # Single-threaded mode
        for video_path in tqdm(remaining, desc="Videos", unit="video"):
            filename = os.path.basename(video_path)
            try:
                _, num_clips = process_one_video(video_path)
                total_clips += num_clips
                tqdm.write(f"  {filename}: {num_clips} clips")
            except Exception as e:
                tqdm.write(f"  ERROR {filename}: {e}")
    else:
        # Pipeline: ThreadPoolExecutor decodes videos, main thread runs YOLO
        # Queue holds pre-decoded frames (bounded to limit memory)
        result_queue = queue.Queue(maxsize=num_workers)

        # Per-video decode timeout: 10 minutes max per video file
        VIDEO_DECODE_TIMEOUT = 600

        def decode_and_enqueue(video_path):
            """Decode a video and put results in the queue."""
            try:
                import signal
                import functools
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE

                # Use a sub-executor with timeout to kill hung decodes
                with _TPE(max_workers=1) as mini:
                    future = mini.submit(decode_video_frames, video_path, config)
                    try:
                        frames = future.result(timeout=VIDEO_DECODE_TIMEOUT)
                        result_queue.put((video_path, frames, None))
                    except _TE:
                        result_queue.put((video_path, None,
                                          f"Decode timeout ({VIDEO_DECODE_TIMEOUT}s)"))
            except Exception as e:
                result_queue.put((video_path, None, str(e)))

        def feed_videos():
            """Submit videos to the thread pool for decoding."""
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                pool.map(decode_and_enqueue, remaining)
            result_queue.put((None, None, None))  # Sentinel

        # Start feeder in background thread
        feeder = threading.Thread(target=feed_videos, daemon=True)
        feeder.start()

        pbar = tqdm(total=len(remaining), desc="Videos", unit="video")
        processed = 0

        while True:
            try:
                video_path, frames, error = result_queue.get(timeout=VIDEO_DECODE_TIMEOUT + 60)
            except queue.Empty:
                tqdm.write("  WARN: Queue read timed out, checking feeder...")
                if not feeder.is_alive():
                    break
                continue

            if video_path is None:
                break

            filename = os.path.basename(video_path)

            if error:
                tqdm.write(f"  DECODE ERROR {filename}: {error}")
                pbar.update(1)
                continue

            try:
                detections = detect_persons_batch(frames, detector, gpu_batch_size=args.gpu_batch_size)
                del frames
                results = extract_clips_from_detections(video_path, config, detections)
                del detections

                num_clips = len(results)
                total_clips += num_clips

                for manifest_row, detection_json, clip_id in results:
                    metadata_writer.append_to_manifest(manifest_row)
                    metadata_writer.write_clip_detection_json(clip_id, detection_json)

                checkpoint_mgr.mark_completed(
                    video_path, num_clips, num_clips * config["clip_duration_sec"]
                )
                tqdm.write(f"  {filename}: {num_clips} clips")
            except Exception as e:
                tqdm.write(f"  ERROR {filename}: {e}")

            pbar.update(1)
            processed += 1

        feeder.join(timeout=10)
        pbar.close()

    print("-" * 60)
    print(f"Done! {total_clips} clips from {len(remaining)} videos")
    print(f"Stats: {checkpoint_mgr.get_stats()}")
    print(f"Manifest: {metadata_writer.manifest_path}")


if __name__ == "__main__":
    main()
