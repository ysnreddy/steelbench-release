#!/usr/bin/env python3
"""Anonymize frames by detecting and blurring faces.

For each frame:
1. Run face detection (MediaPipe or YOLOv8-face)
2. Apply Gaussian blur to detected face regions
3. Save anonymized frame to output directory

At 7-10m surveillance distance, faces are <20px and mostly undetectable.
This is a precautionary measure for NeurIPS ethics compliance.

Usage:
    python scripts/anonymize_frames.py --input active_batch/frames --output steelbench_release/public/frames
"""

import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm


def anonymize_with_mediapipe(input_dir, output_dir, confidence=0.3, ids_file=None):
    import mediapipe as mp

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=confidence)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    total_frames = 0
    faces_found = 0
    faces_blurred = 0

    clip_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if ids_file:
        import json as _json
        wanted = set(_json.load(open(ids_file)))
        before = len(clip_dirs)
        clip_dirs = [d for d in clip_dirs if d.name in wanted]
        print(f"Filtered by ids-file: kept {len(clip_dirs)} of {before} clip directories")
    print(f"Processing {len(clip_dirs)} clip directories...")

    for clip_dir in tqdm(clip_dirs, desc="Anonymizing"):
        out_dir = output_path / clip_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_file in sorted(clip_dir.glob("frame_*.jpg")):
            total_frames += 1
            img = cv2.imread(str(frame_file))
            if img is None:
                # Copy file as-is if can't read
                import shutil
                shutil.copy2(str(frame_file), str(out_dir / frame_file.name))
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                faces_found += 1
                h, w = img.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    # Expand bbox by 20% for safety margin
                    pad_w = bbox.width * 0.2
                    pad_h = bbox.height * 0.2
                    x1 = max(0, int((bbox.xmin - pad_w) * w))
                    y1 = max(0, int((bbox.ymin - pad_h) * h))
                    x2 = min(w, int((bbox.xmin + bbox.width + pad_w) * w))
                    y2 = min(h, int((bbox.ymin + bbox.height + pad_h) * h))

                    if x2 > x1 and y2 > y1:
                        roi = img[y1:y2, x1:x2]
                        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                        img[y1:y2, x1:x2] = blurred
                        faces_blurred += 1

            cv2.imwrite(str(out_dir / frame_file.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    detector.close()

    print(f"\n{'='*50}")
    print(f"Anonymization complete")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Frames with faces detected: {faces_found} ({faces_found/total_frames*100:.2f}%)")
    print(f"  Total faces blurred: {faces_blurred}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")

    # Save report
    report = {
        "total_frames": total_frames,
        "frames_with_faces": faces_found,
        "faces_blurred": faces_blurred,
        "detection_rate": f"{faces_found/total_frames*100:.2f}%",
        "method": "mediapipe",
        "confidence_threshold": confidence,
        "note": "At 7-10m surveillance distance, faces are <20px. Low detection rate is expected.",
    }
    import json
    report_path = output_path.parent / "anonymization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Anonymize frames by blurring faces")
    parser.add_argument("--input", required=True, help="Input frames directory")
    parser.add_argument("--output", required=True, help="Output directory for anonymized frames")
    parser.add_argument("--method", default="mediapipe", choices=["mediapipe"],
                        help="Face detection method")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Face detection confidence threshold")
    parser.add_argument("--ids-file", default=None,
                        help="Optional JSON array of clip IDs; only those clip "
                             "directories will be processed.")
    args = parser.parse_args()

    if args.method == "mediapipe":
        anonymize_with_mediapipe(args.input, args.output, args.confidence,
                                 ids_file=args.ids_file)


if __name__ == "__main__":
    main()
