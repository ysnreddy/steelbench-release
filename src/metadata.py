"""Metadata logging: CSV manifest and per-clip detection JSONs."""

import csv
import glob
import json
import os

MANIFEST_COLUMNS = [
    "clip_id", "clip_path", "frames_dir", "source_video", "source_dir",
    "camera_id", "site", "work_area", "date", "source_start_sec", "source_end_sec",
    "clip_duration_sec", "total_persons_detected", "min_persons",
    "max_persons", "avg_persons", "person_positions", "avg_confidence",
    "max_confidence", "person_detection_ratio", "avg_bbox_area_ratio",
    "max_bbox_area_ratio", "min_bbox_area_ratio", "brisque_score",
    "mean_pixel_intensity", "visual_condition_auto", "extraction_timestamp",
]


class MetadataWriter:
    """Writes clip metadata to CSV manifest and per-clip JSON files."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "metadata", "clips_manifest.csv")
        self.detections_dir = os.path.join(output_dir, "metadata", "clip_detections")
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        os.makedirs(self.detections_dir, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """Write CSV header if file doesn't exist or is empty."""
        if not os.path.exists(self.manifest_path) or os.path.getsize(self.manifest_path) == 0:
            with open(self.manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()

    def append_to_manifest(self, row_dict):
        """Append a single clip row to the manifest CSV."""
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writerow(row_dict)

    def write_clip_detection_json(self, clip_id, detection_data):
        """Write per-clip detection detail JSON."""
        json_path = os.path.join(self.detections_dir, f"{clip_id}.json")
        with open(json_path, "w") as f:
            json.dump(detection_data, f, indent=2)


def compute_clip_summary(clip_id, clip_path, frames_dir, source_video, source_dir,
                         camera_id, date, start_sec, end_sec, sampled_detections,
                         quality_metrics, extraction_timestamp,
                         site="", work_area=""):
    """Compute summary statistics for a clip from its frame-level detections.

    Args:
        sampled_detections: list of dicts, each with keys:
            frame_index, timestamp_sec, num_persons, mean_intensity, persons (list)
        quality_metrics: dict from QualityScorer.score_frames()

    Returns:
        (manifest_row, detection_json) tuple
    """
    # Aggregate person counts
    person_counts = [d["num_persons"] for d in sampled_detections]
    frames_with_person = sum(1 for c in person_counts if c > 0)
    total_detections = sum(person_counts)

    # Aggregate confidence and bbox area
    all_confidences = []
    all_bbox_areas = []
    all_positions = set()
    for d in sampled_detections:
        for p in d["persons"]:
            all_confidences.append(p["confidence"])
            all_bbox_areas.append(p["bbox_area_ratio"])
            all_positions.add(p["position_x"])
            all_positions.add(p["position_depth"])

    manifest_row = {
        "clip_id": clip_id,
        "clip_path": clip_path,
        "frames_dir": frames_dir,
        "source_video": source_video,
        "source_dir": source_dir,
        "camera_id": camera_id,
        "site": site,
        "work_area": work_area,
        "date": date,
        "source_start_sec": start_sec,
        "source_end_sec": end_sec,
        "clip_duration_sec": round(end_sec - start_sec, 2),
        "total_persons_detected": total_detections,
        "min_persons": min(person_counts) if person_counts else 0,
        "max_persons": max(person_counts) if person_counts else 0,
        "avg_persons": round(sum(person_counts) / len(person_counts), 2) if person_counts else 0,
        "person_positions": ",".join(sorted(all_positions)),
        "avg_confidence": round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0,
        "max_confidence": round(max(all_confidences), 4) if all_confidences else 0,
        "person_detection_ratio": round(frames_with_person / len(sampled_detections), 4) if sampled_detections else 0,
        "avg_bbox_area_ratio": round(sum(all_bbox_areas) / len(all_bbox_areas), 6) if all_bbox_areas else 0,
        "max_bbox_area_ratio": round(max(all_bbox_areas), 6) if all_bbox_areas else 0,
        "min_bbox_area_ratio": round(min(all_bbox_areas), 6) if all_bbox_areas else 0,
        "brisque_score": quality_metrics["mean_brisque"],
        "mean_pixel_intensity": quality_metrics["mean_pixel_intensity"],
        "visual_condition_auto": quality_metrics["visual_condition_auto"],
        "extraction_timestamp": extraction_timestamp,
    }

    detection_json = {
        "clip_id": clip_id,
        "source_video": source_video,
        "clip_start_sec": start_sec,
        "clip_end_sec": end_sec,
        "camera_id": camera_id,
        "date": date,
        "representative_frames": sorted(
            glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        ) if os.path.exists(frames_dir) else [],
        "brisque_scores": quality_metrics["brisque_scores"],
        "mean_brisque": quality_metrics["mean_brisque"],
        "mean_pixel_intensity": quality_metrics["mean_pixel_intensity"],
        "visual_condition_auto": quality_metrics["visual_condition_auto"],
        "sampled_frames": sampled_detections,
    }

    return manifest_row, detection_json
