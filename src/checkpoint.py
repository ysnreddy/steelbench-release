"""Resume/checkpoint logic for tracking processed videos across runs."""

import json
import os
from datetime import datetime


class CheckpointManager:
    """Tracks which videos have been fully processed, persisted to JSON."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        return {
            "completed_videos": [],
            "last_updated": None,
            "stats": {
                "videos_processed": 0,
                "clips_extracted": 0,
                "total_clip_duration_sec": 0.0,
            },
        }

    def _save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, video_path):
        """Check if a video has already been processed."""
        return os.path.abspath(video_path) in self.data["completed_videos"]

    def mark_completed(self, video_path, num_clips=0, clip_duration=0.0):
        """Mark a video as fully processed."""
        abs_path = os.path.abspath(video_path)
        if abs_path not in self.data["completed_videos"]:
            self.data["completed_videos"].append(abs_path)
            self.data["stats"]["videos_processed"] += 1
            self.data["stats"]["clips_extracted"] += num_clips
            self.data["stats"]["total_clip_duration_sec"] += clip_duration
            self._save()

    def get_stats(self):
        return self.data["stats"]
