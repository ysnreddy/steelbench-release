"""Visual quality scoring: BRISQUE + low-light detection."""

import os

import cv2
import numpy as np

# Locate BRISQUE model files relative to this package
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRISQUE_MODEL = os.path.join(_PACKAGE_ROOT, "brisque_model_live.yml")
_BRISQUE_RANGE = os.path.join(_PACKAGE_ROOT, "brisque_range_live.yml")


class QualityScorer:
    """Computes BRISQUE quality scores and mean pixel intensity."""

    def __init__(self, low_light_threshold=60):
        self.low_light_threshold = low_light_threshold
        self.brisque = None
        if os.path.exists(_BRISQUE_MODEL) and os.path.exists(_BRISQUE_RANGE):
            try:
                self.brisque = cv2.quality.QualityBRISQUE_create(
                    _BRISQUE_MODEL, _BRISQUE_RANGE
                )
            except Exception:
                self.brisque = None

    def compute_brisque(self, frame):
        """Compute BRISQUE score for a single frame. Lower = better quality."""
        if self.brisque is not None:
            try:
                score = self.brisque.compute(frame)
                return round(float(score[0]), 2)
            except Exception:
                return -1.0
        return -1.0

    def compute_mean_intensity(self, frame):
        """Compute mean pixel intensity of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return round(float(np.mean(gray)), 2)

    def classify_visual_condition(self, mean_intensity):
        """Auto-classify visual condition based on intensity."""
        if mean_intensity < self.low_light_threshold:
            return "low_light"
        return "clean"

    def score_frames(self, frames):
        """Score a list of frames, returning aggregate metrics.

        Returns dict with:
            brisque_scores: list of per-frame scores
            mean_brisque: average BRISQUE
            mean_pixel_intensity: average intensity across frames
            visual_condition_auto: auto-detected condition
        """
        brisque_scores = [self.compute_brisque(f) for f in frames]
        intensities = [self.compute_mean_intensity(f) for f in frames]

        valid_brisque = [s for s in brisque_scores if s >= 0]
        mean_brisque = round(np.mean(valid_brisque), 2) if valid_brisque else -1.0
        mean_intensity = round(np.mean(intensities), 2)

        return {
            "brisque_scores": brisque_scores,
            "mean_brisque": mean_brisque,
            "mean_pixel_intensity": mean_intensity,
            "visual_condition_auto": self.classify_visual_condition(mean_intensity),
        }
