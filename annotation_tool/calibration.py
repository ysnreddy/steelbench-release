"""Calibration subset management and decoy clip insertion.

Manages the blind calibration subset (250 clips with known ground truth)
and inserts decoy clips (wrong VLM labels) to measure annotator vigilance.
"""

import json
import os
import random


class CalibrationManager:
    """Manages calibration subsets and decoy clips."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.calibration_path = os.path.join(data_dir, "calibration_set.json")
        self.decoy_path = os.path.join(data_dir, "decoy_clips.json")
        os.makedirs(data_dir, exist_ok=True)
        self.calibration_set = self._load_json(self.calibration_path)
        self.decoy_clips = self._load_json(self.decoy_path)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_json(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def create_calibration_set(self, ground_truth_clips, n=250):
        """Create calibration subset from clips with known ground truth.

        Half will be shown WITH VLM labels (anchored), half WITHOUT (blind).

        Args:
            ground_truth_clips: list of dicts with clip_id, action_class,
                                spatial_tags, ppe, etc. (expert-verified)
            n: total calibration set size
        """
        selected = random.sample(ground_truth_clips, min(n, len(ground_truth_clips)))
        random.shuffle(selected)

        mid = len(selected) // 2
        anchored = selected[:mid]
        blind = selected[mid:]

        self.calibration_set = {
            "total": len(selected),
            "anchored_clips": {c["clip_id"]: c for c in anchored},
            "blind_clips": {c["clip_id"]: c for c in blind},
        }

        self._save_json(self.calibration_set, self.calibration_path)
        return {
            "total": len(selected),
            "anchored": len(anchored),
            "blind": len(blind),
        }

    def is_calibration_clip(self, clip_id):
        """Check if a clip is part of the calibration set."""
        anchored = self.calibration_set.get("anchored_clips", {})
        blind = self.calibration_set.get("blind_clips", {})
        return clip_id in anchored or clip_id in blind

    def get_calibration_condition(self, clip_id):
        """Get the calibration condition for a clip.

        Returns: "anchored" (show VLM label), "blind" (hide VLM label), or None
        """
        if clip_id in self.calibration_set.get("anchored_clips", {}):
            return "anchored"
        if clip_id in self.calibration_set.get("blind_clips", {}):
            return "blind"
        return None

    def get_ground_truth(self, clip_id):
        """Get ground truth annotation for a calibration clip."""
        anchored = self.calibration_set.get("anchored_clips", {})
        blind = self.calibration_set.get("blind_clips", {})
        return anchored.get(clip_id) or blind.get(clip_id)

    def create_decoy_set(self, verification_clip_ids, decoy_ratio=0.10,
                         ground_truth_map=None):
        """Create decoy clips with deliberately wrong VLM labels.

        Args:
            verification_clip_ids: list of clip_ids in the verification queue
            decoy_ratio: fraction of clips to make decoys (default 10%)
            ground_truth_map: dict {clip_id: correct_annotation} for decoys

        Returns:
            dict with decoy stats
        """
        from .agreement import ACTION_GROUPS

        n_decoys = max(1, int(len(verification_clip_ids) * decoy_ratio))
        decoy_ids = random.sample(
            verification_clip_ids,
            min(n_decoys, len(verification_clip_ids))
        )

        # All action classes for generating wrong labels
        all_actions = []
        for classes in ACTION_GROUPS.values():
            all_actions.extend(classes)

        decoys = {}
        for clip_id in decoy_ids:
            correct = ground_truth_map.get(clip_id, {}) if ground_truth_map else {}
            correct_action = correct.get("action_class", "")

            # Pick a wrong action from a different group
            wrong_actions = [a for a in all_actions if a != correct_action]
            wrong_action = random.choice(wrong_actions) if wrong_actions else ""

            decoys[clip_id] = {
                "correct_action": correct_action,
                "displayed_wrong_action": wrong_action,
                "is_decoy": True,
            }

        self.decoy_clips = decoys
        self._save_json(self.decoy_clips, self.decoy_path)

        return {
            "total_decoys": len(decoys),
            "decoy_ratio": round(len(decoys) / len(verification_clip_ids), 3),
        }

    def is_decoy(self, clip_id):
        """Check if a clip is a decoy."""
        return clip_id in self.decoy_clips

    def get_decoy_info(self, clip_id):
        """Get decoy information (wrong label to display)."""
        return self.decoy_clips.get(clip_id)

    def get_all_decoy_ids(self):
        """Get set of all decoy clip IDs."""
        return set(self.decoy_clips.keys())

    def create_batch_calibration_split(self, batch_clips, n=250, seed=42,
                                        stratify_by=("action_class", "severity_level")):
        """Anchored/blind calibration split that does NOT need ground truth.

        Used for Phase 2 anchoring bias measurement: stratified-sample n clips from
        the curated batch, split 50/50 anchored vs blind. The annotation UI hides
        the VLM prefill for blind clips so we can later measure how much anchoring
        the prefill introduces (anchored vs blind accuracy gap, computed against
        ground truth from expert review).

        Args:
            batch_clips: list of clip dicts with clip_id + stratify_by keys
            n: total calibration subset size
            seed: rng seed for reproducibility
            stratify_by: tuple of keys to stratify on

        Returns dict with totals.
        """
        from collections import defaultdict
        rng = random.Random(seed)
        if not batch_clips:
            return {"total": 0, "anchored": 0, "blind": 0}

        strata = defaultdict(list)
        for c in batch_clips:
            stratum = tuple(str(c.get(k, "")) for k in stratify_by)
            strata[stratum].append(c)

        total = len(batch_clips)
        # Proportional allocation per stratum
        selected = []
        for stratum, pool in strata.items():
            k_stratum = max(1, round(n * len(pool) / total))
            rng.shuffle(pool)
            selected.extend(pool[:k_stratum])

        # Trim to exactly n
        rng.shuffle(selected)
        selected = selected[:n]

        # Force even count so anchored/blind are balanced
        if len(selected) % 2 == 1:
            selected = selected[:-1]

        mid = len(selected) // 2
        anchored = selected[:mid]
        blind = selected[mid:]

        self.calibration_set = {
            "total": len(selected),
            "seed": seed,
            "stratify_by": list(stratify_by),
            "anchored_clips": {c["clip_id"]: {"clip_id": c["clip_id"]} for c in anchored},
            "blind_clips": {c["clip_id"]: {"clip_id": c["clip_id"]} for c in blind},
        }
        self._save_json(self.calibration_set, self.calibration_path)
        return {
            "total": len(selected),
            "anchored": len(anchored),
            "blind": len(blind),
        }
