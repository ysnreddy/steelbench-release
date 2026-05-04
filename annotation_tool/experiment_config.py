"""Experiment configuration loader and validator.

Each experiment is defined by a YAML file in experiments/.
The config drives batch curation, annotator assignment, safety routing,
calibration, and deployment.

Usage:
    from annotation_tool.experiment_config import load_experiment
    config = load_experiment("experiments/pilot_v2.yaml")
"""

import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = str(Path(__file__).parent.parent)

# Required top-level keys
REQUIRED_KEYS = ["experiment_name", "clip_selection", "annotators"]

# Valid annotator roles
VALID_ROLES = {"tier_1", "tier_2", "tier_3"}

# Default values for optional fields
DEFAULTS = {
    "description": "",
    "clip_selection": {
        "size": 50,
        "targets": {},
        "min_per_class": 0,
        "unclassified_budget": 0,
        "exclude_annotated": True,
    },
    "assignment": {
        "double_annotate_ratio": 0.15,
        "decoy_ratio": 0.10,
        "calibration_size": 0,
        "experts_only": False,
        "tier2_audit_ratio": 0.15,
    },
    "calibration": {
        "subset_size": 0,
        "anchored_blind_split": 0.5,
        "stratify_by": ["action_class", "severity_level"],
        "calibration_dir": "active_batch/data",
    },
    "safety": {
        "auto_flag_zones": [],
        "escalate_all_unsafe_acts": True,
    },
}


def load_experiment(config_path):
    """Load and validate an experiment config YAML.

    Args:
        config_path: path to YAML file (absolute or relative to project root)

    Returns:
        Validated config dict with defaults filled in.

    Raises:
        FileNotFoundError: if config file doesn't exist
        ValueError: if config is invalid
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Empty experiment config")

    # Validate required keys
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required key: '{key}'")

    # Validate annotators
    annotators = config.get("annotators", {})
    if not annotators:
        raise ValueError("No annotators defined")

    for name, info in annotators.items():
        if isinstance(info, str):
            # Simple format: "ravi: tier_1"
            config["annotators"][name] = {"role": info}
            info = config["annotators"][name]
        role = info.get("role", "")
        if role not in VALID_ROLES:
            raise ValueError(
                f"Annotator '{name}' has invalid role '{role}'. "
                f"Valid roles: {VALID_ROLES}"
            )

    # Check at least one tier_1 (unless experts_only mode)
    experts_only = config.get("assignment", {}).get("experts_only", False)
    tier_1_count = sum(
        1 for a in config["annotators"].values() if a.get("role") == "tier_1"
    )
    tier_2_count = sum(
        1 for a in config["annotators"].values() if a.get("role") == "tier_2"
    )
    if tier_1_count == 0 and not experts_only:
        raise ValueError("At least one tier_1 annotator is required (or set experts_only: true)")
    if experts_only and tier_2_count == 0:
        raise ValueError("experts_only mode requires at least one tier_2 annotator")

    # Fill defaults for optional sections
    config.setdefault("description", DEFAULTS["description"])

    cs = config.get("clip_selection", {})
    for k, v in DEFAULTS["clip_selection"].items():
        cs.setdefault(k, v)
    config["clip_selection"] = cs

    # Type validation for clip_selection
    if not isinstance(cs.get("size"), int):
        raise ValueError(f"clip_selection.size must be int, got {type(cs['size'])}")
    if not isinstance(cs.get("targets"), dict):
        raise ValueError(f"clip_selection.targets must be dict, got {type(cs['targets'])}")
    if not isinstance(cs.get("min_per_class"), int):
        raise ValueError(f"clip_selection.min_per_class must be int, got {type(cs['min_per_class'])}")

    assignment = config.get("assignment", {})
    for k, v in DEFAULTS["assignment"].items():
        assignment.setdefault(k, v)
    config["assignment"] = assignment

    safety = config.get("safety", {})
    for k, v in DEFAULTS["safety"].items():
        safety.setdefault(k, v)
    config["safety"] = safety

    # Store source path
    config["_config_path"] = config_path

    return config


def get_annotators_by_role(config, role):
    """Get list of annotator names for a given role."""
    return [
        name for name, info in config.get("annotators", {}).items()
        if info.get("role") == role
    ]


def get_tier1_annotators(config):
    return get_annotators_by_role(config, "tier_1")


def get_tier2_annotators(config):
    return get_annotators_by_role(config, "tier_2")


def get_tier3_annotators(config):
    return get_annotators_by_role(config, "tier_3")


def get_safety_zones(config):
    """Get list of camera zones that are safety-critical."""
    return config.get("safety", {}).get("auto_flag_zones", [])


def get_clip_targets(config):
    """Get per-class clip targets from config."""
    return config.get("clip_selection", {}).get("targets", {})


def validate_experiment_dir(config):
    """Check that required data files exist for this experiment."""
    issues = []

    # Check camera_zones.yaml exists if safety zones configured
    if get_safety_zones(config):
        zones_path = os.path.join(
            PROJECT_ROOT, "annotation_tool", "config", "camera_zones.yaml"
        )
        if not os.path.exists(zones_path):
            issues.append(f"camera_zones.yaml not found at {zones_path}")

    # Check Tier A manifest exists
    output_dir = os.environ.get(
        "STEELBENCH_OUTPUT_DIR",
        os.path.join(PROJECT_ROOT, "output"),
    )
    manifest = os.path.join(output_dir, "metadata", "tier_a_manifest.csv")
    if not os.path.exists(manifest):
        issues.append(f"Tier A manifest not found at {manifest}")

    return issues
