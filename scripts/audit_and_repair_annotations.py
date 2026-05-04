#!/usr/bin/env python3
"""Audit and repair SteelBench annotation files against the schema validator.

Categorizes every annotation file under <annotations_dir>/<annotator_id>/*.json
into one of:
- valid               — passes schema_validator.validate_annotation()
- invalid_recoverable — Layer 2 with >5 workers (legacy bonus per-person data).
                        Repaired in place: annotation_layer=1, bonus_per_person=True.
- invalid_requeue     — fails validation in a way that requires re-annotation.
                        Moved to <annotator_id>/.invalid/<clip>.json.<timestamp>.
- non_submitted       — status != 'submitted' (skipped/discarded/flagged).
                        Untouched.

Also rebuilds tier2_queue.json to drop entries pointing to clips that no
longer have a submitted/flagged tier_1 file after the move.

Usage:
    # Dry-run (default) — no file changes, just print report
    python scripts/audit_and_repair_annotations.py \\
        --annotations-dir /path/to/active_batch/annotations

    # Apply repairs and re-queue moves
    python scripts/audit_and_repair_annotations.py \\
        --annotations-dir /path/to/active_batch/annotations \\
        --tier2-queue /path/to/annotation_tool/data/assignments/tier2_queue.json \\
        --apply

The script never deletes data. invalid_requeue files are MOVED, not deleted,
to a .invalid/ subdir under the annotator's directory. Atomic backups are
created for in-place repairs (.bak suffix) before any rewrite.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_tool.schema_validator import (  # noqa: E402
    categorize_record,
    validate_annotation,
)


# ---------- Utility ----------


def _atomic_write(path: Path, data: dict) -> None:
    """Atomic JSON write: tmp file → rename. Caller should ensure backup exists."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _backup_path(path: Path) -> Path:
    """Backup file path (.bak suffix). Idempotent — same backup name regardless
    of how many times the script runs."""
    return path.with_suffix(path.suffix + ".bak")


def _invalid_dest(annotator_dir: Path, clip_filename: str) -> Path:
    """Where to move an invalid_requeue file."""
    invalid_dir = annotator_dir / ".invalid"
    invalid_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return invalid_dir / f"{clip_filename}.{ts}"


def _normalize_error_for_grouping(err: str) -> str:
    """Strip variable parts (P1, P2, ...) so errors group across persons."""
    return re.sub(r"\bP\d+", "P*", err)


# ---------- Main audit logic ----------


def audit_annotations(annotations_dir: Path) -> dict:
    """Walk annotations dir, categorize every file. Returns dict with:
    - by_category: {category: [(annotator_id, filename, full_path, errors)]}
    - per_annotator: {annotator: {category: count}}
    - error_counts: Counter of normalized error strings
    """
    by_category: dict[str, list] = {
        "valid": [],
        "invalid_recoverable": [],
        "invalid_requeue": [],
        "non_submitted": [],
    }
    per_annotator: dict[str, dict[str, int]] = {}
    error_counts: Counter = Counter()

    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations_dir does not exist: {annotations_dir}")

    for ann_subdir in sorted(annotations_dir.iterdir()):
        if not ann_subdir.is_dir():
            continue
        if ann_subdir.name.startswith("."):
            continue  # skip .invalid/ from a previous run, etc.
        annotator_id = ann_subdir.name

        for fp in sorted(ann_subdir.iterdir()):
            if not fp.is_file() or not fp.name.endswith(".json"):
                continue
            try:
                with open(fp) as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                # Treat unreadable files as invalid_requeue
                by_category["invalid_requeue"].append(
                    (annotator_id, fp.name, fp, [f"unreadable: {exc}"])
                )
                continue

            cat = categorize_record(rec)
            errors = validate_annotation(rec) if cat in ("invalid_requeue", "invalid_recoverable") else []
            by_category[cat].append((annotator_id, fp.name, fp, errors))

            per_annotator.setdefault(
                annotator_id,
                {"valid": 0, "invalid_recoverable": 0, "invalid_requeue": 0, "non_submitted": 0},
            )
            per_annotator[annotator_id][cat] += 1

            if cat == "invalid_requeue":
                for e in errors:
                    error_counts[_normalize_error_for_grouping(e)] += 1

    return {
        "by_category": by_category,
        "per_annotator": per_annotator,
        "error_counts": error_counts,
    }


# ---------- Repair operations ----------


def repair_recoverable(rec: dict) -> dict:
    """Normalize a bonus per-person record: set annotation_layer=1, set
    bonus_per_person=True, KEEP persons array as-is."""
    new_rec = dict(rec)
    new_rec["annotation_layer"] = 1
    new_rec["bonus_per_person"] = True
    # Add a marker for traceability
    new_rec["_audit_repair_timestamp"] = datetime.now().isoformat()
    new_rec["_audit_repair_action"] = "normalize_layer_to_1_keep_bonus_persons"
    return new_rec


def apply_recoverable_repairs(audit: dict, dry_run: bool) -> int:
    """For each invalid_recoverable record: backup + atomic rewrite."""
    n = 0
    for annotator_id, fname, fp, errors in audit["by_category"]["invalid_recoverable"]:
        n += 1
        print(f"  [{annotator_id}/{fname}] normalize annotation_layer=1, set bonus_per_person=True")
        if dry_run:
            continue
        with open(fp) as f:
            rec = json.load(f)
        # Backup once (idempotent — only if backup doesn't already exist)
        bak = _backup_path(fp)
        if not bak.exists():
            shutil.copy2(fp, bak)
        new_rec = repair_recoverable(rec)
        _atomic_write(fp, new_rec)
    return n


def apply_requeue_moves(audit: dict, dry_run: bool) -> int:
    """For each invalid_requeue file: move to .invalid/<clip>.json.<timestamp>."""
    n = 0
    for annotator_id, fname, fp, errors in audit["by_category"]["invalid_requeue"]:
        n += 1
        annotator_dir = fp.parent
        dest = _invalid_dest(annotator_dir, fname)
        print(f"  [{annotator_id}/{fname}] → .invalid/{dest.name}")
        if dry_run:
            continue
        # Make sure .invalid/ exists
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(fp), str(dest))
    return n


def cleanup_tier2_queue(
    audit: dict,
    annotations_dir: Path,
    tier2_queue_path: Path,
    dry_run: bool,
) -> int:
    """Drop tier2_queue.json entries for clips that no longer have a
    submitted/flagged tier_1 file in the annotations dir."""
    if not tier2_queue_path or not tier2_queue_path.exists():
        print(f"  (tier2_queue.json not found at {tier2_queue_path}, skipping cleanup)")
        return 0

    with open(tier2_queue_path) as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            print(f"  (tier2_queue.json is corrupt, skipping cleanup)")
            return 0
    if not isinstance(queue, list):
        print(f"  (tier2_queue.json is not a list, skipping cleanup)")
        return 0

    # Build the set of clip_ids that still have a tier_1 record after the move
    still_has_tier1: set[str] = set()
    for ann_subdir in annotations_dir.iterdir():
        if not ann_subdir.is_dir() or ann_subdir.name.startswith("."):
            continue
        for fp in ann_subdir.iterdir():
            if not fp.is_file() or not fp.name.endswith(".json"):
                continue
            try:
                with open(fp) as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("annotator_role") != "tier_1":
                continue
            if rec.get("status") not in ("submitted", "flagged"):
                continue
            cid = rec.get("clip_id")
            if cid:
                still_has_tier1.add(cid)

    new_queue = [q for q in queue if q.get("clip_id") in still_has_tier1]
    n_dropped = len(queue) - len(new_queue)

    if n_dropped == 0:
        print(f"  (no tier2_queue.json entries to drop)")
        return 0

    print(f"  dropping {n_dropped} tier2_queue.json entries (clips no longer have tier_1)")
    for q in queue:
        if q.get("clip_id") not in still_has_tier1:
            print(f"    - {q.get('clip_id')}: {q.get('reason', '?')}")

    if not dry_run:
        bak = _backup_path(tier2_queue_path)
        if not bak.exists():
            shutil.copy2(tier2_queue_path, bak)
        tmp = tier2_queue_path.with_suffix(tier2_queue_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(new_queue, f, indent=2)
        os.replace(tmp, tier2_queue_path)

    return n_dropped


# ---------- Reporting ----------


def print_report(audit: dict) -> None:
    by_cat = audit["by_category"]
    per_ann = audit["per_annotator"]
    err_counts = audit["error_counts"]

    total_submitted = sum(
        len(by_cat[c]) for c in ("valid", "invalid_recoverable", "invalid_requeue")
    )
    print()
    print("=" * 72)
    print("AUDIT REPORT")
    print("=" * 72)
    print(f"Total submitted records: {total_submitted}")
    print(f"  valid:               {len(by_cat['valid'])}")
    print(f"  invalid_recoverable: {len(by_cat['invalid_recoverable'])}  (will be repaired in place)")
    print(f"  invalid_requeue:     {len(by_cat['invalid_requeue'])}  (will be moved to .invalid/)")
    print(f"Non-submitted (skipped/discarded/flagged): {len(by_cat['non_submitted'])}")
    print()
    print("Per-annotator breakdown:")
    for ann in sorted(per_ann):
        c = per_ann[ann]
        total = sum(c.values())
        print(
            f"  {ann}: {total} → "
            f"valid={c['valid']}, "
            f"recoverable={c['invalid_recoverable']}, "
            f"requeue={c['invalid_requeue']}, "
            f"other={c['non_submitted']}"
        )

    if err_counts:
        print()
        print("Top errors for invalid_requeue records:")
        for err, count in err_counts.most_common(15):
            print(f"  {count:3d}× {err[:100]}")

    if by_cat["invalid_recoverable"]:
        print()
        print("invalid_recoverable records (will be repaired in place):")
        for ann_id, fname, _, _ in sorted(by_cat["invalid_recoverable"]):
            print(f"  {ann_id}/{fname}")

    if by_cat["invalid_requeue"]:
        print()
        print("invalid_requeue records (will be moved to .invalid/):")
        for ann_id, fname, _, errs in sorted(by_cat["invalid_requeue"]):
            short = "; ".join(
                sorted(set(_normalize_error_for_grouping(e)[:60] for e in errs[:5]))
            )
            print(f"  {ann_id}/{fname}")
            print(f"    {short}")
    print("=" * 72)


# ---------- Entry point ----------


def main():
    parser = argparse.ArgumentParser(
        description="Audit and repair SteelBench annotation files"
    )
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Path to active_batch/annotations dir",
    )
    parser.add_argument(
        "--tier2-queue",
        default=None,
        help="Path to tier2_queue.json (optional, will be cleaned up after re-queue)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply repairs and moves. Without this flag, only prints the report (dry-run).",
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir).resolve()
    tier2_queue_path = Path(args.tier2_queue).resolve() if args.tier2_queue else None
    dry_run = not args.apply

    print(f"Annotations dir: {annotations_dir}")
    print(f"Tier 2 queue:    {tier2_queue_path}")
    print(f"Mode:            {'DRY RUN (no changes)' if dry_run else 'APPLY (will modify files)'}")

    audit = audit_annotations(annotations_dir)
    print_report(audit)

    print()
    if dry_run:
        print("DRY RUN — no files were modified. Re-run with --apply to execute.")
    else:
        print("APPLYING REPAIRS...")
        n_repaired = apply_recoverable_repairs(audit, dry_run=False)
        print(f"  → repaired {n_repaired} bonus records in place")

        print()
        print("MOVING REQUEUE FILES...")
        n_moved = apply_requeue_moves(audit, dry_run=False)
        print(f"  → moved {n_moved} files to .invalid/")

        print()
        print("CLEANING UP TIER 2 QUEUE...")
        n_dropped = cleanup_tier2_queue(audit, annotations_dir, tier2_queue_path, dry_run=False)
        print(f"  → dropped {n_dropped} stale tier2_queue entries")

        print()
        print("DONE.")
        print("Re-run with --apply removed (default dry-run) to verify zero invalids remain.")


if __name__ == "__main__":
    main()
