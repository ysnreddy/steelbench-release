"""Safety rules loading and per-clip prompt generation.

Loads rules from safety_rules.yaml (extracted from unsafe_acts.pdf) and
camera_zones.csv to build per-clip safety prompts for VLM annotation.

Zone detection hierarchy for applying rules:
    1. Camera metadata (camera_zones.csv zone_tags) — definitive
    2. VLM image detection + camera metadata hints — if no zone tag
    3. General rules only — fallback

Every clip gets at minimum the general/universal rules. Zone-tagged clips
get general PLUS zone-specific rules.
"""

import csv
import os
from collections import defaultdict

import yaml


def load_safety_rules(config_dir=None):
    """Load safety rules from YAML (extracted from unsafe_acts.pdf).

    Returns:
        dict with 'department_rules' and 'general_rules' keys.
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "config")

    rules_path = os.path.join(config_dir, "safety_rules.yaml")
    with open(rules_path) as f:
        return yaml.safe_load(f)


def load_camera_zones(config_dir=None):
    """Load camera zone mappings from CSV.

    Returns:
        dict mapping camera_id → {site, work_area, zone_tags: [list], ...}
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "config")

    zones_path = os.path.join(config_dir, "camera_zones.csv")
    cameras = {}
    if not os.path.exists(zones_path):
        return cameras

    with open(zones_path) as f:
        for row in csv.DictReader(f):
            cam_id = row.get("camera_id", "").strip()
            if not cam_id:
                continue
            zone_tags_str = row.get("zone_tags", "").strip()
            if not zone_tags_str:
                zone_tags_str = row.get("zone_tags_suggestion", "general_work").strip()
            zone_tags = [t.strip() for t in zone_tags_str.split(",") if t.strip()]
            cameras[cam_id] = {
                "site": row.get("site", "").strip(),
                "work_area": row.get("work_area", "").strip(),
                "zone_tags": zone_tags,
                "location_description": row.get("location_description", "").strip(),
            }
    return cameras


def _site_match(cam_site, dept_sites):
    """Fuzzy site match: trimmed, case-insensitive, substring either direction.

    Handles naming variants like 'CRM' vs 'CRM 1&2', 'BF' vs 'BF CDI', etc.
    """
    if not cam_site or not dept_sites:
        return False
    cs = cam_site.strip().lower()
    for ds in dept_sites:
        d = (ds or "").strip().lower()
        if not d:
            continue
        if cs == d or cs in d or d in cs:
            return True
    return False


def get_applicable_rules(camera_id, rules_config, camera_zones, site=None, work_area=None):
    """Get all safety rules applicable to a given camera.

    Matching strategy (in priority order):
      1. Site from the manifest matches `department_rules[*].sites` — use that dept's rules.
      2. If still nothing AND camera_zones has a valid (non-junk) zone_tag, fall back to
         zone_tag overlap with `department_rules[*].zone_tags`.
      3. Otherwise general-only; the VLM is instructed to infer zone from image.
    """
    general = rules_config.get("general_rules", {})
    dept_rules = rules_config.get("department_rules", {})

    result = {
        "camera_id": camera_id,
        "site": site or "",
        "work_area": work_area or "",
        "general_rules": general,
        "zone_rules": {},
        "match_basis": "none",
    }

    # 1. Primary: site-based matching (manifest metadata — trusted)
    for dept_name, dept_info in dept_rules.items():
        if _site_match(site, dept_info.get("sites", [])):
            result["zone_rules"][dept_name] = {
                "observations": dept_info.get("observations", {}),
                "matched_by": "site",
                "matched_site": site,
            }
    if result["zone_rules"]:
        result["match_basis"] = "site"
        return result

    # 2. Fallback: zone_tags from camera_zones.csv (only if present and not junk)
    # "junk" heuristic: tag starts with 'clip_' (common paste-error pattern)
    KNOWN_ZONE_TAGS = {
        "furnace_zone", "hot_area", "confined_space", "crane_zone",
        "vehicle_area", "near_machinery", "ground_level", "elevated",
        "restricted", "walkway", "storage_area", "locomotive_area",
        "gas_zone", "general_work",
    }
    cam_info = camera_zones.get(camera_id, {})
    raw_tags = cam_info.get("zone_tags", []) or []
    clean_tags = {t for t in raw_tags if t in KNOWN_ZONE_TAGS}

    if clean_tags:
        for dept_name, dept_info in dept_rules.items():
            dept_zones = set(dept_info.get("zone_tags", []) or [])
            if dept_zones & clean_tags:
                result["zone_rules"][dept_name] = {
                    "observations": dept_info.get("observations", {}),
                    "matched_by": "zone_tag",
                    "matched_zones": sorted(dept_zones & clean_tags),
                }
        if result["zone_rules"]:
            result["match_basis"] = "zone_tag"

    return result


def build_safety_prompt_block(camera_id, site, work_area, rules_config=None,
                               camera_zones=None, config_dir=None):
    """Build the safety rules text block for the VLM prompt.

    This is the per-clip safety section that gets inserted into the
    VLM annotation prompt. Includes site context and applicable rules.

    Args:
        camera_id: Camera identifier from manifest
        site: Site name from manifest (e.g., "BF CDI", "SMS 1")
        work_area: Work area from manifest
        rules_config: Pre-loaded rules (or None to load from disk)
        camera_zones: Pre-loaded camera zones (or None to load from disk)
        config_dir: Config directory path

    Returns:
        str: Formatted safety rules text for the VLM prompt
    """
    if rules_config is None:
        rules_config = load_safety_rules(config_dir)
    if camera_zones is None:
        camera_zones = load_camera_zones(config_dir)

    applicable = get_applicable_rules(camera_id, rules_config, camera_zones,
                                      site=site, work_area=work_area)

    lines = []
    lines.append(f"SITE CONTEXT: These frames are from the '{site}' area, work zone '{work_area}'.")
    lines.append("This metadata comes from the camera's mounting location and is trusted.")
    lines.append("")
    lines.append("SAFETY COMPLIANCE EVALUATION")
    lines.append("You are a safety officer reviewing this clip. Evaluate each worker against:")
    lines.append("  (a) the GENERAL safety rules below (apply everywhere in the plant), AND")
    lines.append("  (b) the SITE-SPECIFIC rules listed for this area (when a site match is found).")
    lines.append("When a site-specific rule (UA-DEPT-XX) and a general rule (UA-G-XX) both apply,")
    lines.append("prefer the site-specific one (it's more precise). Do NOT invent rules.")
    lines.append("")

    # General/universal rules (always included)
    general = applicable.get("general_rules", {})
    if general:
        lines.append("--- GENERAL SAFETY RULES (apply to all areas) ---")
        for rule_id, desc in sorted(general.items()):
            lines.append(f"  {rule_id}: {desc}")
        lines.append("")

    # Site-specific rules
    zone_rules = applicable.get("zone_rules", {})
    match_basis = applicable.get("match_basis", "none")
    if zone_rules:
        lines.append(f"--- SITE-SPECIFIC RULES (matched by {match_basis}: "
                     f"{site!r}) ---")
        for dept_name, dept_info in sorted(zone_rules.items()):
            lines.append(f"  [{dept_name.upper()}]")
            for rule_id, desc in sorted(dept_info.get("observations", {}).items()):
                lines.append(f"    {rule_id}: {desc}")
            lines.append("")
    else:
        lines.append("--- NO DIRECT SITE MATCH ---")
        lines.append(f"The site '{site}' did not match any department rule block directly.")
        lines.append("Infer the zone from the image itself (look for: furnace glow, hot metal,")
        lines.append("overhead crane, confined space, vehicles, conveyor, locomotive, rolling mill,")
        lines.append("gas cylinders, welding, etc.) and cite any relevant general rule accordingly.")
        lines.append("")

    lines.append("OUTPUT FORMAT for unsafe_act field:")
    lines.append('  "none" — if no violation observed')
    lines.append('  "UA-G-XX: description" — if a general rule is violated')
    lines.append('  "UA-DEPT-XX: description" — if a zone-specific rule is violated')
    lines.append('  "UA-G-XX | UA-BF-YY" — multiple violations, pipe-separated')
    lines.append('  "other: description" — if you observe an unsafe act not in the rules above')

    return "\n".join(lines)


def get_all_department_sites(rules_config=None, config_dir=None):
    """Get mapping of department names to their associated sites.

    Useful for building camera_zones.csv from manifest metadata.
    """
    if rules_config is None:
        rules_config = load_safety_rules(config_dir)

    dept_rules = rules_config.get("department_rules", {})
    result = {}
    for dept_name, dept_info in dept_rules.items():
        result[dept_name] = {
            "sites": dept_info.get("sites", []),
            "zone_tags": dept_info.get("zone_tags", []),
        }
    return result
