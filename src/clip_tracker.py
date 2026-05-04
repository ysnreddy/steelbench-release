"""Merge person detection intervals and slice into fixed-duration clips."""


def build_intervals(detections_timeline, merge_gap_sec=5, padding_sec=2,
                    video_duration_sec=None):
    """Build merged person-visible intervals from detection timeline.

    Args:
        detections_timeline: list of (timestamp_sec, detected_bool) tuples, sorted by time
        merge_gap_sec: merge intervals closer than this
        padding_sec: add padding before/after each interval
        video_duration_sec: total video duration for clamping

    Returns:
        list of (start_sec, end_sec) tuples
    """
    # Extract intervals where person is detected
    raw_intervals = []
    current_start = None

    for ts, detected in detections_timeline:
        if detected and current_start is None:
            current_start = ts
        elif not detected and current_start is not None:
            raw_intervals.append((current_start, ts))
            current_start = None

    # Close final interval if still open
    if current_start is not None:
        last_ts = detections_timeline[-1][0]
        raw_intervals.append((current_start, last_ts))

    if not raw_intervals:
        return []

    # Merge intervals that are close together
    merged = [raw_intervals[0]]
    for start, end in raw_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_sec:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Add padding and clamp
    padded = []
    for start, end in merged:
        start = max(0, start - padding_sec)
        if video_duration_sec is not None:
            end = min(video_duration_sec, end + padding_sec)
        else:
            end = end + padding_sec
        padded.append((start, end))

    return padded


def slice_into_clips(intervals, clip_duration_sec=15, min_duration_sec=5):
    """Slice merged intervals into fixed-duration clip boundaries.

    Args:
        intervals: list of (start_sec, end_sec) tuples
        clip_duration_sec: target clip duration
        min_duration_sec: minimum duration to keep a clip (for tail segments)

    Returns:
        list of (start_sec, end_sec) tuples for each clip
    """
    clips = []
    for start, end in intervals:
        duration = end - start
        if duration < min_duration_sec:
            continue

        pos = start
        while pos + clip_duration_sec <= end:
            clips.append((round(pos, 2), round(pos + clip_duration_sec, 2)))
            pos += clip_duration_sec

        # Handle tail segment
        remaining = end - pos
        if remaining >= min_duration_sec:
            clips.append((round(pos, 2), round(end, 2)))

    return clips
