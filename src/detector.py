"""YOLOv8 person detection wrapper with batched inference support."""

from ultralytics import YOLO


class PersonDetector:
    """Detects persons in frames using YOLOv8.

    Supports both single-frame and batched inference for GPU efficiency.
    """

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.35,
                 person_class_id=0, device="0", half=True):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.device = device
        self.half = half

    def _parse_results(self, results, frame_shapes):
        """Parse YOLO results into per-frame detection lists."""
        all_detections = []
        for i, result in enumerate(results):
            h, w = frame_shapes[i]
            frame_area = h * w
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_area = bbox_w * bbox_h
                bbox_area_ratio = bbox_area / frame_area

                # Position classification
                center_x = (x1 + x2) / 2
                if center_x < w / 3:
                    position_x = "left"
                elif center_x < 2 * w / 3:
                    position_x = "center"
                else:
                    position_x = "right"

                # Depth estimation based on bbox height relative to frame
                height_ratio = bbox_h / h
                position_depth = "foreground" if height_ratio > 0.3 else "background"

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4),
                    "bbox_area_ratio": round(bbox_area_ratio, 6),
                    "position_x": position_x,
                    "position_depth": position_depth,
                })
            all_detections.append(detections)
        return all_detections

    def detect(self, frame):
        """Run person detection on a single frame.

        Returns list of dicts with keys:
            bbox, confidence, bbox_area_ratio, position_x, position_depth
        """
        h, w = frame.shape[:2]
        results = self.model(frame, classes=[self.person_class_id],
                             conf=self.confidence_threshold, device=self.device,
                             half=self.half, verbose=False)
        return self._parse_results(results, [(h, w)])[0]

    def detect_batch(self, frames, batch_size=32):
        """Run person detection on a batch of frames for GPU efficiency.

        Args:
            frames: list of numpy arrays (BGR images)
            batch_size: number of frames per GPU batch

        Returns:
            list of detection lists (one per frame)
        """
        if not frames:
            return []

        frame_shapes = [(f.shape[0], f.shape[1]) for f in frames]
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_shapes = frame_shapes[i:i + batch_size]
            results = self.model(batch, classes=[self.person_class_id],
                                 conf=self.confidence_threshold, device=self.device,
                                 half=self.half, verbose=False)
            all_detections.extend(self._parse_results(results, batch_shapes))

        return all_detections
