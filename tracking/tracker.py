"""
Temporal Multi-Object Pothole Tracker for PotholeGuard-AI.
Maintains persistent track IDs across consecutive video frames using
Intersection-over-Union (IoU) and Centroid Distance matching with Exponential Moving Average (EMA) smoothing.
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class PotholeTrack:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], features: Dict[str, Any], alpha: float = 0.6):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.features = features
        self.alpha = alpha  # EMA smoothing factor
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.warning_sent = False
        self.last_warning_time = 0.0
        self.bbox_history: List[Tuple[int, int, int, int]] = [bbox]
        self.risk_history: List[Dict[str, float]] = [{
            "timestamp": time.time(),
            "risk_score": float(features.get("risk_score", 0.0)),
            "depth_score": float(features.get("depth_score", 0.0)),
            "confidence": float(features.get("confidence", 0.9))
        }]

    def update(self, bbox: Tuple[int, int, int, int], features: Dict[str, Any]):
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = time.time()
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)

        # EMA smoothing on bounding box
        x1 = int(self.alpha * bbox[0] + (1 - self.alpha) * self.bbox[0])
        y1 = int(self.alpha * bbox[1] + (1 - self.alpha) * self.bbox[1])
        x2 = int(self.alpha * bbox[2] + (1 - self.alpha) * self.bbox[2])
        y2 = int(self.alpha * bbox[3] + (1 - self.alpha) * self.bbox[3])
        self.bbox = (x1, y1, x2, y2)
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        # EMA smoothing on continuous numerical features
        for key in ["size_score", "depth_score", "uncertainty", "confidence", "path_relevance_score", "risk_score"]:
            if key in features and key in self.features:
                self.features[key] = round(
                    self.alpha * float(features[key]) + (1 - self.alpha) * float(self.features[key]),
                    4
                )
            elif key in features:
                self.features[key] = features[key]

        # Update discrete labels with latest values
        for label_key in ["size_class", "depth_class", "position_class", "relevance_class", "risk_class", "recommendation"]:
            if label_key in features:
                self.features[label_key] = features[label_key]


class PotholeTracker:
    def __init__(self, iou_threshold: float = 0.25, max_distance_px: float = 80.0, max_age_frames: int = 8):
        self.iou_threshold = iou_threshold
        self.max_distance_px = max_distance_px
        self.max_age_frames = max_age_frames
        self.next_track_id = 1
        self.tracks: List[PotholeTrack] = []

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter_area / float(areaA + areaB - inter_area)

    @staticmethod
    def compute_centroid_distance(c1: Tuple[int, int], c2: Tuple[int, int]) -> float:
        return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

    def update(self, detections: List[Dict[str, Any]]) -> List[PotholeTrack]:
        """
        Update tracks with new detections from current frame.
        detections: list of dicts with 'bbox' and other feature keys.
        """
        # Increment time since update for existing tracks
        for t in self.tracks:
            t.time_since_update += 1

        matched_tracks = set()
        matched_detections = set()

        # Match existing tracks with new detections using IoU + Centroid distance
        for d_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            det_centroid = ((det_bbox[0] + det_bbox[2]) // 2, (det_bbox[1] + det_bbox[3]) // 2)
            
            best_match_idx = -1
            best_score = -1.0

            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_tracks:
                    continue
                iou = self.compute_iou(track.bbox, det_bbox)
                dist = self.compute_centroid_distance(track.centroid, det_centroid)

                if iou >= self.iou_threshold or dist <= self.max_distance_px:
                    score = iou + (1.0 / (1.0 + dist / 20.0))
                    if score > best_score:
                        best_score = score
                        best_match_idx = t_idx

            if best_match_idx != -1:
                self.tracks[best_match_idx].update(det_bbox, det)
                matched_tracks.add(best_match_idx)
                matched_detections.add(d_idx)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = PotholeTrack(self.next_track_id, det["bbox"], det)
                self.next_track_id += 1
                self.tracks.append(new_track)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age_frames]

        return self.tracks
