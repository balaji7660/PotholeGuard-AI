"""
Warning Manager for PotholeGuard-AI.
Controls voice alerts and UI notifications.
Prevents annoying repeated announcements for the same tracked pothole.
Enforces configurable cooldown windows and tracks active alerts.
"""
import time
from typing import Dict, Any, Optional, List
from tracking.tracker import PotholeTrack

class WarningManager:
    def __init__(self, cooldown_seconds: float = 3.5):
        self.cooldown_seconds = cooldown_seconds
        self.last_global_alert_time = 0.0
        self.announced_tracks: Dict[int, float] = {}

    def process_tracks_for_warnings(self, tracks: List[PotholeTrack]) -> Optional[Dict[str, Any]]:
        """
        Evaluate tracks and determine if a voice warning should be emitted.
        Returns warning payload or None if throttled/no new critical event.
        """
        now = time.time()
        if now - self.last_global_alert_time < self.cooldown_seconds:
            return None

        # Sort tracks by risk severity (highest risk first)
        sorted_tracks = sorted(
            tracks,
            key=lambda t: float(t.features.get("risk_score", 0.0)),
            reverse=True
        )

        for track in sorted_tracks:
            risk_class = track.features.get("risk_class", "SAFE")
            risk_score = float(track.features.get("risk_score", 0.0))
            track_id = track.track_id

            # Skip already warned tracks unless risk escalated significantly
            if track_id in self.announced_tracks:
                continue

            if risk_class == "DANGER":
                self.announced_tracks[track_id] = now
                self.last_global_alert_time = now
                track.warning_sent = True
                return {
                    "alert_type": "DANGER",
                    "track_id": track_id,
                    "voice_text": "Danger. Deep pothole ahead. Use caution.",
                    "risk_score": risk_score,
                    "size_class": track.features.get("size_class", "LARGE"),
                    "depth_class": track.features.get("depth_class", "DEEP")
                }
            elif risk_class == "HIGH_WARNING":
                self.announced_tracks[track_id] = now
                self.last_global_alert_time = now
                track.warning_sent = True
                return {
                    "alert_type": "HIGH_WARNING",
                    "track_id": track_id,
                    "voice_text": "Warning. Significant pothole approaching.",
                    "risk_score": risk_score,
                    "size_class": track.features.get("size_class", "MEDIUM"),
                    "depth_class": track.features.get("depth_class", "MEDIUM")
                }
            elif risk_class == "WARNING":
                self.announced_tracks[track_id] = now
                self.last_global_alert_time = now
                track.warning_sent = True
                return {
                    "alert_type": "WARNING",
                    "track_id": track_id,
                    "voice_text": "Warning. Pothole ahead.",
                    "risk_score": risk_score,
                    "size_class": track.features.get("size_class", "MEDIUM"),
                    "depth_class": track.features.get("depth_class", "MEDIUM")
                }
            elif risk_class == "UNCERTAIN":
                self.announced_tracks[track_id] = now
                self.last_global_alert_time = now
                track.warning_sent = True
                return {
                    "alert_type": "UNCERTAIN",
                    "track_id": track_id,
                    "voice_text": "Potential road hazard. Use caution.",
                    "risk_score": risk_score,
                    "size_class": track.features.get("size_class", "UNKNOWN"),
                    "depth_class": track.features.get("depth_class", "UNKNOWN")
                }

        # Cleanup old announced tracks
        active_ids = {t.track_id for t in tracks}
        self.announced_tracks = {
            t_id: t_time for t_id, t_time in self.announced_tracks.items()
            if t_id in active_ids
        }

        return None
