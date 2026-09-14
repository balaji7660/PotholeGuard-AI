"""
simulation/road.py
Simulated 2D road environment for visualization.

Generates a bird's-eye view road image showing:
  - Lane boundaries
  - Lane centre line
  - Pothole hazard zones
  - Vehicle position
  - Predicted trajectory (all 3 actions)
  - Selected / safe trajectory highlighted
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# Colours (BGR)
C_ROAD     = (50,  50,  50)
C_LANE_BD  = (220, 220, 220)   # lane boundary white
C_CENTRE   = (0,   255, 255)   # lane centre cyan dashed
C_POTHOLE  = (0,   0,   180)   # pothole dark red
C_VEHICLE  = (0,   200, 0)     # vehicle green
C_TRAJ_SAFE   = (0,   255, 0)  # safe trajectory green
C_TRAJ_UNSAFE = (0,   0,   255)# unsafe trajectory red
C_TRAJ_NEUTRAL= (200, 200, 0)  # neutral trajectory yellow
C_SRL_BOX  = (255, 165, 0)     # SRL override orange

ACTION_NAMES = {0: "MAINTAIN", 1: "SHIFT LEFT", 2: "SHIFT RIGHT", 3: "EMERGENCY BRAKE"}
ACTION_ARROWS = {0: "→", 1: "↖", 2: "↗", 3: "⛔"}


class RoadRenderer:
    """
    Renders a top-down road with vehicle, potholes, and trajectories.

    Parameters
    ----------
    canvas_w, canvas_h : int
        Output image dimensions in pixels.
    n_lanes : int
        Number of lanes (1 = single lane, 2 = dual carriageway shown).
    """

    def __init__(
        self,
        canvas_w: int = 700,
        canvas_h: int = 400,
        n_lanes: int = 1,
    ) -> None:
        self.W = canvas_w
        self.H = canvas_h

        # Lane parameters (in pixel space)
        self.lane_w  = int(canvas_w * 0.40)   # single lane width in px
        self.lane_x0 = (canvas_w - self.lane_w) // 2   # left boundary x
        self.lane_x1 = self.lane_x0 + self.lane_w       # right boundary x
        self.cx      = canvas_w // 2                     # centre x

        self.veh_h   = int(canvas_h * 0.09)
        self.veh_w   = int(self.lane_w * 0.28)

    # ------------------------------------------------------------------

    def render(
        self,
        vehicle_x_norm: float = 0.0,           # lateral offset [-1, 1]
        vehicle_y_norm: float = 0.65,           # longitudinal position [0,1]
        potholes_norm:  List[dict] = None,      # list of {cx,cy,rx,ry} in [0,1]
        trajectories:   Optional[dict] = None, # {action: [(x_norm,y_norm),...]}
        selected_action: int = 0,
        collision_map:  Optional[dict] = None,  # {action: bool}
        srl_override:   bool = False,
        final_action:   int = 0,
        override_reason: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw and return the road canvas as a BGR numpy array.
        """
        canvas = np.full((self.H, self.W, 3), C_ROAD, dtype=np.uint8)

        # Road background
        cv2.rectangle(canvas, (self.lane_x0, 0), (self.lane_x1, self.H),
                      (70, 70, 70), -1)

        # Lane boundaries
        cv2.line(canvas, (self.lane_x0, 0), (self.lane_x0, self.H), C_LANE_BD, 3)
        cv2.line(canvas, (self.lane_x1, 0), (self.lane_x1, self.H), C_LANE_BD, 3)

        # Dashed centre line
        dash_len, gap = 20, 15
        y = 0
        while y < self.H:
            cv2.line(canvas, (self.cx, y), (self.cx, min(y + dash_len, self.H)), C_CENTRE, 1)
            y += dash_len + gap

        # Potholes
        if potholes_norm:
            for ph in potholes_norm:
                px = int(self.lane_x0 + ph["cx"] * self.lane_w)
                py = int(ph["cy"] * self.H)
                rx = max(10, int(ph.get("rx", 0.06) * self.lane_w))
                ry = max(8,  int(ph.get("ry", 0.05) * self.H))
                cv2.ellipse(canvas, (px, py), (rx, ry), 0, 0, 360, C_POTHOLE, -1)
                cv2.ellipse(canvas, (px, py), (rx, ry), 0, 0, 360, (0, 0, 100), 2)
                cv2.putText(canvas, "🕳", (px - 10, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Trajectories
        if trajectories:
            for action, wpts in trajectories.items():
                if not wpts:
                    continue
                unsafe = (collision_map or {}).get(action, False)
                is_selected = (action == selected_action)
                if unsafe:
                    color = C_TRAJ_UNSAFE
                elif is_selected:
                    color = C_TRAJ_SAFE if not srl_override else C_SRL_BOX
                else:
                    color = C_TRAJ_NEUTRAL

                thickness = 3 if is_selected else 1
                vx = int(self.cx + vehicle_x_norm * self.lane_w / 2)
                vy = int(vehicle_y_norm * self.H)
                prev_pt = (vx, vy)
                for x_n, y_n in wpts:
                    pt = (
                        int(self.cx + x_n * self.lane_w / 2),
                        int(vy - y_n * self.H * 0.5)   # forward = up
                    )
                    pt = (
                        np.clip(pt[0], self.lane_x0, self.lane_x1),
                        np.clip(pt[1], 0, self.H - 1),
                    )
                    cv2.line(canvas, prev_pt, pt, color, thickness)
                    prev_pt = pt

                # Arrowhead at end
                if wpts:
                    cv2.arrowedLine(canvas, prev_pt,
                                    (prev_pt[0], max(0, prev_pt[1] - 15)),
                                    color, thickness, tipLength=0.4)

        # Vehicle
        vx = int(self.cx + vehicle_x_norm * self.lane_w / 2)
        vy = int(vehicle_y_norm * self.H)
        vx = np.clip(vx, self.lane_x0 + self.veh_w // 2, self.lane_x1 - self.veh_w // 2)
        car_pts = np.array([
            [vx - self.veh_w // 2, vy - self.veh_h // 2],
            [vx + self.veh_w // 2, vy - self.veh_h // 2],
            [vx + self.veh_w // 2, vy + self.veh_h // 2],
            [vx - self.veh_w // 2, vy + self.veh_h // 2],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [car_pts], C_VEHICLE)
        cv2.polylines(canvas, [car_pts], True, (0, 150, 0), 2)

        # Windshield highlight
        cv2.rectangle(canvas,
                       (vx - self.veh_w//4, vy - self.veh_h//2 + 2),
                       (vx + self.veh_w//4, vy - self.veh_h//4),
                       (150, 230, 200), -1)

        # Action overlay
        action_text = f"{ACTION_ARROWS.get(final_action, '')} {ACTION_NAMES.get(final_action, '')}"
        color_text  = (0, 200, 255) if not srl_override else (0, 100, 255)
        cv2.putText(canvas, action_text, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color_text, 2)

        if srl_override and override_reason:
            short_reason = override_reason[:40]
            cv2.putText(canvas, f"SRL: {short_reason}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_SRL_BOX, 1)

        # Legend
        legend_items = [
            ("── Selected trajectory", C_TRAJ_SAFE),
            ("── Unsafe trajectory",   C_TRAJ_UNSAFE),
            ("── Other action",        C_TRAJ_NEUTRAL),
        ]
        for i, (txt, col) in enumerate(legend_items):
            cv2.putText(canvas, txt, (self.W - 200, 25 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        return canvas
