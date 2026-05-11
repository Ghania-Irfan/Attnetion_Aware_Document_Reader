"""Gaze tracking with MediaPipe Face Mesh and iris-ratio based zone detection."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from config import Config

try:
    FACE_MESH_MODULE = mp.solutions.face_mesh
except Exception:
    try:
        from mediapipe.python.solutions import face_mesh as FACE_MESH_MODULE
    except Exception as exc:
        raise ImportError(
            "This project requires MediaPipe Solutions API (FaceMesh/Hands). "
            "Your current mediapipe package does not provide it. "
            "Use the project venv and reinstall dependencies from requirements.txt."
        ) from exc


@dataclass
class GazeResult:
    iris_ratio_horizontal: float
    iris_ratio_vertical: float
    zone_horizontal: str
    zone_vertical: str
    combined_zone: str
    dwell_progress_horizontal: float
    dwell_progress_vertical: float
    navigation_action: Optional[str]
    scroll_action: Optional[str]
    face_detected: bool
    # Additional debug info
    raw_h: float = 0.0
    raw_v: float = 0.0


class GazeTracker:
    """Tracks eye gaze zones (horizontal + vertical) and emits navigation/scroll actions."""

    def __init__(self) -> None:
        self._face_mesh = FACE_MESH_MODULE.FaceMesh(
            max_num_faces=Config.FACE_MESH_MAX_NUM_FACES,
            refine_landmarks=Config.FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=Config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )

        # Horizontal state
        self._current_zone_horizontal = "CENTER"
        self._zone_start_time_horizontal = time.monotonic()
        self._last_fire_time_horizontal = {"left": 0.0, "right": 0.0}
        
        # Vertical state (for auto-scroll)
        self._current_zone_vertical = "CENTER"
        self._zone_start_time_vertical = time.monotonic()
        self._last_fire_time_vertical = {"up": 0.0, "down": 0.0}
        self._scroll_repeat_timer = {"up": 0.0, "down": 0.0}
        
        # Track last scroll direction to prevent flipping
        self._active_scroll_direction = None
        
        # Gaze smoothing buffer
        self._gaze_buffer: deque = deque(maxlen=Config.GAZE_SMOOTH_SAMPLES)
        
        # For storing last few scroll zones to add hysteresis
        self._scroll_zone_history = deque(maxlen=5)
        
        # Debug
        self._frame_count = 0

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()

    def process(self, frame_bgr) -> GazeResult:
        """Process frame and return gaze metrics and optional navigation/scroll actions."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self._reset_state()
            self._active_scroll_direction = None
            self._gaze_buffer.clear()
            return GazeResult(
                iris_ratio_horizontal=0.5,
                iris_ratio_vertical=0.5,
                zone_horizontal="CENTER",
                zone_vertical="CENTER",
                combined_zone="CENTER",
                dwell_progress_horizontal=0.0,
                dwell_progress_vertical=0.0,
                navigation_action=None,
                scroll_action=None,
                face_detected=False,
                raw_h=0.5,
                raw_v=0.5,
            )

        landmarks = result.multi_face_landmarks[0].landmark
        
        # === HORIZONTAL GAZE (Left/Right) ===
        iris_x = landmarks[Config.IRIS_LANDMARK_INDEX].x
        left_corner_x = landmarks[Config.EYE_LEFT_CORNER_INDEX].x
        right_corner_x = landmarks[Config.EYE_RIGHT_CORNER_INDEX].x

        eye_width = max(abs(right_corner_x - left_corner_x), Config.EYE_WIDTH_EPSILON)
        eye_min_x = min(left_corner_x, right_corner_x)
        iris_ratio_horizontal = (iris_x - eye_min_x) / eye_width
        iris_ratio_horizontal = max(0.0, min(1.0, iris_ratio_horizontal))
        if Config.INVERT_GAZE_HORIZONTAL:
            iris_ratio_horizontal = 1.0 - iris_ratio_horizontal

        # Use smoothed value for zone detection to reduce jitter
        self._gaze_buffer.append((iris_ratio_horizontal, 0.5))  # Only need H for now
        smoothed_h = np.mean([s[0] for s in self._gaze_buffer])
        
        zone_horizontal = self._zone_from_ratio_horizontal(smoothed_h)
        
        # === VERTICAL GAZE (Up/Down) ===
        if Config.VERTICAL_GAZE_ENABLED:
            iris_y = landmarks[Config.IRIS_LANDMARK_INDEX].y
            
            left_eye_top    = min(landmarks[159].y, landmarks[145].y, landmarks[33].y)
            left_eye_bottom = max(landmarks[23].y,  landmarks[27].y,  landmarks[133].y)
            right_eye_top   = min(landmarks[386].y, landmarks[374].y, landmarks[362].y)
            right_eye_bottom = max(landmarks[253].y, landmarks[257].y, landmarks[263].y)
            
            eye_top_y    = (left_eye_top    + right_eye_top)    / 2
            eye_bottom_y = (left_eye_bottom + right_eye_bottom) / 2
            eye_height   = max(eye_bottom_y - eye_top_y, Config.EYE_WIDTH_EPSILON)
            
            raw_ratio = (iris_y - eye_top_y) / eye_height
            iris_ratio_vertical = max(0.0, min(1.0, raw_ratio))
        else:
            iris_ratio_vertical = 0.5
            
        zone_vertical = self._zone_from_ratio_vertical(iris_ratio_vertical)
        
        # === Combined Zone ===
        combined_zone = self._get_combined_zone(zone_horizontal, zone_vertical)
        scroll_direction = self._get_scroll_direction(combined_zone)
        
        # Add hysteresis to prevent rapid zone switching
        self._scroll_zone_history.append(scroll_direction)
        # Use most common direction in history for stability
        stable_scroll_direction = self._get_stable_scroll_direction()

        # Debug output more frequently for scroll zones
        self._frame_count += 1
        if self._frame_count % 15 == 0 or (combined_zone in ["TOP_LEFT", "BOTTOM_LEFT"]):
            up_arrow   = "⬆️" if zone_vertical == "TOP"    else "  "
            down_arrow = "⬇️" if zone_vertical == "BOTTOM" else "  "
            
            scroll_info = ""
            if combined_zone == "TOP_LEFT":
                scroll_info = " 🔄 SCROLL UP ZONE ⬆️"
            elif combined_zone == "BOTTOM_LEFT":
                scroll_info = " 🔄 SCROLL DOWN ZONE ⬇️"
                
            print(
                f"[GAZE] H={iris_ratio_horizontal:.2f}({zone_horizontal}) "
                f"V={iris_ratio_vertical:.2f}({up_arrow}{down_arrow}) "
                f"→ {combined_zone}{scroll_info}",
                flush=True,
            )
        
        # === Update dwell and actions ===
        dwell_progress_horizontal, navigation_action = self._update_dwell_and_action_horizontal(zone_horizontal)
        dwell_progress_vertical, scroll_action = self._update_dwell_and_action_vertical(
            zone_horizontal, zone_vertical, combined_zone, stable_scroll_direction
        )
        
        return GazeResult(
            iris_ratio_horizontal=iris_ratio_horizontal,
            iris_ratio_vertical=iris_ratio_vertical,
            zone_horizontal=zone_horizontal,
            zone_vertical=zone_vertical,
            combined_zone=combined_zone,
            dwell_progress_horizontal=dwell_progress_horizontal,
            dwell_progress_vertical=dwell_progress_vertical,
            navigation_action=navigation_action,
            scroll_action=scroll_action,
            face_detected=True,
            raw_h=iris_ratio_horizontal,
            raw_v=iris_ratio_vertical,
        )

    def _get_combined_zone(self, zone_horizontal: str, zone_vertical: str) -> str:
        """Get combined zone from horizontal and vertical zones."""
        if zone_horizontal == "LEFT":
            if zone_vertical == "TOP":
                return "TOP_LEFT"
            elif zone_vertical == "BOTTOM":
                return "BOTTOM_LEFT"
            else:
                return "LEFT"
        elif zone_horizontal == "RIGHT":
            if zone_vertical == "TOP":
                return "TOP_RIGHT"
            elif zone_vertical == "BOTTOM":
                return "BOTTOM_RIGHT"
            else:
                return "RIGHT"
        elif zone_vertical == "TOP":
            return "TOP"
        elif zone_vertical == "BOTTOM":
            return "BOTTOM"
        return "CENTER"

    def _get_scroll_direction(self, combined_zone: str) -> Optional[str]:
        """Get scroll direction from combined zone."""
        if combined_zone == "TOP_LEFT":
            return "up"
        elif combined_zone == "BOTTOM_LEFT":
            return "down"
        return None
    
    def _get_stable_scroll_direction(self) -> Optional[str]:
        """Get the most common scroll direction from recent history (hysteresis)."""
        if not self._scroll_zone_history:
            return None
        
        # Count occurrences
        up_count = self._scroll_zone_history.count("up")
        down_count = self._scroll_zone_history.count("down")
        
        # Require at least 60% agreement for 5-sample window (3 out of 5)
        total = len(self._scroll_zone_history)
        if up_count > down_count and up_count >= max(3, total * 0.6):
            return "up"
        elif down_count > up_count and down_count >= max(3, total * 0.6):
            return "down"
        
        # If no clear majority, keep previous direction or return None
        return self._active_scroll_direction

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_vertical(self, cap, seconds: float = 3.5) -> Tuple[float, float]:
        """Calibrate vertical gaze for auto-scroll (UP/DOWN detection)."""
        screen_w, screen_h = pyautogui.size()
        
        cv2.namedWindow(Config.CALIBRATION_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            Config.CALIBRATION_WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        
        dot_center_x = int(screen_w * Config.CALIBRATION_DOT_CENTER_X_RATIO)
        dot_top_y    = int(screen_h * Config.CALIBRATION_DOT_TOP_Y_RATIO)
        dot_bottom_y = int(screen_h * Config.CALIBRATION_DOT_BOTTOM_Y_RATIO)
        
        def draw_vertical_overlay(target: str, remaining: float, captured: int, current_value: float = 0.5) -> None:
            canvas = np.full(
                (screen_h, screen_w, 3),
                Config.CALIBRATION_BG_COLOR_BGR,
                dtype=np.uint8,
            )
            if target == "TOP":
                dot_y = dot_top_y
                instruction = "Look UP at the TOP dot"
            elif target == "BOTTOM":
                dot_y = dot_bottom_y
                instruction = "Look DOWN at the BOTTOM dot"
            else:
                dot_y = screen_h // 2
                instruction = "Look at the dot"
            
            cv2.line(canvas, (dot_center_x, 0), (dot_center_x, screen_h), (100, 100, 100), 1)
            cv2.line(canvas, (0, dot_y), (screen_w, dot_y), (100, 100, 100), 1)
            
            cv2.circle(
                canvas,
                (dot_center_x, dot_y),
                Config.CALIBRATION_DOT_RADIUS,
                Config.CALIBRATION_DOT_COLOR_BGR,
                -1,
            )
            
            meter_height = 200
            meter_y      = screen_h - meter_height - 50
            meter_width  = 30
            meter_x      = 50
            cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), 2)
            fill_height = int(current_value * meter_height)
            cv2.rectangle(canvas, (meter_x, meter_y + meter_height - fill_height),
                         (meter_x + meter_width, meter_y + meter_height), (0, 255, 0), -1)
            cv2.putText(canvas, f"{current_value:.2f}", (meter_x, meter_y - 10),
                       Config.DEBUG_FONT, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, "UP (0)",   (meter_x, meter_y - 25),               Config.DEBUG_FONT, 0.4, (0, 255, 0), 1)
            cv2.putText(canvas, "DOWN (1)", (meter_x, meter_y + meter_height + 15), Config.DEBUG_FONT, 0.4, (0, 255, 0), 1)
            
            info_lines = [
                "VERTICAL GAZE CALIBRATION (for Auto-Scroll)",
                instruction,
                f"Time left: {max(remaining, 0.0):.1f}s",
                f"Samples: {captured}",
                f"Current V ratio: {current_value:.3f}",
                "",
                "TIP: Lower ratio (0) = looking UP -> Scroll UP",
                "TIP: Higher ratio (1) = looking DOWN -> Scroll DOWN",
                "Keep your head still and only move your eyes!",
            ]
            y = Config.CALIBRATION_TEXT_Y
            for line in info_lines:
                cv2.putText(
                    canvas, line,
                    (Config.CALIBRATION_TEXT_X, y),
                    Config.DEBUG_FONT,
                    Config.CALIBRATION_TEXT_FONT_SCALE,
                    Config.CALIBRATION_TEXT_COLOR_BGR,
                    Config.CALIBRATION_TEXT_THICKNESS,
                )
                y += Config.CALIBRATION_TEXT_LINE_SPACING
            cv2.imshow(Config.CALIBRATION_WINDOW_NAME, canvas)
            cv2.waitKey(Config.CALIBRATION_WAITKEY_MS)
        
        def collect_vertical_target(target: str, secs: float) -> list:
            print(f"\n[CALIBRATE] Look {target} at the dot for {secs} seconds...", flush=True)
            settle_deadline = time.monotonic() + Config.CALIBRATION_SETTLE_SECONDS
            while time.monotonic() < settle_deadline:
                draw_vertical_overlay(target, settle_deadline - time.monotonic(), 0, 0.5)
            
            ratios: list = []
            deadline = time.monotonic() + secs
            last_print_time = time.monotonic()
            while time.monotonic() < deadline:
                ret, frame = cap.read()
                if not ret:
                    draw_vertical_overlay(target, deadline - time.monotonic(), len(ratios),
                                         ratios[-1] if ratios else 0.5)
                    continue
                result = self.process(frame)
                if result.face_detected:
                    ratios.append(result.iris_ratio_vertical)
                    if time.monotonic() - last_print_time > 0.5:
                        print(f"  Sample {len(ratios)}: {result.iris_ratio_vertical:.3f}", flush=True)
                        last_print_time = time.monotonic()
                draw_vertical_overlay(target, deadline - time.monotonic(), len(ratios),
                                     ratios[-1] if ratios else 0.5)
            return ratios
        
        try:
            print("\n" + "=" * 60)
            print("VERTICAL CALIBRATION STARTED")
            print("=" * 60)
            top_samples    = collect_vertical_target("TOP",    seconds)
            bottom_samples = collect_vertical_target("BOTTOM", seconds)
        finally:
            cv2.destroyWindow(Config.CALIBRATION_WINDOW_NAME)
        
        top_ok    = len(top_samples)    >= Config.CALIBRATION_MIN_SAMPLES_PER_TARGET
        bottom_ok = len(bottom_samples) >= Config.CALIBRATION_MIN_SAMPLES_PER_TARGET
        
        print(f"\n[CALIBRATION RESULTS]")
        print(f"  UP samples: {len(top_samples)} collected")
        if top_samples:
            print(f"    Range: [{min(top_samples):.3f} - {max(top_samples):.3f}]")
            print(f"    Mean:  {np.mean(top_samples):.3f}")
        print(f"  DOWN samples: {len(bottom_samples)} collected")
        if bottom_samples:
            print(f"    Range: [{min(bottom_samples):.3f} - {max(bottom_samples):.3f}]")
            print(f"    Mean:  {np.mean(bottom_samples):.3f}")
        
        if top_ok and bottom_ok:
            top_threshold    = max(0.20, min(0.45, max(top_samples)    + Config.CALIBRATION_MARGIN))
            bottom_threshold = max(0.55, min(0.80, min(bottom_samples) - Config.CALIBRATION_MARGIN))
            
            Config.TOP_ZONE_THRESHOLD    = top_threshold
            Config.BOTTOM_ZONE_THRESHOLD = bottom_threshold
            
            print(f"\nCALIBRATION COMPLETE!")
            print(f"   UP   threshold: V < {top_threshold:.3f}")
            print(f"   DOWN threshold: V > {bottom_threshold:.3f}")
        else:
            print(f"\nWARNING: Insufficient samples! Using defaults.")
        
        print("=" * 60 + "\n")
        return Config.TOP_ZONE_THRESHOLD, Config.BOTTOM_ZONE_THRESHOLD

    def calibrate_horizontal(self, cap, seconds: float = 3.0) -> float:
        """Calibrate horizontal gaze (left/right)."""
        print(f"[CALIBRATE HORIZONTAL] Look as far LEFT as comfortable for {seconds}s ...", flush=True)
        deadline  = time.monotonic() + seconds
        min_ratio = 1.0
        seen_face = False
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            result = self.process(frame)
            if result.face_detected:
                seen_face = True
                min_ratio = min(min_ratio, result.raw_h if hasattr(result, 'raw_h') else result.iris_ratio_horizontal)
        if seen_face:
            new_threshold = round(min_ratio + Config.CALIBRATION_MARGIN, 3)
            Config.LEFT_ZONE_THRESHOLD = new_threshold
            print(f"[CALIBRATE HORIZONTAL] LEFT_ZONE_THRESHOLD set to {new_threshold}", flush=True)
        else:
            print("[CALIBRATE HORIZONTAL] No face detected. Keeping existing thresholds.", flush=True)
        return min_ratio

    def calibrate_both(self, cap, left_seconds: float = 3.0, right_seconds: float = 3.0) -> Tuple[float, float]:
        """Calibrate horizontal gaze with fullscreen center/left/right dot guidance."""
        screen_w, screen_h = pyautogui.size()
        dot_y = int(screen_h * Config.CALIBRATION_DOT_Y_RATIO)
        dot_x_map = {
            "CENTER": int(screen_w * Config.CALIBRATION_DOT_CENTER_X_RATIO),
            "LEFT":   int(screen_w * Config.CALIBRATION_DOT_LEFT_X_RATIO),
            "RIGHT":  int(screen_w * Config.CALIBRATION_DOT_RIGHT_X_RATIO),
        }

        cv2.namedWindow(Config.CALIBRATION_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            Config.CALIBRATION_WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

        def draw_overlay(target: str, remaining: float, captured: int, current_h: float = 0.5) -> None:
            canvas = np.full(
                (screen_h, screen_w, 3),
                Config.CALIBRATION_BG_COLOR_BGR,
                dtype=np.uint8,
            )
            cv2.circle(
                canvas,
                (dot_x_map[target], dot_y),
                Config.CALIBRATION_DOT_RADIUS,
                Config.CALIBRATION_DOT_COLOR_BGR,
                -1,
            )
            
            meter_width  = 300
            meter_height = 20
            meter_x      = screen_w - meter_width - 50
            meter_y      = 50
            cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), 2)
            fill_width = int(current_h * meter_width)
            cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + fill_width, meter_y + meter_height), (0, 255, 0), -1)
            cv2.putText(canvas, f"H: {current_h:.2f}", (meter_x, meter_y - 10),
                       Config.DEBUG_FONT, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, "LEFT (0)",  (meter_x - 50,               meter_y + 15), Config.DEBUG_FONT, 0.4, (0, 255, 0), 1)
            cv2.putText(canvas, "RIGHT (1)", (meter_x + meter_width + 10, meter_y + 15), Config.DEBUG_FONT, 0.4, (0, 255, 0), 1)
            
            info_lines = [
                "HORIZONTAL GAZE CALIBRATION",
                f"Target: {target}",
                f"Time left: {max(remaining, 0.0):.1f}s",
                f"Samples: {captured}",
                "Keep your head still and only move your eyes!",
            ]
            y = Config.CALIBRATION_TEXT_Y
            for line in info_lines:
                cv2.putText(
                    canvas, line,
                    (Config.CALIBRATION_TEXT_X, y),
                    Config.DEBUG_FONT,
                    Config.CALIBRATION_TEXT_FONT_SCALE,
                    Config.CALIBRATION_TEXT_COLOR_BGR,
                    Config.CALIBRATION_TEXT_THICKNESS,
                )
                y += Config.CALIBRATION_TEXT_LINE_SPACING
            cv2.imshow(Config.CALIBRATION_WINDOW_NAME, canvas)
            cv2.waitKey(Config.CALIBRATION_WAITKEY_MS)

        def collect_target(target: str, seconds: float) -> list:
            print(f"[CALIBRATE] Look at the {target} dot for {seconds}s ...", flush=True)
            settle_deadline = time.monotonic() + Config.CALIBRATION_SETTLE_SECONDS
            while time.monotonic() < settle_deadline:
                draw_overlay(target, settle_deadline - time.monotonic(), 0, 0.5)

            ratios: list = []
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                ret, frame = cap.read()
                if not ret:
                    draw_overlay(target, deadline - time.monotonic(), len(ratios),
                                ratios[-1] if ratios else 0.5)
                    continue
                result = self.process(frame)
                if result.face_detected:
                    ratios.append(result.raw_h if hasattr(result, 'raw_h') else result.iris_ratio_horizontal)
                draw_overlay(target, deadline - time.monotonic(), len(ratios),
                            ratios[-1] if ratios else 0.5)
            return ratios

        try:
            print("\n" + "=" * 60)
            print("HORIZONTAL CALIBRATION STARTED")
            print("=" * 60)
            center_samples = collect_target("CENTER", min(left_seconds, right_seconds))
            left_samples   = collect_target("LEFT",   left_seconds)
            right_samples  = collect_target("RIGHT",  right_seconds)
        finally:
            cv2.destroyWindow(Config.CALIBRATION_WINDOW_NAME)

        left_ok  = len(left_samples)  >= Config.CALIBRATION_MIN_SAMPLES_PER_TARGET
        right_ok = len(right_samples) >= Config.CALIBRATION_MIN_SAMPLES_PER_TARGET

        if left_ok:
            left_threshold = max(0.20, min(0.45, max(left_samples) + Config.CALIBRATION_MARGIN))
            Config.LEFT_ZONE_THRESHOLD = left_threshold
            print(f"\nLEFT threshold: H < {left_threshold:.3f} = looking LEFT")
        else:
            print(f"\nInsufficient LEFT samples. Keeping existing threshold.")

        if right_ok:
            right_threshold = max(0.55, min(0.80, min(right_samples) - Config.CALIBRATION_MARGIN))
            Config.RIGHT_ZONE_THRESHOLD = right_threshold
            print(f"RIGHT threshold: H > {right_threshold:.3f} = looking RIGHT")
        else:
            print(f"Insufficient RIGHT samples. Keeping existing threshold.")

        print("=" * 60 + "\n")
        return Config.LEFT_ZONE_THRESHOLD, Config.RIGHT_ZONE_THRESHOLD

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zone_from_ratio_horizontal(self, iris_ratio: float) -> str:
        if iris_ratio < Config.LEFT_ZONE_THRESHOLD:
            return "LEFT"
        if iris_ratio > Config.RIGHT_ZONE_THRESHOLD:
            return "RIGHT"
        return "CENTER"

    def _zone_from_ratio_vertical(self, iris_ratio: float) -> str:
        if Config.VERTICAL_GAZE_ENABLED:
            if iris_ratio < Config.TOP_ZONE_THRESHOLD:
                return "TOP"
            if iris_ratio > Config.BOTTOM_ZONE_THRESHOLD:
                return "BOTTOM"
        return "CENTER"

    def _update_dwell_and_action_horizontal(self, zone: str) -> Tuple[float, Optional[str]]:
        now = time.monotonic()
        if zone != self._current_zone_horizontal:
            self._current_zone_horizontal   = zone
            self._zone_start_time_horizontal = now

        if zone == "CENTER":
            return 0.0, None

        dwell_elapsed  = now - self._zone_start_time_horizontal
        dwell_progress = min(dwell_elapsed / Config.DWELL_SECONDS, 1.0)

        action          = "left" if zone == "LEFT" else "right"
        cooldown_elapsed = now - self._last_fire_time_horizontal[action]
        if dwell_elapsed >= Config.DWELL_SECONDS and cooldown_elapsed >= Config.ACTION_COOLDOWN_SECONDS:
            self._last_fire_time_horizontal[action] = now
            self._zone_start_time_horizontal        = now
            return 1.0, action

        return dwell_progress, None

    def _update_dwell_and_action_vertical(
        self, zone_horizontal: str, zone_vertical: str, combined_zone: str, scroll_direction: Optional[str]
    ) -> Tuple[float, Optional[str]]:
        if not Config.AUTO_SCROLL_ENABLED:
            return 0.0, None
            
        now = time.monotonic()
        
        # Check if user is looking at LEFT edge for scrolling
        is_scroll_zone = (zone_horizontal == "LEFT" and
                         (zone_vertical == "TOP" or zone_vertical == "BOTTOM"))
        
        if not is_scroll_zone:
            if self._active_scroll_direction is not None:
                self._active_scroll_direction = None
                print(f"[SCROLL] Exited scroll zone", flush=True)
            self._zone_start_time_vertical = now
            return 0.0, None
        
        if scroll_direction is None:
            return 0.0, None
            
        # New direction or first time entering
        if self._active_scroll_direction != scroll_direction:
            self._active_scroll_direction   = scroll_direction
            self._zone_start_time_vertical  = now
            self._scroll_repeat_timer[scroll_direction] = 0.0
            direction_name = "UP" if scroll_direction == "up" else "DOWN"
            print(f"[SCROLL] Entered {direction_name} scroll zone - hold gaze to start scrolling", flush=True)
            return 0.0, None
        
        # Calculate dwell time
        dwell_elapsed = now - self._zone_start_time_vertical
        
        # Show progress during initial dwell
        if dwell_elapsed < Config.SCROLL_DWELL_SECONDS:
            dwell_progress = dwell_elapsed / Config.SCROLL_DWELL_SECONDS
            # Only print progress occasionally to avoid spam
            if self._frame_count % 10 == 0:
                print(f"[SCROLL] Dwell progress: {dwell_progress:.0%}", flush=True)
            return dwell_progress, None
        
        # Dwell threshold reached - start or continue scrolling
        dwell_progress = 1.0
        
        # Check cooldown for first scroll
        cooldown_elapsed = now - self._last_fire_time_vertical[scroll_direction]
        repeat_elapsed = now - self._scroll_repeat_timer[scroll_direction]
        
        # First scroll or after cooldown
        if cooldown_elapsed >= Config.SCROLL_COOLDOWN_SECONDS:
            self._last_fire_time_vertical[scroll_direction] = now
            self._scroll_repeat_timer[scroll_direction] = now
            direction_arrow = "UP" if scroll_direction == "up" else "DOWN"
            print(f"[SCROLL] 🔄 {direction_arrow} - scrolling started!", flush=True)
            return dwell_progress, scroll_direction
        # Repeat scroll at interval
        elif repeat_elapsed >= Config.SCROLL_REPEAT_INTERVAL:
            self._scroll_repeat_timer[scroll_direction] = now
            return dwell_progress, scroll_direction
                
        return dwell_progress, None

    def _reset_state(self) -> None:
        self._current_zone_horizontal    = "CENTER"
        self._zone_start_time_horizontal = time.monotonic()
        self._current_zone_vertical      = "CENTER"
        self._zone_start_time_vertical   = time.monotonic()