"""Hand gesture detection using MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp

from config import Config

try:
    HANDS_MODULE = mp.solutions.hands
except Exception:
    try:
        from mediapipe.python.solutions import hands as HANDS_MODULE
    except Exception as exc:
        raise ImportError(
            "This project requires MediaPipe Solutions API (FaceMesh/Hands). "
            "Your current mediapipe package does not provide it. "
            "Use the project venv and reinstall dependencies from requirements.txt."
        ) from exc


@dataclass
class GestureResult:
    gesture: str
    extended_fingers: int
    hand_detected: bool
    hand_position: tuple[float, float] | None = None
    finger_positions: dict = None
    debug_info: dict = None  # For debugging


class GestureDetector:
    """Classifies OPEN_PALM, FIST, V_SIGN (index+middle), and tracks hand position."""

    # Finger tip landmark indices
    FINGER_TIPS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    
    # Finger PIP landmarks for extension check
    FINGER_PIPS = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }
    
    # Finger MCP landmarks for better detection
    FINGER_MCP = {
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17,
    }

    def __init__(self) -> None:
        self._hands = HANDS_MODULE.Hands(
            max_num_hands=Config.HANDS_MAX_NUM_HANDS,
            min_detection_confidence=Config.HANDS_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.HANDS_MIN_TRACKING_CONFIDENCE,
        )
        self._frame_count = 0

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr) -> GestureResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return GestureResult(gesture="NONE", extended_fingers=0, hand_detected=False)

        landmarks = result.multi_hand_landmarks[0].landmark
        hand_label = result.multi_handedness[0].classification[0].label
        
        # Get normalized hand position (wrist)
        wrist = landmarks[0]
        hand_x = wrist.x
        hand_y = wrist.y
        
        # Get finger positions for tracking
        finger_positions = {}
        for finger_name, tip_idx in self.FINGER_TIPS.items():
            finger_positions[finger_name] = (landmarks[tip_idx].x, landmarks[tip_idx].y)

        # Extended fingers count using standard method
        extended_count = self._count_extended_fingers(landmarks, hand_label)
        
        # Detect V-sign using multiple methods for reliability
        is_v_sign = self._detect_v_sign_improved(landmarks, hand_label)
        
        # Debug output every 30 frames
        self._frame_count += 1
        if self._frame_count % 30 == 0 and is_v_sign:
            print(f"[GESTURE] V-SIGN detected! Extended count: {extended_count}", flush=True)
        
        # Determine gesture
        if is_v_sign:
            gesture = "V_SIGN"
        elif extended_count >= 4:
            gesture = "OPEN_PALM"
        elif extended_count <= 1:
            gesture = "FIST"
        else:
            gesture = "NONE"

        # Create debug info
        debug_info = {
            "extended_count": extended_count,
            "is_v_sign": is_v_sign,
            "hand_label": hand_label,
        }

        return GestureResult(
            gesture=gesture,
            extended_fingers=extended_count,
            hand_detected=True,
            hand_position=(hand_x, hand_y),
            finger_positions=finger_positions,
            debug_info=debug_info,
        )

    def _detect_v_sign_improved(self, landmarks, hand_label: str) -> bool:
        """
        Improved V-sign detection with multiple checks.
        Returns True if index and middle fingers are extended, others are curled.
        """
        # Get y-coordinates for each finger (higher y = lower on screen)
        # For V-sign: index and middle should be extended (tip_y < pip_y)
        # Ring and pinky should be curled (tip_y > pip_y)
        
        # Check index finger
        index_tip_y = landmarks[self.FINGER_TIPS["index"]].y
        index_pip_y = landmarks[self.FINGER_PIPS["index"]].y
        index_extended = index_tip_y < index_pip_y - 0.02  # Threshold for clearer detection
        
        # Check middle finger
        middle_tip_y = landmarks[self.FINGER_TIPS["middle"]].y
        middle_pip_y = landmarks[self.FINGER_PIPS["middle"]].y
        middle_extended = middle_tip_y < middle_pip_y - 0.02
        
        # Check ring finger (should be curled)
        ring_tip_y = landmarks[self.FINGER_TIPS["ring"]].y
        ring_pip_y = landmarks[self.FINGER_PIPS["ring"]].y
        ring_curled = ring_tip_y > ring_pip_y + 0.01
        
        # Check pinky (should be curled)
        pinky_tip_y = landmarks[self.FINGER_TIPS["pinky"]].y
        pinky_pip_y = landmarks[self.FINGER_PIPS["pinky"]].y
        pinky_curled = pinky_tip_y > pinky_pip_y + 0.01
        
        # For V-sign, index and middle should be extended, others curled
        result = index_extended and middle_extended and ring_curled and pinky_curled
        
        # Debug for V-sign detection
        if self._frame_count % 60 == 0:
            print(f"[V-SIGN DEBUG] Index: {index_extended} (tip:{index_tip_y:.3f} < pip:{index_pip_y:.3f})", flush=True)
            print(f"              Middle: {middle_extended} (tip:{middle_tip_y:.3f} < pip:{middle_pip_y:.3f})", flush=True)
            print(f"              Ring curled: {ring_curled} (tip:{ring_tip_y:.3f} > pip:{ring_pip_y:.3f})", flush=True)
            print(f"              Pinky curled: {pinky_curled} (tip:{pinky_tip_y:.3f} > pip:{pinky_pip_y:.3f})", flush=True)
            print(f"              V-SIGN result: {result}", flush=True)
        
        return result

    def _detect_v_sign(self, landmarks, hand_label: str) -> bool:
        """
        Original V-sign detection (kept for compatibility).
        """
        # Check index finger extended
        index_tip_y = landmarks[self.FINGER_TIPS["index"]].y
        index_pip_y = landmarks[self.FINGER_PIPS["index"]].y
        index_extended = index_tip_y < index_pip_y
        
        # Check middle finger extended
        middle_tip_y = landmarks[self.FINGER_TIPS["middle"]].y
        middle_pip_y = landmarks[self.FINGER_PIPS["middle"]].y
        middle_extended = middle_tip_y < middle_pip_y
        
        # Check ring finger NOT extended (curled)
        ring_tip_y = landmarks[self.FINGER_TIPS["ring"]].y
        ring_pip_y = landmarks[self.FINGER_PIPS["ring"]].y
        ring_curled = ring_tip_y > ring_pip_y
        
        # Check pinky NOT extended (curled)
        pinky_tip_y = landmarks[self.FINGER_TIPS["pinky"]].y
        pinky_pip_y = landmarks[self.FINGER_PIPS["pinky"]].y
        pinky_curled = pinky_tip_y > pinky_pip_y
        
        return index_extended and middle_extended and ring_curled and pinky_curled

    def _count_extended_fingers(self, landmarks, hand_label: str) -> int:
        """Count extended fingers using tip vs pip comparison."""
        count = 0
        
        # Check index, middle, ring, pinky
        for finger in ["index", "middle", "ring", "pinky"]:
            tip_y = landmarks[self.FINGER_TIPS[finger]].y
            pip_y = landmarks[self.FINGER_PIPS[finger]].y
            if tip_y < pip_y:
                count += 1

        # Thumb extension (horizontal check)
        thumb_tip_x = landmarks[self.FINGER_TIPS["thumb"]].x
        thumb_ip_x = landmarks[self.FINGER_PIPS["thumb"]].x
        if hand_label == "Right":
            thumb_extended = thumb_tip_x < thumb_ip_x
        else:
            thumb_extended = thumb_tip_x > thumb_ip_x
        if thumb_extended:
            count += 1

        return count