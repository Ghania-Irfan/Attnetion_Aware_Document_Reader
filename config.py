"""Centralized configuration for the attention-aware presentation controller."""

from __future__ import annotations


class Config:
    # Webcam
    CAMERA_INDEX = 0
    CAMERA_FRAME_WIDTH = 1280
    CAMERA_FRAME_HEIGHT = 720

    # Gaze / FaceMesh
    FACE_MESH_MAX_NUM_FACES = 1
    FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
    FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5
    FACE_MESH_REFINE_LANDMARKS = True
    IRIS_LANDMARK_INDEX = 468
    EYE_LEFT_CORNER_INDEX = 33
    EYE_RIGHT_CORNER_INDEX = 133
    EYE_WIDTH_EPSILON = 1e-6
    
    # Set this to TRUE if left/right is inverted
    INVERT_GAZE_HORIZONTAL = True
    
    # Zone thresholds (horizontal)
    LEFT_ZONE_THRESHOLD = 0.35   # Made more sensitive - look left easier
    RIGHT_ZONE_THRESHOLD = 0.65  # Made more sensitive - look right easier
    
    # Vertical gaze thresholds (for auto-scroll)
    VERTICAL_GAZE_ENABLED = True
    TOP_ZONE_THRESHOLD = 0.40     # Looking up threshold
    BOTTOM_ZONE_THRESHOLD = 0.60  # Looking down threshold
    
    # Auto-scroll configuration
    AUTO_SCROLL_ENABLED = True
    SCROLL_DWELL_SECONDS = 0.5     # 0.5 seconds hold to start scrolling
    SCROLL_COOLDOWN_SECONDS = 0.2   # Short cooldown between scrolls
    SCROLL_REPEAT_INTERVAL = 0.1    # Fast repeat for smooth scrolling
    
    # Auto-scroll for PowerPoint preview pane
    PPT_PREVIEW_PANE_SELECT_KEY = "f6"
    SCROLL_DIRECTION_UP = "up"
    SCROLL_DIRECTION_DOWN = "down"
    
    # Gaze smoothing
    GAZE_SMOOTH_SAMPLES = 3

    # Calibration
    AUTO_CALIBRATE_LEFT_ON_STARTUP = False
    AUTO_CALIBRATE_BOTH_ON_STARTUP = False
    LEFT_CALIBRATION_SECONDS = 3.0
    RIGHT_CALIBRATION_SECONDS = 3.0
    VERTICAL_CALIBRATION_SECONDS = 3.0
    CALIBRATION_MARGIN = 0.08
    MIN_CENTER_BAND_WIDTH = 0.15
    LEFT_TRIGGER_SENSITIVITY = 0.5
    RIGHT_TRIGGER_SENSITIVITY = 0.5
    CALIBRATION_WINDOW_NAME = "Gaze Calibration"
    CALIBRATION_DOT_Y_RATIO = 0.5
    CALIBRATION_DOT_LEFT_X_RATIO = 0.12
    CALIBRATION_DOT_CENTER_X_RATIO = 0.5
    CALIBRATION_DOT_RIGHT_X_RATIO = 0.88
    CALIBRATION_DOT_TOP_Y_RATIO = 0.15
    CALIBRATION_DOT_BOTTOM_Y_RATIO = 0.85
    CALIBRATION_DOT_RADIUS = 24
    CALIBRATION_DOT_COLOR_BGR = (0, 255, 255)
    CALIBRATION_BG_COLOR_BGR = (0, 0, 0)
    CALIBRATION_TEXT_COLOR_BGR = (255, 255, 255)
    CALIBRATION_TEXT_X = 60
    CALIBRATION_TEXT_Y = 80
    CALIBRATION_TEXT_LINE_SPACING = 36
    CALIBRATION_TEXT_FONT_SCALE = 0.9
    CALIBRATION_TEXT_THICKNESS = 2
    CALIBRATION_SETTLE_SECONDS = 1.0
    CALIBRATION_MIN_SAMPLES_PER_TARGET = 20
    CALIBRATION_WAITKEY_MS = 1

    # Dwell / cooldown timings
    DWELL_SECONDS = 1.5
    ACTION_COOLDOWN_SECONDS = 2.0
    GESTURE_CONFIRMATION_SECONDS = 1.2
    GESTURE_ZOOM_COOLDOWN_SECONDS = 1.8

    # Hand gestures / MediaPipe Hands
    HANDS_MAX_NUM_HANDS = 1
    HANDS_MIN_DETECTION_CONFIDENCE = 0.5
    HANDS_MIN_TRACKING_CONFIDENCE = 0.5
    FINGER_TIP_PIP_PAIRS = ((8, 6), (12, 10), (16, 14), (20, 18))
    THUMB_TIP_INDEX = 4
    THUMB_IP_INDEX = 2

    # Overlay behavior
    OVERLAY_WIDTH = 800
    OVERLAY_HEIGHT = 500
    OVERLAY_ZOOM_FACTOR = 2.0
    OVERLAY_PAN_LERP_FACTOR = 0.15
    OVERLAY_REFRESH_MS = 50
    OVERLAY_BG_COLOR = "black"
    OVERLAY_STOP_JOIN_TIMEOUT_SECONDS = 1.0

    # UI / debugging
    DEBUG_WINDOW_NAME = "Attention Controller Debug"
    DEBUG_TEXT_COLOR_BGR = (0, 255, 0)
    DEBUG_PROGRESS_BG_BGR = (60, 60, 60)
    DEBUG_PROGRESS_FG_BGR = (0, 200, 255)
    DEBUG_PROGRESS_WIDTH = 300
    DEBUG_PROGRESS_HEIGHT = 18
    DEBUG_PROGRESS_ORIGIN_X = 20
    DEBUG_PROGRESS_ORIGIN_Y = 150
    DEBUG_TEXT_ORIGIN_X = 20
    DEBUG_TEXT_ORIGIN_Y = 30
    DEBUG_TEXT_LINE_SPACING = 30
    DEBUG_PROGRESS_LABEL_OFFSET_Y = 8
    DEBUG_FONT = 0
    DEBUG_FONT_SCALE = 0.75
    DEBUG_FONT_THICKNESS = 2
    DEBUG_WAITKEY_MS = 1
    EXIT_KEY = "q"
    OPEN_PPT_KEY = "o"
    CALIBRATE_KEY = "c"
    CALIBRATE_VERTICAL_KEY = "v"
    MIRROR_DEBUG_VIEW = True

    # Stability / control
    ENABLE_PYAUTOGUI_FAILSAFE = False

    # PowerPoint launch / slideshow
    PPT_FILE_TYPES = (("PowerPoint files", "*.pptx *.ppt"), ("All files", "*.*"))
    PPT_OPEN_TO_SLIDESHOW_DELAY_SECONDS = 2.0
    PPT_WINDOW_TITLE_CANDIDATES = ("PowerPoint Slide Show", "PowerPoint")
    PPT_FOCUS_RETRY_DELAY_SECONDS = 0.1
    PPT_FOCUS_MAX_ATTEMPTS = 10
    PPT_ZOOM_IN_KEY_CANDIDATES = ("+", "plus", "add")
    PPT_ZOOM_OUT_KEY_CANDIDATES = ("-", "minus", "subtract")
    
    # Scroll keys for PowerPoint preview pane (use PAGE UP/DOWN for better scrolling)
    PPT_SCROLL_UP_KEY = "pageup"
    PPT_SCROLL_DOWN_KEY = "pagedown"


    # Gesture timing
    GESTURE_CONFIRMATION_SECONDS = 1.2  # Hold gesture for 1.2 seconds to trigger
    GESTURE_ZOOM_COOLDOWN_SECONDS = 1.5  # Wait 1.5 seconds before next zoom