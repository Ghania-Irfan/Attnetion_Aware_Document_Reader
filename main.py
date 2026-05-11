"""Entry point for the Attention-Aware Presentation Controller."""

from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import pyautogui

from action_controller import ActionController, ControllerState
from config import Config
from gaze_tracker import GazeTracker
from gesture_detector import GestureDetector
from attention_monitor import AttentionMonitor


class PowerPointManager:
    """Manages PowerPoint window detection and focusing, with preview pane support."""
    
    def __init__(self):
        self._current_ppt_window = None
        self._last_check_time = 0
        self._check_interval = 1.0
        self._preview_pane_focused = False
        
    def find_ppt_window(self):
        """Find PowerPoint window (works for both slideshow and edit mode)."""
        slideshow_windows = pyautogui.getWindowsWithTitle("PowerPoint Slide Show")
        if slideshow_windows:
            self._current_ppt_window = slideshow_windows[0]
            return self._current_ppt_window
            
        all_windows = pyautogui.getAllWindows()
        for window in all_windows:
            title = window.title
            if ".pptx" in title.lower() or ".ppt" in title.lower():
                if "powerpoint" in title.lower() or window.visible:
                    self._current_ppt_window = window
                    return window
                    
        ppt_windows = pyautogui.getWindowsWithTitle("PowerPoint")
        if ppt_windows:
            self._current_ppt_window = ppt_windows[0]
            return self._current_ppt_window
            
        try:
            import win32gui

            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    class_name  = win32gui.GetClassName(hwnd)
                    if "PPTFrameClass" in class_name or "screenClass" in class_name:
                        windows.append((hwnd, window_text))
            
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            if windows:
                for title in Config.PPT_WINDOW_TITLE_CANDIDATES:
                    for window in pyautogui.getWindowsWithTitle(title):
                        self._current_ppt_window = window
                        return window
        except ImportError:
            pass
            
        return None
    
    def ensure_focus(self, focus_preview_pane: bool = False) -> bool:
        """Ensure PowerPoint window has focus, optionally focus the preview pane."""
        now = time.time()
        if now - self._last_check_time < self._check_interval:
            if self._current_ppt_window and self._current_ppt_window.visible:
                try:
                    if self._current_ppt_window.isMinimized:
                        self._current_ppt_window.restore()
                    self._current_ppt_window.activate()
                    
                    if focus_preview_pane and not self.is_in_slideshow():
                        time.sleep(0.05)
                        pyautogui.press(Config.PPT_PREVIEW_PANE_SELECT_KEY)
                        self._preview_pane_focused = True
                    
                    return True
                except Exception:
                    pass
                    
        self._last_check_time = now
        ppt_window = self.find_ppt_window()
        
        if ppt_window is not None:
            try:
                if ppt_window.isMinimized:
                    ppt_window.restore()
                ppt_window.activate()
                
                if focus_preview_pane and not self.is_in_slideshow():
                    time.sleep(0.05)
                    pyautogui.press(Config.PPT_PREVIEW_PANE_SELECT_KEY)
                    self._preview_pane_focused = True
                    
                return True
            except Exception:
                pass
                
        return False
    
    def ensure_preview_pane_focus(self) -> bool:
        """Specifically ensure the preview pane has focus for scrolling."""
        if not self.ensure_focus(focus_preview_pane=True):
            return False
        if self.is_in_slideshow():
            print("[SCROLL] In slideshow mode - auto-scroll only works in edit mode", flush=True)
            return False
        return True
    
    def is_in_slideshow(self) -> bool:
        """Check if PowerPoint is in slideshow mode."""
        ppt_window = self.find_ppt_window()
        if ppt_window:
            return "Slide Show" in ppt_window.title
        return False
    
    def is_ppt_running(self) -> bool:
        """Check if any PowerPoint window is running."""
        return self.find_ppt_window() is not None


def _open_ppt_in_slideshow() -> bool:
    """Open PowerPoint file and start slideshow."""
    picker_root = tk.Tk()
    picker_root.withdraw()
    picker_root.attributes("-topmost", True)
    try:
        ppt_path = filedialog.askopenfilename(
            title="Select PowerPoint file",
            filetypes=Config.PPT_FILE_TYPES,
        )
    finally:
        picker_root.destroy()

    if not ppt_path:
        print("[PPT] Selection cancelled.", flush=True)
        return False

    try:
        os.startfile(ppt_path)
        print(f"[PPT] Opened: {ppt_path}", flush=True)
    except OSError as exc:
        print(f"[PPT] Failed to open file: {exc}", flush=True)
        return False

    time.sleep(Config.PPT_OPEN_TO_SLIDESHOW_DELAY_SECONDS)
    
    ppt_manager = PowerPointManager()
    ppt_manager.ensure_focus()
    pyautogui.press("f5")
    time.sleep(0.5)
    ppt_manager.ensure_focus()
    
    print(f"[PPT] Sent F5 for slideshow mode.", flush=True)
    return True


def _draw_debug_panel(
    frame,
    gaze_result,
    gesture_result,
    state_name: str,
    ppt_controls_enabled: bool,
    in_slideshow: bool,
    attention_status,
) -> None:
    """Draw debug information on the frame."""
    attention_indicator = "ATTENTION LOST" if not attention_status.is_attentive else "ATTENTION OK"
    if attention_status.showing_popup:
        attention_indicator = "POPUP SHOWING"

    lines = [
        f"Zone H: {gaze_result.zone_horizontal} | Zone V: {gaze_result.zone_vertical}",
        f"Combined: {gaze_result.combined_zone}",
        f"Gesture: {gesture_result.gesture} ({gesture_result.extended_fingers} fingers)",
        f"State: {state_name}",
        f"Iris H: {gaze_result.iris_ratio_horizontal:.2f} | V: {gaze_result.iris_ratio_vertical:.2f}",
        f"Nav Action: {gaze_result.navigation_action or 'None'}",
        f"Scroll Action: {gaze_result.scroll_action or 'None'}",
        f"PPT Controls: {'ON' if ppt_controls_enabled else 'OFF'} | Mode: {'Slideshow' if in_slideshow else 'Edit'}",
        f"Attention: {attention_indicator}",
        f"Seconds Away: {attention_status.seconds_away:.1f}s",
    ]
    
    # Add V-sign hint if zoomed
    if state_name == "ZOOMED":
        lines.append("✌️ V-sign: Move fingers to PAN zoomed view!")

    y = Config.DEBUG_TEXT_ORIGIN_Y
    for line in lines:
        if "LOST" in line or "POPUP" in line:
            color = (0, 0, 255)       # Red
        elif "OK" in line:
            color = (0, 255, 0)       # Green
        elif "V-sign" in line:
            color = (0, 255, 255)     # Yellow
        else:
            color = Config.DEBUG_TEXT_COLOR_BGR
            
        cv2.putText(
            frame,
            line,
            (Config.DEBUG_TEXT_ORIGIN_X, y),
            Config.DEBUG_FONT,
            Config.DEBUG_FONT_SCALE,
            color,
            Config.DEBUG_FONT_THICKNESS,
        )
        y += Config.DEBUG_TEXT_LINE_SPACING

    # Horizontal dwell bar
    bar_x = Config.DEBUG_PROGRESS_ORIGIN_X
    bar_y = Config.DEBUG_PROGRESS_ORIGIN_Y
    bar_w = Config.DEBUG_PROGRESS_WIDTH
    bar_h = Config.DEBUG_PROGRESS_HEIGHT

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), Config.DEBUG_PROGRESS_BG_BGR, -1)
    filled_w = int(bar_w * max(0.0, min(1.0, gaze_result.dwell_progress_horizontal)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), Config.DEBUG_PROGRESS_FG_BGR, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), Config.DEBUG_TEXT_COLOR_BGR, 1)
    cv2.putText(
        frame,
        "Dwell H (L/R for slides)",
        (bar_x, bar_y - Config.DEBUG_PROGRESS_LABEL_OFFSET_Y),
        Config.DEBUG_FONT,
        Config.DEBUG_FONT_SCALE,
        Config.DEBUG_TEXT_COLOR_BGR,
        Config.DEBUG_FONT_THICKNESS,
    )

    # Vertical dwell bar
    bar_y_v    = bar_y + bar_h + 20
    filled_w_v = int(bar_w * max(0.0, min(1.0, gaze_result.dwell_progress_vertical)))
    cv2.rectangle(frame, (bar_x, bar_y_v), (bar_x + bar_w, bar_y_v + bar_h), Config.DEBUG_PROGRESS_BG_BGR, -1)
    cv2.rectangle(frame, (bar_x, bar_y_v), (bar_x + filled_w_v, bar_y_v + bar_h), (100, 100, 255), -1)
    cv2.rectangle(frame, (bar_x, bar_y_v), (bar_x + bar_w, bar_y_v + bar_h), Config.DEBUG_TEXT_COLOR_BGR, 1)
    
    if gaze_result.combined_zone == "TOP_LEFT":
        scroll_label = "Looking TOP-LEFT -> Scroll UP"
        scroll_label_color = (0, 255, 255)
    elif gaze_result.combined_zone == "BOTTOM_LEFT":
        scroll_label = "Looking BOTTOM-LEFT -> Scroll DOWN"
        scroll_label_color = (0, 255, 255)
    else:
        scroll_label = "Dwell V (Look LEFT + UP/DOWN to scroll previews)"
        scroll_label_color = Config.DEBUG_TEXT_COLOR_BGR
        
    cv2.putText(
        frame,
        scroll_label,
        (bar_x, bar_y_v - Config.DEBUG_PROGRESS_LABEL_OFFSET_Y),
        Config.DEBUG_FONT,
        Config.DEBUG_FONT_SCALE,
        scroll_label_color,
        Config.DEBUG_FONT_THICKNESS,
    )


def main() -> None:
    pyautogui.FAILSAFE = Config.ENABLE_PYAUTOGUI_FAILSAFE

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or unavailable. Please connect a webcam and retry.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_FRAME_HEIGHT)

    gaze_tracker = GazeTracker()
    gesture_detector = GestureDetector()
    ppt_manager = PowerPointManager()
    attention_monitor = AttentionMonitor()
    
    # No zoom overlay - V-sign will control pan directly via mouse
    action_controller = ActionController(
        ensure_target_focus=ppt_manager.ensure_focus,
        zoom_overlay=None  # No popup overlay
    )
    
    ppt_controls_enabled = False
    last_slideshow_check = 0
    is_in_slideshow = False

    print("\n" + "=" * 70)
    print("=== Attention-Aware Presentation Controller ===")
    print("=" * 70)
    print("\nCONTROLS:")
    print("  'o' - Open PowerPoint file and start slideshow")
    print("  'c' - Calibrate horizontal gaze (Left/Center/Right)")
    print("  'v' - Calibrate vertical gaze (Up/Down) for auto-scroll")
    print("  'a' - Toggle Attention Monitor ON/OFF")
    print("  'q' - Quit")
    
    print("\nZOOM FEATURE (No Popup Window):")
    print("  OPEN PALM (hold 1.2s) -> Zoom In")
    print("  FIST      (hold 1.2s) -> Zoom Out")
    print("  ✌️ V-SIGN while zoomed -> Move fingers to PAN the zoomed view!")
    print("     (Moves mouse with middle button held for smooth panning)")
    
    print("\nAUTO-SCROLL FEATURE (Edit Mode):")
    print("  Look at TOP-LEFT corner  -> Scroll UP through slide previews")
    print("  Look at BOTTOM-LEFT corner -> Scroll DOWN through slide previews")
    print("  Hold gaze for 0.8 seconds to start scrolling")
    
    print("\nNAVIGATION FEATURE (Slideshow Mode):")
    print("  Look LEFT for 1.5s  -> Previous slide")
    print("  Look RIGHT for 1.5s -> Next slide")
    
    print("\nATTENTION MONITOR:")
    print("  If you look away from camera for 2+ seconds,")
    print("  a popup will appear asking you to look at camera")
    print("=" * 70 + "\n")

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                raise RuntimeError("Failed to read from webcam stream.")

            gaze_result = gaze_tracker.process(raw_frame)
            gesture_result = gesture_detector.process(raw_frame)
            
            attention_status = attention_monitor.update_face_detection(gaze_result.face_detected)
            
            ppt_running = ppt_manager.is_ppt_running()
            
            now = time.time()
            if now - last_slideshow_check > 0.5:
                last_slideshow_check = now
                ppt_window = ppt_manager.find_ppt_window()
                if ppt_window:
                    is_in_slideshow = "Slide Show" in ppt_window.title
                    ppt_controls_enabled = ppt_running
                else:
                    is_in_slideshow = False
                    ppt_controls_enabled = False
            
            state = action_controller.update(
                gaze_result,
                gesture_result,
                controls_enabled=ppt_controls_enabled,
            )

            debug_frame = cv2.flip(raw_frame, 1) if Config.MIRROR_DEBUG_VIEW else raw_frame.copy()
            _draw_debug_panel(
                debug_frame, gaze_result, gesture_result, state.name,
                ppt_controls_enabled, is_in_slideshow, attention_status,
            )
            cv2.imshow(Config.DEBUG_WINDOW_NAME, debug_frame)

            key = cv2.waitKey(Config.DEBUG_WAITKEY_MS) & 0xFF
            
            if key == ord(Config.EXIT_KEY):
                print("\nShutting down... Goodbye!")
                break
                
            if key == ord(Config.OPEN_PPT_KEY):
                success = _open_ppt_in_slideshow()
                if success:
                    ppt_controls_enabled = True
                last_slideshow_check = 0
                
            if key == ord(Config.CALIBRATE_KEY):
                print("\n[CALIBRATE] Horizontal calibration started via 'c' key.", flush=True)
                gaze_tracker.calibrate_both(
                    cap,
                    left_seconds=Config.LEFT_CALIBRATION_SECONDS,
                    right_seconds=Config.RIGHT_CALIBRATION_SECONDS,
                )
                print("[CALIBRATE] Horizontal calibration complete!\n")
                
            if key == ord(Config.CALIBRATE_VERTICAL_KEY):
                print("\n[CALIBRATE] Vertical calibration started via 'v' key.", flush=True)
                gaze_tracker.calibrate_vertical(
                    cap,
                    seconds=Config.VERTICAL_CALIBRATION_SECONDS,
                )
                print("[CALIBRATE] Vertical calibration complete!\n")
                
            if key == ord('a'):
                current_state = attention_monitor.is_enabled()
                attention_monitor.set_enabled(not current_state)
                print(f"[ATTENTION] Monitor {'ENABLED' if not current_state else 'DISABLED'}", flush=True)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nCleaning up resources...")
        # Ensure V-sign pan is deactivated
        action_controller._deactivate_v_sign()
        cap.release()
        cv2.destroyAllWindows()
        gesture_detector.close()
        gaze_tracker.close()
        attention_monitor.close()
        print("Cleanup complete!")


if __name__ == "__main__":
    main()