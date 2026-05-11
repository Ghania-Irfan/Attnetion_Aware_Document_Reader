"""State machine and action arbitration for gaze and hand gesture signals."""

from __future__ import annotations

import time
from enum import Enum, auto

import pyautogui
from config import Config


class ControllerState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    SCROLLING = auto()
    ZOOMED = auto()


class ActionController:
    """Prioritizes gestures over gaze navigation and controls app state."""

    def __init__(self, ensure_target_focus=None, zoom_overlay=None) -> None:
        self._ensure_target_focus = ensure_target_focus
        self._zoom_overlay = zoom_overlay
        self.state = ControllerState.IDLE
        self._last_zoom_fire_time = {"in": 0.0, "out": 0.0}
        self._last_gesture = "NONE"
        self._gesture_start_time = 0.0
        self._is_zoomed = False  # Track zoom state separately
        
        # V-sign tracking for pan control
        self._v_sign_active = False
        self._last_finger_position = (0.5, 0.5)
        self._last_mouse_position = None
        self._pan_sensitivity = 12.0
        
        # Screen dimensions
        self._screen_width, self._screen_height = pyautogui.size()
        
        # Track if panning is currently happening
        self._is_panning = False
        
        # Debug counter
        self._debug_counter = 0

    def update(self, gaze_result, gesture_result, controls_enabled: bool = True) -> ControllerState:
        if not controls_enabled:
            self.state = ControllerState.IDLE
            self._is_zoomed = False
            self._last_gesture = "NONE"
            self._gesture_start_time = 0.0
            self._deactivate_v_sign()
            return self.state

        # Debug: Print gesture every 60 frames
        self._debug_counter += 1
        if self._debug_counter % 60 == 0:
            print(f"[ACTION] Gesture: {gesture_result.gesture}, State: {self.state.name}, Zoomed: {self._is_zoomed}, V_active: {self._v_sign_active}", flush=True)

        # ── 1. HAND GESTURES (Highest Priority) ──────────────────────────────
        
        # Check for OPEN_PALM to ZOOM IN
        if gesture_result.gesture == "OPEN_PALM":
            self._handle_zoom_gesture("OPEN_PALM", "in")
            return self.state
        
        # Check for FIST to ZOOM OUT
        elif gesture_result.gesture == "FIST":
            self._handle_zoom_gesture("FIST", "out")
            return self.state
        
        # Check for V-SIGN to PAN (only when zoomed in)
        elif gesture_result.gesture == "V_SIGN":
            if self._is_zoomed:
                self._handle_v_sign_pan(gesture_result)
            else:
                if self._debug_counter % 30 == 0:
                    print(f"[V-SIGN] V-sign detected but NOT ZOOMED! First make OPEN_PALM gesture to zoom in.", flush=True)
            return self.state
        
        else:
            # Reset gesture tracking for non-gesture frames
            if self._last_gesture not in ["OPEN_PALM", "FIST"]:
                self._last_gesture = "NONE"
                self._gesture_start_time = 0.0
            # Deactivate V-sign if gesture is lost
            if self._v_sign_active:
                self._deactivate_v_sign()

        # ── 2. SCROLL ACTIONS ──────────────────────────────────────────────
        if gaze_result.scroll_action:
            try:
                if self._ensure_target_focus is not None:
                    try:
                        if hasattr(self._ensure_target_focus, '__self__'):
                            if hasattr(self._ensure_target_focus.__self__, 'ensure_preview_pane_focus'):
                                self._ensure_target_focus.__self__.ensure_preview_pane_focus()
                            else:
                                self._ensure_target_focus()
                        else:
                            self._ensure_target_focus()
                    except Exception:
                        self._ensure_target_focus()
                
                if gaze_result.scroll_action == "up":
                    pyautogui.press(Config.PPT_SCROLL_UP_KEY)
                    print(f"[SCROLL] ⬆️ SCROLLING UP", flush=True)
                elif gaze_result.scroll_action == "down":
                    pyautogui.press(Config.PPT_SCROLL_DOWN_KEY)
                    print(f"[SCROLL] ⬇️ SCROLLING DOWN", flush=True)
                    
            except Exception as e:
                print(f"[SCROLL] Error: {e}", flush=True)
            self.state = ControllerState.SCROLLING
            return self.state

        # ── 3. NAVIGATION ACTIONS ──────────────────────────────────────────
        if gaze_result.navigation_action:
            try:
                if self._ensure_target_focus is not None:
                    self._ensure_target_focus()
                pyautogui.press(gaze_result.navigation_action)
                print(f"[NAVIGATION] Triggered {gaze_result.navigation_action}", flush=True)
            except Exception:
                pass
            self.state = ControllerState.NAVIGATING
        else:
            # Update state based on zoom and gaze
            if self._is_zoomed:
                self.state = ControllerState.ZOOMED
            elif gaze_result.zone_horizontal == "CENTER" or not gaze_result.face_detected:
                self.state = ControllerState.IDLE
            else:
                self.state = ControllerState.NAVIGATING

        return self.state

    def _handle_zoom_gesture(self, gesture_name: str, direction: str) -> None:
        """Handle zoom in/out gestures with hold timing."""
        now = time.monotonic()
        
        # If this is a new gesture, start tracking
        if self._last_gesture != gesture_name:
            self._last_gesture = gesture_name
            self._gesture_start_time = now
            print(f"[GESTURE] Started holding {gesture_name} - hold for {Config.GESTURE_CONFIRMATION_SECONDS}s to zoom {direction}", flush=True)
            return

        # Check if we've held long enough
        held_for = now - self._gesture_start_time
        if held_for >= Config.GESTURE_CONFIRMATION_SECONDS:
            # Check cooldown
            now_time = time.monotonic()
            if now_time - self._last_zoom_fire_time[direction] < Config.GESTURE_ZOOM_COOLDOWN_SECONDS:
                print(f"[ZOOM] Cooldown active for {direction}", flush=True)
                return
            
            # Trigger zoom
            self._last_zoom_fire_time[direction] = now_time
            print(f"[GESTURE] ✓ Triggering zoom {direction} after {held_for:.1f}s!", flush=True)
            
            if self._ensure_target_focus is not None:
                self._ensure_target_focus()
            
            # Send zoom key
            key_candidates = (
                Config.PPT_ZOOM_IN_KEY_CANDIDATES
                if direction == "in"
                else Config.PPT_ZOOM_OUT_KEY_CANDIDATES
            )
            for key in key_candidates:
                try:
                    pyautogui.press(key)
                    print(f"[ZOOM] {'🔍 Zoomed IN' if direction == 'in' else '🔍 Zoomed OUT'} (pressed '{key}')", flush=True)
                    
                    # Update zoom state
                    if direction == "in":
                        self._is_zoomed = True
                        self.state = ControllerState.ZOOMED
                        print(f"[ZOOM] ✓ Now in ZOOMED mode - use V-sign to pan!", flush=True)
                    else:
                        self._is_zoomed = False
                        self.state = ControllerState.IDLE
                        print(f"[ZOOM] ✓ Exited zoom mode", flush=True)
                    
                    break
                except Exception as e:
                    print(f"[ZOOM] Failed to press {key}: {e}", flush=True)
                    continue
            
            # Reset gesture tracking after triggering
            self._last_gesture = "NONE"
            self._gesture_start_time = 0.0

    def _handle_v_sign_pan(self, gesture_result: any) -> None:
        """Handle V-sign gesture for pan control in PowerPoint."""
        
        # Get finger position (use index finger tip as tracking point)
        finger_pos = None
        if hasattr(gesture_result, 'finger_positions') and gesture_result.finger_positions:
            finger_pos = gesture_result.finger_positions.get("index")
        
        if not finger_pos:
            return
            
        finger_x, finger_y = finger_pos
        
        # Activate pan mode if not already active
        if not self._v_sign_active:
            print(f"[V-SIGN] ✌️ V-SIGN ACTIVATED! Starting pan mode...", flush=True)
            self._v_sign_active = True
            self._last_finger_position = (finger_x, finger_y)
            
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            self._last_mouse_position = (current_x, current_y)
            
            # For PowerPoint panning: Hold Ctrl + Left mouse button
            try:
                pyautogui.keyDown('ctrl')
                pyautogui.mouseDown(button='left')
                self._is_panning = True
                print(f"[V-SIGN] ✓ Holding Ctrl+LeftClick - Move your hand to PAN the zoomed view!", flush=True)
            except Exception as e:
                print(f"[V-SIGN] Failed to start pan: {e}", flush=True)
                self._is_panning = False
            
            return
        
        # Calculate finger movement delta
        delta_x = finger_x - self._last_finger_position[0]
        delta_y = finger_y - self._last_finger_position[1]
        
        # Apply deadzone to avoid jitter
        if abs(delta_x) < 0.002 and abs(delta_y) < 0.002:
            return
        
        # Calculate mouse movement based on finger movement
        move_x = delta_x * self._pan_sensitivity * 200
        move_y = delta_y * self._pan_sensitivity * 200
        
        # Get current mouse position
        if self._last_mouse_position:
            new_x = self._last_mouse_position[0] + move_x
            new_y = self._last_mouse_position[1] + move_y
        else:
            current_x, current_y = pyautogui.position()
            new_x = current_x + move_x
            new_y = current_y + move_y
        
        # Clamp to screen bounds
        new_x = max(0, min(self._screen_width, new_x))
        new_y = max(0, min(self._screen_height, new_y))
        
        # Move the mouse while holding the button (this pans PowerPoint)
        if self._is_panning:
            pyautogui.moveTo(new_x, new_y, duration=0.01)
            
            # Update last mouse position
            self._last_mouse_position = (new_x, new_y)
            
            # Debug output occasionally
            if self._debug_counter % 30 == 0:
                print(f"[V-SIGN] Panning: hand move ({delta_x:.3f}, {delta_y:.3f}) -> mouse move ({move_x:.1f}, {move_y:.1f})", flush=True)
        
        # Update last finger position
        self._last_finger_position = (finger_x, finger_y)

    def _deactivate_v_sign(self) -> None:
        """Deactivate V-sign pan mode and release pan buttons."""
        if self._v_sign_active:
            print(f"[V-SIGN] ✌️ V-SIGN DEACTIVATED - Stopping pan", flush=True)
            
            if self._is_panning:
                # Release Ctrl key and left mouse button
                try:
                    pyautogui.mouseUp(button='left')
                    pyautogui.keyUp('ctrl')
                    print(f"[V-SIGN] ✓ Released Ctrl+LeftClick", flush=True)
                except Exception as e:
                    print(f"[V-SIGN] Error releasing: {e}", flush=True)
                
                self._is_panning = False
            
            self._v_sign_active = False
            self._last_mouse_position = None
            self._last_finger_position = (0.5, 0.5)
    
    def hide_zoom_overlay(self) -> None:
        """Hide the zoom overlay if it exists."""
        if self._zoom_overlay:
            self._zoom_overlay.hide()