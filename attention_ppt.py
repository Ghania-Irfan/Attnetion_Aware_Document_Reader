import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from datetime import datetime

class BlinkDetector:
    """Detects blinks using Eye Aspect Ratio (EAR)"""
    
    def __init__(self, ear_threshold=0.2):
        self.EAR_THRESHOLD = ear_threshold
        self.blink_in_progress = False
        self.blink_start_time = None
        self.last_blink_end = 0
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR)"""
        # Compute vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, left_ear, right_ear):
        """Detect if a blink is occurring"""
        avg_ear = (left_ear + right_ear) / 2.0
        current_time = time.time()
        
        if avg_ear < self.EAR_THRESHOLD:
            if not self.blink_in_progress:
                self.blink_in_progress = True
                self.blink_start_time = current_time
            return False, 0
        else:
            if self.blink_in_progress:
                duration = current_time - self.blink_start_time
                self.blink_in_progress = False
                self.last_blink_end = current_time
                return True, duration
            return False, 0

class GazeDetector:
    """Detects gaze direction using eye landmark positions"""
    
    def __init__(self):
        # MediaPipe face mesh indices for eyes
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
    def get_gaze_direction(self, landmarks, frame_width):
        """Determine if user is looking left, right, or center"""
        try:
            # Get left eye center position
            left_eye_points = []
            for idx in self.LEFT_EYE_INDICES:
                pt = landmarks.landmark[idx]
                left_eye_points.append((pt.x, pt.y))
            
            # Get right eye center position
            right_eye_points = []
            for idx in self.RIGHT_EYE_INDICES:
                pt = landmarks.landmark[idx]
                right_eye_points.append((pt.x, pt.y))
            
            # Calculate eye centers
            left_eye_center_x = np.mean([p[0] for p in left_eye_points])
            right_eye_center_x = np.mean([p[0] for p in right_eye_points])
            
            # Calculate nose position (reference point)
            nose_x = landmarks.landmark[1].x  # Tip of nose
            
            # Simple gaze detection based on eye position relative to nose
            eye_center_avg = (left_eye_center_x + right_eye_center_x) / 2
            
            # Normalize to screen width
            gaze_ratio = (eye_center_avg - nose_x) * 10  # Scale factor
            
            # Determine direction
            if gaze_ratio < -0.1:
                return "LEFT"
            elif gaze_ratio > 0.1:
                return "RIGHT"
            else:
                return "CENTER"
                
        except Exception as e:
            print(f"Gaze detection error: {e}")
            return "UNKNOWN"

class PopupManager:
    """Manages popup notifications"""
    
    @staticmethod
    def show_popup(title, message, popup_type="warning"):
        """Show a popup notification"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            # Create hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)  # Make it appear on top
            
            if popup_type == "warning":
                messagebox.showwarning(title, message)
            elif popup_type == "info":
                messagebox.showinfo(title, message)
            elif popup_type == "error":
                messagebox.showerror(title, message)
            
            root.destroy()
            return True
        except Exception as e:
            print(f"[POPUP] Could not show popup: {e}")
            print(f"[POPUP] Message: {title} - {message}")
            return False

class AttentionMonitor:
    """Monitors user attention with visual dimming and popups"""
    
    def __init__(self, away_threshold=5):
        self.last_attention_time = time.time()
        self.away_threshold = away_threshold
        self.screen_dimmed = False
        self.warning_shown = False
        
        print(f"[ATTENTION] Monitoring started - Screen will dim after {away_threshold}s")
    
    def update_attention(self, face_detected):
        """Update attention status"""
        current_time = time.time()
        away_time = current_time - self.last_attention_time
        
        if face_detected:
            # User is looking at screen
            self.last_attention_time = current_time
            self.warning_shown = False
            
            # Restore screen if it was dimmed
            if self.screen_dimmed:
                print(f"[ATTENTION] ✅ User looked back after {away_time:.1f}s")
                self.restore_screen()
        else:
            # User is not looking at screen
            if not self.screen_dimmed:
                # Show warning before dimming
                if away_time > self.away_threshold * 0.7 and not self.warning_shown:
                    warning_msg = f"Warning: You haven't looked at the screen for {int(away_time)} seconds.\n"
                    warning_msg += f"Screen will dim in {self.away_threshold - int(away_time)} seconds."
                    print(f"[ATTENTION] ⚠️  {warning_msg}")
                    
                    # Show popup warning
                    PopupManager.show_popup(
                        "Attention Warning",
                        f"You're not looking at the screen!\n\n"
                        f"Screen will dim in {self.away_threshold - int(away_time)} seconds.\n"
                        f"Please look back at the screen.",
                        "warning"
                    )
                    self.warning_shown = True
                
                # Dim screen after threshold
                if away_time > self.away_threshold:
                    print(f"[ATTENTION] 🔴 User not looking for {away_time:.1f}s - Dimming screen")
                    self.dim_screen()
    
    def dim_screen(self):
        """Dim the screen visually and show popup"""
        if not self.screen_dimmed:
            self.screen_dimmed = True
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🔴 SCREEN DIMMED - Attention lost")
            
            # Show popup notification
            PopupManager.show_popup(
                "Screen Dimmed",
                "You're not looking at the screen!\n\n"
                "Screen has been visually dimmed.\n"
                "Look back at the screen to restore brightness.",
                "warning"
            )
    
    def restore_screen(self):
        """Restore screen brightness"""
        if self.screen_dimmed:
            self.screen_dimmed = False
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ SCREEN RESTORED - Attention regained")
            
            # Show welcome back popup
            PopupManager.show_popup(
                "Welcome Back",
                "Great! You're looking at the screen again.\n\n"
                "Screen brightness has been restored.",
                "info"
            )
    
    def get_away_time(self):
        """Get how long user has been away"""
        return time.time() - self.last_attention_time

class PPTController:
    """Controls PowerPoint presentation"""
    
    def __init__(self):
        print("[PPT] Controller initialized. Make sure PowerPoint is in presentation mode!")
        
    def next_slide(self):
        """Go to next slide"""
        try:
            pyautogui.press('right')
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 📄 NEXT SLIDE")
            return True
        except Exception as e:
            print(f"[PPT] Error: {e}")
            return False
        
    def prev_slide(self):
        """Go to previous slide"""
        try:
            pyautogui.press('left')
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 📄 PREVIOUS SLIDE")
            return True
        except Exception as e:
            print(f"[PPT] Error: {e}")
            return False

class AttentionPPT:
    """Main application class"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize components
        self.blink_detector = BlinkDetector(ear_threshold=0.25)
        self.gaze_detector = GazeDetector()
        self.attention_monitor = AttentionMonitor(away_threshold=5)
        self.ppt_controller = PPTController()
        
        # State variables
        self.last_navigation_time = 0
        self.navigation_cooldown = 2
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.window_name = "Attention-Aware PPT Reader"
        
    def draw_partitions(self, frame):
        """Draw left/right partitions on the frame"""
        height, width = frame.shape[:2]
        
        # Define partition boundaries (35% each side)
        left_boundary = int(width * 0.35)
        right_boundary = int(width * 0.65)
        
        # Draw left partition (GREEN)
        cv2.rectangle(frame, (0, 0), (left_boundary, height), (0, 100, 0), 3)
        cv2.putText(frame, "LOOK HERE", 
                   (20, 60), self.font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Then BLINK 1s", 
                   (20, 100), self.font, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "PREVIOUS SLIDE", 
                   (20, 140), self.font, 0.7, (0, 255, 0), 2)
        
        # Draw right partition (RED)
        cv2.rectangle(frame, (right_boundary, 0), (width, height), (0, 0, 100), 3)
        cv2.putText(frame, "LOOK HERE", 
                   (right_boundary + 20, 60), self.font, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Then BLINK 1s", 
                   (right_boundary + 20, 100), self.font, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, "NEXT SLIDE", 
                   (right_boundary + 20, 140), self.font, 0.7, (0, 0, 255), 2)
        
        # Draw center guidance
        cv2.line(frame, (left_boundary, 0), (left_boundary, height), (255, 255, 255), 1)
        cv2.line(frame, (right_boundary, 0), (right_boundary, height), (255, 255, 255), 1)
        
        # Center indicator
        cv2.putText(frame, "CENTER", 
                   (width//2 - 50, 50), self.font, 0.7, (255, 255, 255), 2)
        
        return left_boundary, right_boundary
    
    def draw_status_panel(self, frame, gaze_direction, blink_detected, blink_duration, away_time):
        """Draw status information on the frame"""
        height, width = frame.shape[:2]
        
        # Create status panel
        panel_height = 140
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Status indicators
        gaze_color = (0, 255, 0) if gaze_direction != "UNKNOWN" else (0, 0, 255)
        blink_color = (0, 255, 0) if blink_detected else (255, 255, 255)
        screen_color = (0, 255, 0) if not self.attention_monitor.screen_dimmed else (0, 0, 255)
        
        # Status text
        status_text = f"GAZE: {gaze_direction} | "
        status_text += f"BLINK: {blink_duration:.1f}s | "
        status_text += f"AWAY: {away_time:.1f}s"
        
        cv2.putText(panel, status_text, (10, 30), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Screen status
        screen_status = f"SCREEN: {'NORMAL' if not self.attention_monitor.screen_dimmed else 'DIMMED (VISUAL)'}"
        cv2.putText(panel, screen_status, (10, 60), 
                   self.font, 0.6, screen_color, 1)
        
        # Instructions
        instructions = "INSTRUCTIONS: Look left/right + Hold blink (1 second) to navigate slides"
        cv2.putText(panel, instructions, (10, 90), 
                   self.font, 0.5, (200, 200, 0), 1)
        
        # Attention warning
        if away_time > 3 and not self.attention_monitor.screen_dimmed:
            warning = f"⚠️  Look at screen! Will dim in {max(0, 5 - int(away_time))}s"
            warning_color = (0, 165, 255) if away_time < 5 else (0, 0, 255)
            cv2.putText(panel, warning, (10, 120), 
                       self.font, 0.5, warning_color, 1)
        
        # Add visual indicators
        cv2.circle(panel, (width - 100, 25), 10, gaze_color, -1)
        cv2.putText(panel, "Gaze", (width - 80, 30), self.font, 0.5, gaze_color, 1)
        
        cv2.circle(panel, (width - 180, 25), 10, blink_color, -1)
        cv2.putText(panel, "Blink", (width - 160, 30), self.font, 0.5, blink_color, 1)
        
        cv2.circle(panel, (width - 260, 25), 10, screen_color, -1)
        cv2.putText(panel, "Screen", (width - 240, 30), self.font, 0.5, screen_color, 1)
        
        # Quit instruction
        cv2.putText(panel, "Press 'Q' to quit", (width - 150, 120), 
                   self.font, 0.5, (0, 200, 200), 1)
        
        # Combine panel with frame
        frame[:panel_height, :] = panel
        
        return frame
    
    def handle_navigation(self, gaze_direction, blink_detected, blink_duration):
        """Handle slide navigation based on gaze and blink"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_navigation_time < self.navigation_cooldown:
            return False
        
        # Check for long blink (0.8-1.5 seconds)
        if blink_detected and 0.8 <= blink_duration <= 1.5:
            if gaze_direction == "RIGHT":
                success = self.ppt_controller.next_slide()
                if success:
                    self.last_navigation_time = current_time
                    return True
                    
            elif gaze_direction == "LEFT":
                success = self.ppt_controller.prev_slide()
                if success:
                    self.last_navigation_time = current_time
                    return True
        
        return False
    
    def run(self):
        """Main application loop"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ ERROR: Could not open webcam!")
            print("Please check if webcam is connected and not in use by another application.")
            return
        
        print("\n" + "="*70)
        print("🎯 ATTENTION-AWARE PPT READER - FYP PROTOTYPE")
        print("="*70)
        print("📹 Starting webcam...")
        print("💡 TIPS:")
        print("   - Sit 50-100 cm from camera")
        print("   - Ensure good lighting on your face")
        print("   - Keep your face visible in the camera")
        print("   - Look at LEFT/RIGHT boxes and BLINK for 1 second to navigate")
        print("   - Look away for 5 seconds to test screen dimming & popups")
        print("="*70)
        print("\n⚠️  IMPORTANT:")
        print("   1. Open PowerPoint and press F5 to start slideshow")
        print("   2. Screen will dim VISUALLY when you look away")
        print("   3. Popup notifications will appear")
        print("="*70 + "\n")
        
        # Check for tkinter
        try:
            import tkinter
            print("✅ Popup notifications enabled")
        except:
            print("⚠️  Tkinter not available - popups will show in console only")
            print("   For popups, install tkinter:")
            print("   Windows: Already installed")
            print("   Mac/Linux: May need to install separately")
        
        # Wait for user to be ready
        print("Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("✅ READY! Starting detection...\n")
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Draw partitions
            left_boundary, right_boundary = self.draw_partitions(frame)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            face_detected = False
            gaze_direction = "UNKNOWN"
            blink_detected = False
            blink_duration = 0
            left_ear = 0
            right_ear = 0
            
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0]
                
                # Get eye landmarks
                left_eye_points = np.array([
                    [landmarks.landmark[i].x * width, landmarks.landmark[i].y * height]
                    for i in self.gaze_detector.LEFT_EYE_INDICES
                ])
                
                right_eye_points = np.array([
                    [landmarks.landmark[i].x * width, landmarks.landmark[i].y * height]
                    for i in self.gaze_detector.RIGHT_EYE_INDICES
                ])
                
                # Draw eye contours
                cv2.polylines(frame, [left_eye_points.astype(np.int32)], True, (0, 255, 255), 1)
                cv2.polylines(frame, [right_eye_points.astype(np.int32)], True, (0, 255, 255), 1)
                
                # Calculate EAR for blink detection
                left_ear = self.blink_detector.eye_aspect_ratio(left_eye_points)
                right_ear = self.blink_detector.eye_aspect_ratio(right_eye_points)
                
                # Detect blink
                blink_detected, blink_duration = self.blink_detector.detect_blink(left_ear, right_ear)
                
                # Detect gaze direction
                gaze_direction = self.gaze_detector.get_gaze_direction(landmarks, width)
                
                # Handle navigation
                navigated = self.handle_navigation(gaze_direction, blink_detected, blink_duration)
                
                if navigated:
                    # Visual feedback for navigation
                    feedback_color = (0, 255, 0) if gaze_direction == "RIGHT" else (255, 0, 0)
                    cv2.putText(frame, f"SLIDE CHANGED!", 
                               (width//2 - 100, height//2), 
                               self.font, 1, feedback_color, 2)
            
            # Update attention monitor
            away_time = self.attention_monitor.get_away_time()
            self.attention_monitor.update_attention(face_detected)
            
            # Draw status panel
            frame = self.draw_status_panel(frame, gaze_direction, blink_detected, blink_duration, away_time)
            
            # Display EAR values
            if face_detected:
                cv2.putText(frame, f"EAR: L={left_ear:.2f} R={right_ear:.2f}", 
                           (width - 250, 160), self.font, 0.5, (255, 255, 0), 1)
            
            # Display blink feedback
            if self.blink_detector.blink_in_progress:
                elapsed = time.time() - self.blink_detector.blink_start_time
                cv2.putText(frame, f"BLINKING: {elapsed:.1f}s", 
                           (width//2 - 100, height - 50), self.font, 0.8, (0, 255, 255), 2)
                
                # Visual progress bar for 1 second target
                bar_width = int((elapsed / 1.0) * 200)
                cv2.rectangle(frame, (width//2 - 100, height - 30), 
                             (width//2 - 100 + bar_width, height - 20), 
                             (0, 255, 255), -1)
            
            # Display gaze direction
            color_map = {
                "LEFT": (0, 255, 0),    # Green
                "RIGHT": (0, 0, 255),   # Red
                "CENTER": (255, 255, 0), # Cyan
                "UNKNOWN": (128, 128, 128) # Gray
            }
            gaze_color = color_map.get(gaze_direction, (128, 128, 128))
            cv2.putText(frame, f"LOOKING: {gaze_direction}", 
                       (width//2 - 120, 220), self.font, 1, gaze_color, 2)
            
            # Visual indicator for gaze direction
            indicator_x = width//2
            if gaze_direction == "LEFT":
                indicator_x = left_boundary//2
            elif gaze_direction == "RIGHT":
                indicator_x = width - (width - right_boundary)//2
            
            cv2.circle(frame, (indicator_x, 270), 20, gaze_color, -1)
            cv2.circle(frame, (indicator_x, 270), 25, gaze_color, 2)
            
            # Show VISUAL dimming effect (dark overlay) when screen is dimmed
            if self.attention_monitor.screen_dimmed:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% dark overlay
                cv2.putText(frame, "⚠️  NOT LOOKING AT SCREEN!", 
                           (width//2 - 200, height//2 - 50), 
                           self.font, 1, (0, 0, 255), 3)
                cv2.putText(frame, "Screen visually dimmed", 
                           (width//2 - 150, height//2), 
                           self.font, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Look back to restore", 
                           (width//2 - 130, height//2 + 50), 
                           self.font, 0.8, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow(self.window_name, frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] 🔴 Shutting down...")
                break
            elif key == ord('n'):  # Manual next slide
                self.ppt_controller.next_slide()
                self.last_navigation_time = time.time()
                print("[TEST] Manual: Next slide")
            elif key == ord('p'):  # Manual previous slide
                self.ppt_controller.prev_slide()
                self.last_navigation_time = time.time()
                print("[TEST] Manual: Previous slide")
            elif key == ord('t'):  # Test popup
                print("[TEST] Testing popup...")
                PopupManager.show_popup("Test Popup", "This is a test notification!", "info")
            elif key == ord('d'):  # Force dim screen
                print("[TEST] Forcing screen dim...")
                self.attention_monitor.dim_screen()
            elif key == ord('r'):  # Force restore screen
                print("[TEST] Forcing screen restore...")
                self.attention_monitor.restore_screen()
        
        # Cleanup
        print("\n[SYSTEM] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] ✅ Application closed successfully")

def main():
    """Main function"""
    print("🚀 Initializing Attention-Aware PPT Reader...")
    print("⚡ Features:")
    print("   • 1-second blink navigation")
    print("   • Visual screen dimming")
    print("   • Popup notifications")
    print("   • Gaze detection")
    
    # Create and run application
    app = AttentionPPT()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Application interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📦 Install required packages:")
        print("   pip install opencv-python mediapipe pyautogui numpy")
        print("\n💡 For popup notifications, ensure tkinter is installed.")
    finally:
        # Ensure cleanup
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()