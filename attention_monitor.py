"""Attention monitoring with popup overlay."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from dataclasses import dataclass


@dataclass
class AttentionStatus:
    is_attentive: bool
    seconds_away: float
    showing_popup: bool


class AttentionMonitor:
    def __init__(self):
        self._last_face_seen_time = time.monotonic()
        self._popup_window = None
        self._popup_active = False
        self._popup_thread = None
        self._enabled = True
        self._lock = threading.Lock()
        self._attention_threshold = 2.0  # Seconds before showing popup
        
    def update_face_detection(self, face_detected: bool) -> AttentionStatus:
        with self._lock:
            now = time.monotonic()
            
            if face_detected:
                self._last_face_seen_time = now
                # Hide popup if it's showing and we have attention back
                if self._popup_active:
                    self._hide_popup()
                seconds_away = 0.0
                is_attentive = True
            else:
                seconds_away = now - self._last_face_seen_time
                is_attentive = seconds_away < self._attention_threshold
                
                # Show popup if enabled and user has been away long enough
                if (self._enabled and 
                    not self._popup_active and 
                    seconds_away >= self._attention_threshold):
                    self._show_popup()
            
            return AttentionStatus(
                is_attentive=is_attentive,
                seconds_away=seconds_away,
                showing_popup=self._popup_active
            )
    
    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled and self._popup_active:
                self._hide_popup()
    
    def set_threshold(self, seconds: float) -> None:
        """Set how many seconds of no face before showing popup."""
        self._attention_threshold = seconds
    
    def is_enabled(self) -> bool:
        return self._enabled
    
    def _show_popup(self) -> None:
        """Show popup on screen."""
        if self._popup_active:
            return
            
        try:
            # Create popup in a new thread to avoid blocking
            def create_popup():
                self._popup_window = tk.Tk()
                self._popup_window.title("Attention Alert")
                self._popup_window.overrideredirect(True)
                self._popup_window.attributes("-topmost", True)
                
                # Get screen size
                screen_w = self._popup_window.winfo_screenwidth()
                screen_h = self._popup_window.winfo_screenheight()
                
                # Create main frame with border
                frame = tk.Frame(self._popup_window, bg='red', padx=5, pady=5)
                frame.pack()
                
                # Inner frame with content
                inner_frame = tk.Frame(frame, bg='black', padx=40, pady=30)
                inner_frame.pack()
                
                # Warning icon and text
                icon_label = tk.Label(
                    inner_frame,
                    text="⚠️",
                    font=('Arial', 60),
                    fg='yellow',
                    bg='black'
                )
                icon_label.pack(pady=(0, 10))
                
                label = tk.Label(
                    inner_frame,
                    text="LOOK AT CAMERA!",
                    font=('Arial', 36, 'bold'),
                    fg='red',
                    bg='black'
                )
                label.pack()
                
                sub_label = tk.Label(
                    inner_frame,
                    text="Face not detected",
                    font=('Arial', 18),
                    fg='white',
                    bg='black'
                )
                sub_label.pack(pady=(10, 0))
                
                # Auto-hide after 3 seconds
                self._popup_window.after(3000, self._hide_popup)
                
                # Position in center
                self._popup_window.update_idletasks()
                w = self._popup_window.winfo_width()
                h = self._popup_window.winfo_height()
                x = (screen_w - w) // 2
                y = (screen_h - h) // 2
                self._popup_window.geometry(f"+{x}+{y}")
                
                # Make semi-transparent
                self._popup_window.attributes("-alpha", 0.92)
                
                # Bind click to dismiss
                self._popup_window.bind("<Button-1>", lambda e: self._hide_popup())
                
                self._popup_window.mainloop()
            
            # Start popup in separate thread
            self._popup_thread = threading.Thread(target=create_popup, daemon=True)
            self._popup_thread.start()
            self._popup_active = True
            print(f"\n[ATTENTION] ⚠️ No face detected for {self._attention_threshold}s! Showing 'LOOK AT CAMERA' popup!", flush=True)
            
        except Exception as e:
            print(f"[ATTENTION] Error showing popup: {e}", flush=True)
    
    def _hide_popup(self) -> None:
        """Hide popup."""
        if self._popup_window:
            try:
                self._popup_window.quit()
                self._popup_window.destroy()
            except:
                pass
            self._popup_window = None
        self._popup_active = False
    
    def close(self) -> None:
        """Clean up resources."""
        self._hide_popup()