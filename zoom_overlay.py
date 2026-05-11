"""Threaded always-on-top zoom overlay rendered with tkinter."""

from __future__ import annotations

import queue
import threading
from typing import Callable

import pyautogui
import tkinter as tk
from PIL import Image, ImageTk

from config import Config


class ZoomOverlay:
    """Borderless Tk overlay with screenshot magnification and smooth panning."""

    def __init__(self) -> None:
        self._command_queue: queue.Queue[tuple[Callable, tuple]] = queue.Queue()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._is_pan_locked = False  # Whether V-sign is controlling pan
        self._v_sign_active = False  # Whether V-sign is currently detected

    def show(self) -> None:
        self._enqueue(self._show_impl)

    def hide(self) -> None:
        self._enqueue(self._hide_impl)

    def update_pan(self, iris_ratio: float) -> None:
        """Update pan position using gaze (when V-sign not active)."""
        ratio = max(0.0, min(1.0, iris_ratio))
        if not self._is_pan_locked:
            # When V-sign is not active, pan follows gaze
            self._enqueue(self._update_target_pan_impl, ratio)
        else:
            # When V-sign is active, pan is controlled by finger position
            # Don't update from gaze
            pass

    def update_pan_from_finger(self, finger_x: float, finger_y: float) -> None:
        """
        Update pan position using finger position (when V-sign is active).
        Maps finger coordinates (0-1 normalized) to pan position.
        """
        # Map finger x (0-1) to pan position (0-1)
        # Clamp to avoid extreme edges
        pan = max(0.0, min(1.0, finger_x))
        self._enqueue(self._update_target_pan_impl, pan)
        self._enqueue(self._set_pan_lock_impl, True)

    def set_pan_lock(self, locked: bool) -> None:
        """Set whether pan is locked to finger control."""
        self._is_pan_locked = locked
        if not locked:
            # When unlocking, don't reset pan - just release control
            pass

    def set_v_sign_active(self, active: bool) -> None:
        """Set whether V-sign is currently active."""
        self._v_sign_active = active
        if not active:
            # Release pan lock when V-sign is no longer detected
            self._enqueue(self._set_pan_lock_impl, False)

    def stop(self) -> None:
        self._enqueue(self._stop_impl)
        self._thread.join(timeout=Config.OVERLAY_STOP_JOIN_TIMEOUT_SECONDS)

    def _enqueue(self, fn: Callable, *args) -> None:
        self._command_queue.put((fn, args))

    def _run_tk(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("Zoom Overlay")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.configure(bg=Config.OVERLAY_BG_COLOR)

        self._screen_w = self._root.winfo_screenwidth()
        self._screen_h = self._root.winfo_screenheight()
        x = (self._screen_w - Config.OVERLAY_WIDTH) // 2
        y = (self._screen_h - Config.OVERLAY_HEIGHT) // 2
        self._root.geometry(f"{Config.OVERLAY_WIDTH}x{Config.OVERLAY_HEIGHT}+{x}+{y}")

        self._label = tk.Label(self._root, bg=Config.OVERLAY_BG_COLOR, borderwidth=0)
        self._label.pack(fill="both", expand=True)

        self._is_visible = False
        self._running = True
        self._target_pan = 0.5
        self._current_pan = 0.5
        self._photo = None
        self._pan_locked = False

        self._root.after(Config.OVERLAY_REFRESH_MS, self._process_commands)
        self._root.after(Config.OVERLAY_REFRESH_MS, self._render_loop)
        self._root.mainloop()

    def _process_commands(self) -> None:
        while True:
            try:
                fn, args = self._command_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:
                # Keep the overlay alive even if a command fails.
                pass

        if self._running:
            self._root.after(Config.OVERLAY_REFRESH_MS, self._process_commands)

    def _show_impl(self) -> None:
        if not self._is_visible:
            self._root.deiconify()
            self._is_visible = True
            # Reset pan lock when showing
            self._pan_locked = False

    def _hide_impl(self) -> None:
        if self._is_visible:
            self._root.withdraw()
            self._is_visible = False
            self._pan_locked = False

    def _update_target_pan_impl(self, ratio: float) -> None:
        self._target_pan = ratio

    def _set_pan_lock_impl(self, locked: bool) -> None:
        self._pan_locked = locked

    def _stop_impl(self) -> None:
        self._running = False
        self._root.quit()
        self._root.destroy()

    def _render_loop(self) -> None:
        if self._running:
            if self._is_visible:
                # Only apply smoothing if not locked (locked means direct control)
                if not self._pan_locked:
                    self._current_pan += (self._target_pan - self._current_pan) * Config.OVERLAY_PAN_LERP_FACTOR
                else:
                    # When locked, follow target more directly
                    self._current_pan += (self._target_pan - self._current_pan) * 0.3
                self._draw_frame()
            self._root.after(Config.OVERLAY_REFRESH_MS, self._render_loop)

    def _draw_frame(self) -> None:
        screenshot = pyautogui.screenshot()
        screen_w, screen_h = screenshot.size

        crop_w = max(1, int(screen_w / Config.OVERLAY_ZOOM_FACTOR))
        crop_h = max(1, int(screen_h / Config.OVERLAY_ZOOM_FACTOR))

        min_cx = crop_w // 2
        max_cx = max(min_cx, screen_w - crop_w // 2)
        center_x = int(min_cx + (max_cx - min_cx) * self._current_pan)
        center_y = screen_h // 2

        left = max(0, min(screen_w - crop_w, center_x - crop_w // 2))
        top = max(0, min(screen_h - crop_h, center_y - crop_h // 2))
        right = left + crop_w
        bottom = top + crop_h

        cropped = screenshot.crop((left, top, right, bottom))
        resized = cropped.resize((Config.OVERLAY_WIDTH, Config.OVERLAY_HEIGHT), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._label.configure(image=self._photo)