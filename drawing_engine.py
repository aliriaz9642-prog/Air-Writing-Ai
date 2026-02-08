import cv2
import numpy as np
import time
from config import *

class DrawingEngine:
    def __init__(self):
        # Canvas layers
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        self.preview_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        
        # Drawing state
        self.prev_x, self.prev_y = None, None
        self.curr_x, self.curr_y = None, None
        
        # Stroke Smoothing (EMA)
        self.ema_x, self.ema_y = None, None
        self.alpha = EMA_ALPHA
        
        # Style
        self.color = COLORS[0]
        self.thickness = DEFAULT_THICKNESS
        self.eraser_thickness = ERASER_THICKNESS
        
        # History for Undo
        self.history = []
        self.max_history = 30

    def smooth_point(self, x, y):
        """Apply EMA smoothing to coordinates."""
        if self.ema_x is None:
            self.ema_x, self.ema_y = x, y
        else:
            self.ema_x = self.alpha * x + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * y + (1 - self.alpha) * self.ema_y
        return int(self.ema_x), int(self.ema_y)

    def draw(self, x, y, is_erasing=False):
        """Draw on the canvas with smoothed points and neon effects."""
        sx, sy = self.smooth_point(x, y)
        
        if self.prev_x is not None:
            if is_erasing:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (sx, sy), 
                         (0, 0, 0), self.eraser_thickness, cv2.LINE_AA)
            else:
                # Premium Neon Stroke:
                # 1. Outer Glow (Thicker, darker/softer)
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (sx, sy), 
                         self.color, self.thickness + 4, cv2.LINE_AA)
                # 2. Inner Core (Thinner, brighter/white)
                # We can mix the color with white for the core
                white_core = (255, 255, 255)
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (sx, sy), 
                         white_core, self.thickness // 2, cv2.LINE_AA)
        
        self.prev_x, self.prev_y = sx, sy
        return sx, sy

    def save_to_history(self):
        """Save current canvas for undo."""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.canvas.copy())

    def undo(self):
        """Restore last saved state."""
        if self.history:
            self.canvas = self.history.pop()
            return True
        return False

    def reset_tracking(self):
        """Reset smoothing and previous points when finger is lifted."""
        self.prev_x, self.prev_y = None, None
        self.ema_x, self.ema_y = None, None

    def clear_canvas(self):
        """Wipe the board."""
        self.save_to_history()
        self.canvas[:] = 0
        self.reset_tracking()

    def apply_neon_glow(self, frame):
        """Combines the canvas with the frame and adds a neon glow effect."""
        # 1. Create a blur of the canvas for the glow
        glow = cv2.GaussianBlur(self.canvas, (15, 15), 0)
        
        # 2. Add the glow back to the canvas (additive blend)
        # We use a copy to avoid mutating the base canvas with blur every frame
        glow_canvas = cv2.addWeighted(self.canvas, 1.0, glow, 0.7, 0)
        
        # 3. Layer the glowing canvas onto the camera frame
        # Mask where the canvas has drawing
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Black out the area of drawing in the frame
        img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        
        # Take only drawing from canvas
        img_fg = cv2.bitwise_and(glow_canvas, glow_canvas, mask=mask)
        
        # Combine
        return cv2.add(img_bg, img_fg)

    def draw_ui(self, frame, active_color_idx, current_mode):
        """Draws a premium HUD for the application."""
        h, w, _ = frame.shape
        
        # Top Panel
        cv2.rectangle(frame, (0, 0), (w, PANEL_HEIGHT), HUD_COLOR, -1)
        cv2.line(frame, (0, PANEL_HEIGHT), (w, PANEL_HEIGHT), ACCENT_COLOR, 2)
        
        # Color Palettes
        for i, color in enumerate(COLORS):
            x_pos = 50 + i * 100
            y_pos = 50
            radius = 25
            
            # Outer ring if active
            if i == active_color_idx:
                cv2.circle(frame, (x_pos, y_pos), radius + 5, (255, 255, 255), 2)
            
            cv2.circle(frame, (x_pos, y_pos), radius, color, -1)
            cv2.circle(frame, (x_pos, y_pos), radius, (200, 200, 200), 1)

        # Mode Display
        mode_text = f"MODE: {current_mode.upper()}"
        cv2.putText(frame, mode_text, (w - 300, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        instr = "Index: DRAW | Index+Middle: SELECT | Fist: ERASE | Palm: CLEAR"
        cv2.putText(frame, instr, (50, PANEL_HEIGHT + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
