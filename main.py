import cv2
import numpy as np
import time
from camera import Camera
from hand_tracking import HandTracker
from drawing_engine import DrawingEngine
from config import *

# ----------------------------
# 🚀 SYSTEM INITIALIZATION
# ----------------------------
print("[SYSTEM] Initializing Jahn AI Core...")
cam = Camera()
hand = HandTracker(detection_conf=0.8, track_conf=0.8)
draw_engine = DrawingEngine()

# State Variables
curr_color_idx = 0
active_mode = "Standby"
draw_engine.color = COLORS[curr_color_idx]

# ----------------------------
# 🎮 MAIN APPLICATION LOOP
# ----------------------------
while True:
    frame = cam.read()
    if frame is None:
        break
    
    # Flip frame for mirror effect (crucial for drawing)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # 1. 🔍 HAND TRACKING
    hand.find_hand(frame)
    lm_list = hand.get_landmarks(frame)
    fingers = hand.fingers_up(lm_list)
    
    # 2. 🧠 GESTURE LOGIC & DRAWING
    if lm_list:
        # Index tip coordinates
        x, y = lm_list[8]
        
        # --- ✍️ DRAW MODE (Index finger ONLY) ---
        if fingers == [0, 1, 0, 0, 0]:
            if active_mode != "Drawing":
                draw_engine.save_to_history()
                active_mode = "Drawing"
            
            # Draw on canvas
            draw_engine.draw(x, y)
            
            # Cursor feedback
            cv2.circle(frame, (x, y), 10, draw_engine.color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)

        # --- 🎨 SELECTION / HOVER MODE (Index + Middle) ---
        elif fingers == [0, 1, 1, 0, 0]:
            active_mode = "Selection"
            draw_engine.reset_tracking()
            
            # Visual Cursor for selection
            cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)
            cv2.line(frame, (lm_list[8][0], lm_list[8][1]), 
                     (lm_list[12][0], lm_list[12][1]), (255,255,255), 2)
            
            # UI Collision Check (Color Selection)
            if y < PANEL_HEIGHT:
                for i in range(len(COLORS)):
                    x_btn = 50 + i * 100
                    if x_btn - 30 < x < x_btn + 30:
                        curr_color_idx = i
                        draw_engine.color = COLORS[i]
                        # Visual feedback for selection
                        cv2.circle(frame, (x_btn, 50), 35, (255, 255, 255), 3)

        # --- 🧽 ERASER MODE (Fist / All Down) ---
        elif sum(fingers) == 0:
            if active_mode != "Erasing":
                draw_engine.save_to_history()
                active_mode = "Erasing"
            
            draw_engine.draw(x, y, is_erasing=True)
            # Eraser cursor
            cv2.circle(frame, (x, y), draw_engine.eraser_thickness, (200, 200, 200), 2)
            cv2.putText(frame, "ERASER", (x + 20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # --- 🗑️ CLEAR MODE (Full Palm / 5 Fingers) ---
        elif fingers == [1, 1, 1, 1, 1]:
            active_mode = "Clearing"
            draw_engine.clear_canvas()
            draw_engine.reset_tracking()
            cv2.putText(frame, "CANVAS CLEARED", (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 3)

        else:
            active_mode = "Standby"
            draw_engine.reset_tracking()
            # Hover cursor
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

    else:
        active_mode = "No Hand Detected"
        draw_engine.reset_tracking()

    # 3. ✨ POST-PROCESSING & UI
    # Combine frame with canvas using premium neon effect
    final_output = draw_engine.apply_neon_glow(frame)
    
    # Draw UI HUD
    final_output = draw_engine.draw_ui(final_output, curr_color_idx, active_mode)
    
    # 4. 📺 DISPLAY
    cv2.imshow(WINDOW_NAME, final_output)
    
    # Keys
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('u'): # Undo
        draw_engine.undo()
    elif key == ord('s'): # Save
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"drawing_{timestamp}.png", final_output)
        print(f"[SYSTEM] Snapshot saved: drawing_{timestamp}.png")

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("[SYSTEM] Jahn AI Core Terminated. Keep Creating.")
