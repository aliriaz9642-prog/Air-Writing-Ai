import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, detection_conf=0.8, track_conf=0.8):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    def find_hand(self, frame):
        """
        Process frame and detect hand landmarks
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return self.results

    def get_landmarks(self, frame):
        """
        Returns a list of all 21 landmarks in pixel coordinates
        """
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            for id, lm in enumerate(hand.landmark):
                lm_list.append([int(lm.x * w), int(lm.y * h)])
        return lm_list

    def fingers_up(self, lm_list):
        """
        Detect which fingers are up
        Returns list: [thumb, index, middle, ring, pinky]
        """
        fingers = []
        if not lm_list:
            return [0, 0, 0, 0, 0]

        # Thumb (special logic because it moves sideways)
        # We check if thumb tip is to the left/right of the IP joint based on hand orientation
        # Simplified: check x coordinate relative to joint
        if lm_list[self.tip_ids[0]][0] > lm_list[self.tip_ids[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for i in range(1, 5):
            if lm_list[self.tip_ids[i]][1] < lm_list[self.tip_ids[i] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
