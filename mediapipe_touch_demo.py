import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from game_logic import TouchGame

# SAME THRESHOLDS 
PART_THRESHOLDS = {
    "NOSE": 40,
    "LEFT EYE": 40,
    "RIGHT EYE": 40,
    "LIPS": 40,
    "CHIN": 45,
    "FOREHEAD": 50,
    "LEFT EAR": 55,
    "RIGHT EAR": 55,
    "LEFT SHOULDER": 60,
    "RIGHT SHOULDER": 60,
}
DEFAULT_THRESHOLD = 40

#  CONFIG 
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

CAM_INDEX    = 0
MIRROR_VIEW  = True
STABLE_FRAMES = 4
INDEX_TIP_ID  = 8

# Face landmark indices
NOSE_TIP          = 1
LEFT_EYE_CENTER   = 468
RIGHT_EYE_CENTER  = 473
UPPER_LIP         = 13
LOWER_LIP         = 14
LEFT_EYE_FALLBACK  = 33
RIGHT_EYE_FALLBACK = 263
LEFT_EAR   = 234
RIGHT_EAR  = 454
CHIN       = 152
FOREHEAD   = 10
LEFT_TEMPLE  = 127
RIGHT_TEMPLE = 356

# Shoulder offsets
SHOULDER_Y_OFFSET_SCALE = 0.85
SHOULDER_X_OFFSET_SCALE = 0.75

#  HELPERS 
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def to_px(nx, ny, w, h):
    return int(nx * w), int(ny * h)

def flip_label(label):
    if not MIRROR_VIEW:
        return label
    return "Left" if label == "Right" else "Right"

def point_to_rect_distance(px, py, x1, y1, x2, y2):
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return math.hypot(px - cx, py - cy)

def make_rect(cx, cy, rw, rh):
    return int(cx - rw/2), int(cy - rh/2), int(cx + rw/2), int(cy + rh/2)

#  MODEL FACTORY (called once per player) 
def make_face_landmarker():
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
    )

def make_hand_landmarker():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
    )

face_lm_p1 = make_face_landmarker()
face_lm_p2 = make_face_landmarker()
hand_lm_p1 = make_hand_landmarker()
hand_lm_p2 = make_hand_landmarker()

#  PER-HALF DETECTION 
def detect_half(frame_half, face_lm, hand_lm, timestamp_ms):
    """
    Runs face + hand detection on one half-frame.
    Draws annotations directly onto frame_half (in-place).
    Returns touch_now dict  {"Left": part_name_or_None, "Right": ...}
    """
    half_h, half_w = frame_half.shape[:2]
    rgb = cv2.cvtColor(frame_half, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    #  FACE 
    face_points = {}
    face_box    = None

    face_res = face_lm.detect_for_video(mp_image, timestamp_ms)
    if face_res.face_landmarks:
        lm = face_res.face_landmarks[0]

        def gp(i):
            if i >= len(lm):
                return None
            return to_px(lm[i].x, lm[i].y, half_w, half_h)

        nose      = gp(NOSE_TIP)
        l_eye     = gp(LEFT_EYE_CENTER) if LEFT_EYE_CENTER < len(lm) else gp(LEFT_EYE_FALLBACK)
        r_eye     = gp(RIGHT_EYE_CENTER) if RIGHT_EYE_CENTER < len(lm) else gp(RIGHT_EYE_FALLBACK)
        upper     = gp(UPPER_LIP)
        lower     = gp(LOWER_LIP)
        left_ear  = gp(LEFT_EAR)
        right_ear = gp(RIGHT_EAR)
        chin      = gp(CHIN)
        forehead  = gp(FOREHEAD)
        lt        = gp(LEFT_TEMPLE)
        rt        = gp(RIGHT_TEMPLE)

        lips = None
        if upper is not None and lower is not None:
            lips = ((upper[0] + lower[0]) // 2, (upper[1] + lower[1]) // 2)

        if forehead is None and lt is not None and rt is not None:
            forehead = ((lt[0] + rt[0]) // 2, (lt[1] + rt[1]) // 2)

        if nose      is not None: face_points["NOSE"]       = nose
        if l_eye     is not None: face_points["LEFT EYE"]   = l_eye
        if r_eye     is not None: face_points["RIGHT EYE"]  = r_eye
        if lips      is not None: face_points["LIPS"]        = lips
        if left_ear  is not None: face_points["LEFT EAR"]   = left_ear
        if right_ear is not None: face_points["RIGHT EAR"]  = right_ear
        if chin      is not None: face_points["CHIN"]        = chin
        if forehead  is not None: face_points["FOREHEAD"]   = forehead

        if l_eye is not None and r_eye is not None:
            eye_dist = dist(l_eye, r_eye)
            fw  = int(eye_dist * 2.4)
            cx  = (l_eye[0] + r_eye[0]) // 2
            cy  = (l_eye[1] + r_eye[1]) // 2
            face_box = make_rect(cx, cy, fw, int(eye_dist * 2.8))

    #  SHOULDERS 
    shoulder_rects = {}
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        fw  = fx2 - fx1
        cx  = (fx1 + fx2) // 2
        cy  = (fy1 + fy2) // 2
        y   = int(cy + fw * SHOULDER_Y_OFFSET_SCALE)
        lx  = int(cx - fw * SHOULDER_X_OFFSET_SCALE)
        rx  = int(cx + fw * SHOULDER_X_OFFSET_SCALE)
        shoulder_rects["LEFT SHOULDER"]  = make_rect(lx, y, fw, fw * 0.55)
        shoulder_rects["RIGHT SHOULDER"] = make_rect(rx, y, fw, fw * 0.55)

    #  HANDS 
    touch_now = {"Left": None, "Right": None}

    hand_res = hand_lm.detect_for_video(mp_image, timestamp_ms)
    if hand_res.hand_landmarks:
        for i, hand in enumerate(hand_res.hand_landmarks):
            raw   = hand_res.handedness[i][0].category_name
            label = flip_label(raw)

            tip    = hand[INDEX_TIP_ID]
            tip_pt = (int(tip.x * half_w), int(tip.y * half_h))

            cv2.circle(frame_half, tip_pt, 7, (0, 255, 255), -1)
            cv2.putText(frame_half, f"{label.upper()} HAND",
                        (tip_pt[0] + 10, tip_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            best_part = None
            best_d    = 10**9

            for name, pt in face_points.items():
                d = dist(tip_pt, pt)
                if d < best_d:
                    best_d    = d
                    best_part = name

            for name, (x1, y1, x2, y2) in shoulder_rects.items():
                d = point_to_rect_distance(tip_pt[0], tip_pt[1], x1, y1, x2, y2)
                if d < best_d:
                    best_d    = d
                    best_part = name

            th = PART_THRESHOLDS.get(best_part, DEFAULT_THRESHOLD)
            if best_part and best_d <= th:
                touch_now[label] = best_part

    return touch_now

# starting state 
stable_counter = {
    "P1": {"Left": 0, "Right": 0},
    "P2": {"Left": 0, "Right": 0},
}
stable_touch = {
    "P1": {"Left": None, "Right": None},
    "P2": {"Left": None, "Right": None},
}

game = TouchGame(duration=5)

window_name = "Children's Touch Game"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

cap          = cv2.VideoCapture(CAM_INDEX)
timestamp_ms = 0
print("Press Q to quit")

#  MAIN LOOP 
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if MIRROR_VIEW:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    mid  = w // 2
    timestamp_ms += 33

    # Split into two halves 
    half_p1 = frame[:, :mid].copy()
    half_p2 = frame[:, mid:].copy()

    # Run detection on each half independently
    tn1 = detect_half(half_p1, face_lm_p1, hand_lm_p1, timestamp_ms)
    tn2 = detect_half(half_p2, face_lm_p2, hand_lm_p2, timestamp_ms)

    frame[:, :mid] = half_p1
    frame[:, mid:] = half_p2

    # White divider line down the middle
    cv2.line(frame, (mid, 0), (mid, h), (255, 255, 255), 2)

    # Player labels at top of each half
    cv2.putText(frame, "PLAYER 1", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "PLAYER 2", (mid + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    #  debounce touch inpput 
    for hand in ("Left", "Right"):
        for player, tn in [("P1", tn1), ("P2", tn2)]:
            if tn[hand]:
                stable_counter[player][hand] += 1
                if stable_counter[player][hand] >= STABLE_FRAMES:
                    stable_touch[player][hand] = tn[hand]
            else:
                stable_counter[player][hand] = 0
                stable_touch[player][hand] = None

    # game update
    game.update_target()

    cv2.putText(frame, f"TOUCH YOUR {game.current_target}",
                (w // 2 - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame,
                f"P1: {game.scores['Player 1']}  |  P2: {game.scores['Player 2']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # scores
    player_cfg = [
        ("P1", "Player 1", (0, 255, 0),  120),
        ("P2", "Player 2", (255, 0, 0), 160),
    ]

    for player_key, player_name, color, y_pos in player_cfg:
        touches = [v for v in stable_touch[player_key].values() if v]
        if touches:
            part   = touches[0]
            scored = game.check_winner(player_name, [part])

            cv2.putText(frame, f"{player_name}: TOUCHING {part}",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if scored:
                cv2.putText(frame, f"{player_name} - NICE!",
                            (20, y_pos + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.putText(frame, "Q = quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()