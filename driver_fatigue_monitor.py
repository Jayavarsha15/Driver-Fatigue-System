"""
====================================================================
Driver Drowsiness Detection System — LAPTOP SIDE
laptop_detection.py
====================================================================
Libraries: opencv-python, mediapipe, numpy, scipy, requests,
           collections, threading, time
====================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import time
import threading
import collections
import math

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────

PI_BASE_URL = "http://team49.local:5000"   # Raspberry Pi mDNS address

# Calibration timing
CALIB_EAR_SECONDS   = 20    # eyes-open calibration duration
CALIB_MAR_FRAMES    = 150   # baseline MAR frames

# EAR / MAR detection params
EAR_CONSEC_FRAMES   = 15    # frames below threshold → eyes closed (faster response)
MAR_STABLE_FRAMES   = 15    # frames above threshold → yawn confirmed

# Head-pose
# Pitch threshold 15° + nose ratio 0.65 — fast, accurate head-down detection
HEAD_PITCH_THRESH   = 15.0  # degrees: pitch > 15 → head down
NOSE_RATIO_THRESH   = 0.65  # nose y-ratio in face bounding box > 0.65 → head down

# Fatigue weights (must sum to ~1.0)
W_PERCLOS           = 0.35
W_BLINK_DROP        = 0.30
W_YAWN_FREQ         = 0.30
W_HEAD_POSE         = 0.05  # small supplement

# Faster exponential smoothing: EXP_ALPHA = 0.35 (was 0.15)
EXP_ALPHA           = 0.35  # higher = faster reaction to drowsiness signals

# Fatigue thresholds — raised MILD to 50 to eliminate false alerts
FATIGUE_MILD_MIN    = 50    # score ≥ 50 → MILD
FATIGUE_HIGH_MIN    = 70    # score ≥ 70 → HIGH DROWSINESS

# Alert cooldowns
VOICE_COOLDOWN_SEC  = 10.0  # min gap between voice alerts
BUZZER_COOLDOWN_SEC = 5.0

# Gyro instability: no delay — single reading above threshold is enough
# Threshold lowered 12000 → 3000 for immediate zig-zag detection
GYRO_UNSTABLE_THRESH = 3000    # abs(gx) or abs(gy) > 3000 → unstable NOW
# GYRO_UNSTABLE_SECS removed: instant detection, no accumulation window

# Eye closure duration for HIGH trigger
EYE_CLOSED_SECS     = 1.0

# Stability guard: NORMAL / MILD transitions require this many seconds
# of continuous qualification before the state actually changes.
STATE_STABILITY_SECS = 2.0    # seconds

# ──────────────────────────────────────────────────────────────────
# MEDIAPIPE FACE MESH SETUP
# ──────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode        = False,
    max_num_faces            = 1,
    refine_landmarks         = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)
mp_drawing   = mp.solutions.drawing_utils
mp_draw_sty  = mp.solutions.drawing_styles

# ──────────────────────────────────────────────────────────────────
# LANDMARK INDICES
# ──────────────────────────────────────────────────────────────────

# EAR: per the prompt spec
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MAR: 4-point mouth
MOUTH_IDX = [61, 291, 13, 14]   # left, right, top, bottom

# Head-pose reference (nose, chin, eye corners, mouth corners)
HP_IDX = [1, 152, 33, 263, 61, 291]

# 3-D face model for solvePnP — generic face (mm)
MODEL_3D = np.array([
    (  0.0,    0.0,    0.0),   # nose tip
    (  0.0, -330.0,  -65.0),   # chin
    (-225.0,  170.0, -135.0),  # left eye corner
    ( 225.0,  170.0, -135.0),  # right eye corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    ( 150.0, -150.0, -125.0),  # right mouth corner
], dtype="double")

# ──────────────────────────────────────────────────────────────────
# UTILITY — EUCLIDEAN DISTANCE
# ──────────────────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ──────────────────────────────────────────────────────────────────
# EAR COMPUTATION
# ──────────────────────────────────────────────────────────────────

def compute_ear(lm, eye_idx, W, H):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Indices: p1=0, p2=1, p3=2, p4=3, p5=4, p6=5 in eye_idx list.
    """
    pts = [(int(lm[i].x * W), int(lm[i].y * H)) for i in eye_idx]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

# ──────────────────────────────────────────────────────────────────
# MAR COMPUTATION
# ──────────────────────────────────────────────────────────────────

def compute_mar(lm, W, H):
    """
    MAR = vertical_distance / horizontal_distance
    MOUTH_IDX = [left(61), right(291), top(13), bottom(14)]
    """
    pts = [(int(lm[i].x * W), int(lm[i].y * H)) for i in MOUTH_IDX]
    vertical   = dist(pts[2], pts[3])   # top–bottom
    horizontal = dist(pts[0], pts[1])   # left–right
    return vertical / (horizontal + 1e-6)

# ──────────────────────────────────────────────────────────────────
# HEAD POSE ESTIMATION
# ──────────────────────────────────────────────────────────────────

def get_camera_matrix(W, H):
    focal  = W
    center = (W / 2.0, H / 2.0)
    cam_mat = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ],
    ], dtype="double")
    dist_c = np.zeros((4, 1))
    return cam_mat, dist_c

def estimate_head_pose(lm, W, H):
    """
    Returns (pitch, yaw) in degrees.
    Pitch is normalised to the range (-90, +90] to prevent the
    Euler-angle wrap-around that produces spurious -179° readings.
    pitch > HEAD_PITCH_THRESH (15°) → head down.
    """
    img_pts = np.array([
        (lm[1  ].x * W, lm[1  ].y * H),   # nose
        (lm[152].x * W, lm[152].y * H),   # chin
        (lm[33 ].x * W, lm[33 ].y * H),   # left eye corner
        (lm[263].x * W, lm[263].y * H),   # right eye corner
        (lm[61 ].x * W, lm[61 ].y * H),   # left mouth
        (lm[291].x * W, lm[291].y * H),   # right mouth
    ], dtype="double")

    cam_mat, dist_c = get_camera_matrix(W, H)
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_3D, img_pts, cam_mat, dist_c,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0

    rmat, _  = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    # euler[i] is a 1-element numpy array; .item() extracts a plain Python
    # float and avoids the NumPy 1.25 DeprecationWarning.
    pitch = float(euler[0].item())
    yaw   = float(euler[1].item())

    # ── Pitch normalisation — fixes the -179° / +179° wrap-around ──
    # decomposeProjectionMatrix can return pitch near ±180° when the
    # face tilts past the gimbal-lock boundary.  Folding into (-90, 90]
    # gives a consistent "positive = chin down" convention.
    if pitch < -90:
        pitch = pitch + 180
    elif pitch > 90:
        pitch = pitch - 180

    return pitch, yaw

def nose_ratio_head_down(lm, W, H):
    """
    [CHANGE 3] Returns True if nose y-position exceeds NOSE_RATIO_THRESH
    (lowered from 0.70 → 0.65) relative to face bounding box height.
    Triggers faster than pitch alone for forward-slump detection.
    """
    ys = [lm[i].y for i in range(len(lm))]
    face_top    = min(ys) * H
    face_bottom = max(ys) * H
    face_height = face_bottom - face_top + 1e-6
    nose_y      = lm[1].y * H
    ratio = (nose_y - face_top) / face_height
    return ratio > NOSE_RATIO_THRESH

# ──────────────────────────────────────────────────────────────────
# ADAPTIVE CALIBRATION STATE
# ──────────────────────────────────────────────────────────────────

class Calibration:
    def __init__(self):
        self.ear_samples  = []
        self.mar_samples  = []
        self.ear_done     = False
        self.mar_done     = False
        self.ear_thresh   = 0.21   # fallback default
        self.mar_thresh   = 0.50   # fallback default
        self.ear_start_t  = None   # wall-clock start of EAR calibration

    @property
    def done(self):
        return self.ear_done and self.mar_done

    def update_ear(self, ear_val, now):
        """Collect EAR during first CALIB_EAR_SECONDS seconds."""
        if self.ear_done:
            return
        if self.ear_start_t is None:
            self.ear_start_t = now
        elapsed = now - self.ear_start_t
        if elapsed < CALIB_EAR_SECONDS:
            self.ear_samples.append(ear_val)
        else:
            if self.ear_samples:
                self.ear_thresh = 0.80 * float(np.median(self.ear_samples))
            self.ear_done = True
            print(f"[CALIB] EAR threshold set → {self.ear_thresh:.4f}")

    def update_mar(self, mar_val):
        """Collect first CALIB_MAR_FRAMES MAR samples."""
        if self.mar_done:
            return
        self.mar_samples.append(mar_val)
        if len(self.mar_samples) >= CALIB_MAR_FRAMES:
            baseline          = float(np.median(self.mar_samples))
            self.mar_thresh   = baseline * 1.4
            self.mar_done     = True
            print(f"[CALIB] MAR threshold set → {self.mar_thresh:.4f}")

calib = Calibration()

# ──────────────────────────────────────────────────────────────────
# FATIGUE SCORE SIGNALS
# ──────────────────────────────────────────────────────────────────

# Rolling windows
PERCLOS_WINDOW_SEC  = 60       # PERCLOS over last 60 s
BLINK_WINDOW_SEC    = 60       # blink rate over last 60 s
YAWN_WINDOW_SEC     = 300      # yawn frequency over last 5 min

perclos_log   = collections.deque()   # (timestamp, closed:bool)
blink_log     = collections.deque()   # timestamps of blink events
yawn_log      = collections.deque()   # timestamps of yawn events

# Per-frame counters
ear_consec_count   = 0
eye_was_closed     = False
mar_stable_count   = 0
mouth_was_open     = False
eye_closed_start   = None     # wall time when eyes first closed

def update_perclos(ear, threshold, now):
    """Track whether eyes are closed each frame; return PERCLOS 0-1."""
    global ear_consec_count, eye_was_closed, eye_closed_start

    closed = ear < threshold
    perclos_log.append((now, closed))

    # Trim old entries
    cutoff = now - PERCLOS_WINDOW_SEC
    while perclos_log and perclos_log[0][0] < cutoff:
        perclos_log.popleft()

    total   = len(perclos_log)
    n_close = sum(1 for _, c in perclos_log if c)
    perclos = n_close / max(total, 1)

    # Track eye-closed duration for HIGH trigger
    if closed:
        if eye_closed_start is None:
            eye_closed_start = now
    else:
        eye_closed_start = None

    return perclos

def update_blink(ear, threshold, now):
    """Detect blinks; return blinks-per-minute."""
    global ear_consec_count, eye_was_closed

    closed = ear < threshold
    if closed:
        ear_consec_count += 1
    else:
        if eye_was_closed and ear_consec_count < EAR_CONSEC_FRAMES:
            # Short closure = blink (not prolonged closure)
            blink_log.append(now)
        ear_consec_count = 0
    eye_was_closed = closed

    cutoff = now - BLINK_WINDOW_SEC
    while blink_log and blink_log[0] < cutoff:
        blink_log.popleft()

    return len(blink_log)   # blinks in last 60 s (normal ~15-20)

def update_yawn(mar, now):
    """Detect yawns using adaptive MAR threshold; return yawns-per-5min."""
    global mar_stable_count, mouth_was_open

    # Yawn condition: MAR > threshold * 1.25 for MAR_STABLE_FRAMES frames
    open_now = mar > calib.mar_thresh * 1.25
    if open_now:
        mar_stable_count += 1
    else:
        if mouth_was_open and mar_stable_count >= MAR_STABLE_FRAMES:
            yawn_log.append(now)
        mar_stable_count = 0
    mouth_was_open = open_now

    cutoff = now - YAWN_WINDOW_SEC
    while yawn_log and yawn_log[0] < cutoff:
        yawn_log.popleft()

    return len(yawn_log)

smoothed_fatigue = 0.0

def compute_fatigue(perclos, blinks_per_min, yawns_5min, head_pitch_deg):
    """
    Weighted fusion → fatigue score 0-100.
    [CHANGE 1] Uses faster exponential smoothing: EXP_ALPHA = 0.35 (was 0.15).
    Higher alpha makes score react faster to real drowsiness signals.

    PERCLOS:      0-1   (1 = always closed)
    Blink drop:   normal ~15-20/min; <10 = fatigue
    Yawn freq:    0-10+ in 5 min; normalise to 0-1
    Head pose:    deviation from 0 degrees
    """
    global smoothed_fatigue

    # Normalise each signal to 0-1
    perclos_score = min(perclos * 3.0, 1.0)          # 0.33 PERCLOS → score 1.0

    normal_blink  = 17.5                              # midpoint of 15-20
    blink_drop    = max(0.0, normal_blink - blinks_per_min) / normal_blink
    blink_score   = min(blink_drop, 1.0)

    yawn_score    = min(yawns_5min / 5.0, 1.0)       # 5 yawns/5 min → score 1.0

    head_score    = min(abs(head_pitch_deg) / 45.0, 1.0)

    raw = (W_PERCLOS    * perclos_score +
           W_BLINK_DROP * blink_score   +
           W_YAWN_FREQ  * yawn_score    +
           W_HEAD_POSE  * head_score)

    raw_100 = raw * 100.0

    # Exponential smoothing
    smoothed_fatigue = (EXP_ALPHA * raw_100 +
                        (1 - EXP_ALPHA) * smoothed_fatigue)
    return round(smoothed_fatigue, 1)

# ──────────────────────────────────────────────────────────────────
# RASPBERRY PI API CALLS  (function names fixed)
# ──────────────────────────────────────────────────────────────────

def get_sensor_data():
    """GET /sensor → dict or None."""
    try:
        r = requests.get(f"{PI_BASE_URL}/sensor", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[API] /sensor error: {e}")
    return None

def send_voice_alert():
    """POST /voice."""
    try:
        r = requests.post(f"{PI_BASE_URL}/voice", timeout=3)
        print(f"[API] /voice → {r.status_code}")
    except Exception as e:
        print(f"[API] /voice error: {e}")

def send_buzzer_alert():
    """POST /buzzer."""
    try:
        r = requests.post(f"{PI_BASE_URL}/buzzer", timeout=3)
        print(f"[API] /buzzer → {r.status_code}")
    except Exception as e:
        print(f"[API] /buzzer error: {e}")

def stop_alert():
    """POST /stop."""
    try:
        r = requests.post(f"{PI_BASE_URL}/stop", timeout=3)
        print(f"[API] /stop → {r.status_code}")
    except Exception as e:
        print(f"[API] /stop error: {e}")

# ──────────────────────────────────────────────────────────────────
# SHARED STATE  (written by CV thread, read by alert thread)
# ──────────────────────────────────────────────────────────────────

state_lock   = threading.Lock()
shared_state = {
    "fatigue"       : 0.0,
    "head_down"     : False,
    "ear"           : 0.3,
    "mar"           : 0.0,
    "pitch"         : 0.0,
    "yaw"           : 0.0,
    "blinks_pm"     : 0.0,
    "perclos"       : 0.0,
    "yawns_5m"      : 0,
    "alert_state"   : "NORMAL",
    # [CHANGE 8] gyro_unstable exposed so HUD can display it
    "gyro_unstable" : False,
}

# ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
#  ALERT DECISION THREAD — 3-CASE LOGIC
# ══════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────

last_voice_t  = 0.0
last_buzzer_t = 0.0
current_alert = "NORMAL"   # track active state to send stop correctly

# Stability timers — track how long we have been continuously in each zone
_mild_first_seen_t   = None   # when score first rose above FATIGUE_MILD_MIN
_normal_first_seen_t = None   # when score first dropped below FATIGUE_MILD_MIN

def alert_decision_loop():
    """
    Background alert-decision thread.  Runs every ~0.3 s.
    ─────────────────────────────────────────────────────────────────
    Priority order (first match wins on every tick):

    CASE 3 — HIGH DROWSINESS   ← NO stability delay
        ANY of:  head_down  |  fatigue ≥ 70  |  eyes closed ≥ 1 s
        Buzzer:  (head_down OR fatigue ≥ 70) AND gyro_unstable AND STRAIGHT
                 → fires immediately; BUZZER_COOLDOWN_SEC prevents re-fire
        Voice:   all other HIGH conditions → once per VOICE_COOLDOWN_SEC

    CASE 2 — MILD FATIGUE      ← NO stability delay
        50 ≤ fatigue < 70  AND  head_down == False
        Voice once per VOICE_COOLDOWN_SEC

    CASE 1 — NORMAL
        fatigue < 50  AND  head_down == False
        /stop sent once on transition (STATE_STABILITY_SECS guard kept
        only to avoid /stop spam during brief score dips)
    ─────────────────────────────────────────────────────────────────
    """
    global last_voice_t, last_buzzer_t, current_alert
    global _mild_first_seen_t, _normal_first_seen_t

    while True:
        with state_lock:
            fatigue   = shared_state["fatigue"]
            head_down = shared_state["head_down"]

        now = time.time()

        # ── Eye-closed duration ───────────────────────────────────
        eyes_closed_long = (
            eye_closed_start is not None and
            (now - eye_closed_start) >= EYE_CLOSED_SECS
        )

        # ── Fetch Pi sensor data ──────────────────────────────────
        sensor    = get_sensor_data()
        raw_gx    = 0
        raw_gy    = 0
        road_type = "STRAIGHT"

        if sensor:
            raw_gx    = sensor.get("gyro", {}).get("gx", 0)
            raw_gy    = sensor.get("gyro", {}).get("gy", 0)
            road_type = sensor.get("gps", {}).get("road_type", "STRAIGHT").upper()

        # ── Gyro: instant single-sample, threshold 3000 ───────────
        # Single reading above 3000 raw units = unstable immediately.
        # No time-based accumulation — reacts to first strong shake.
        gyro_unstable = (abs(raw_gx) > GYRO_UNSTABLE_THRESH or
                         abs(raw_gy) > GYRO_UNSTABLE_THRESH)

        with state_lock:
            shared_state["gyro_unstable"] = gyro_unstable

        # Diagnostic: log sensor values so misclassification is visible
        if gyro_unstable or road_type == "HILL":
            print(f"[SENSOR] gx={raw_gx}  gy={raw_gy}  "
                  f"gyro_unstable={gyro_unstable}  road={road_type}")

        # ═════════════════════════════════════════════════════════
        # CASE 3 — HIGH DROWSINESS  (no stability delay)
        # ═════════════════════════════════════════════════════════
        high_trigger = (head_down or
                        fatigue >= FATIGUE_HIGH_MIN or
                        eyes_closed_long)

        if high_trigger:
            # Reset lower-state timers so re-entry to MILD/NORMAL is clean
            _mild_first_seen_t   = None
            _normal_first_seen_t = None

            if current_alert != "HIGH":
                reason = ("HEAD DOWN"   if head_down        else
                          "EYES CLOSED" if eyes_closed_long else
                          f"FATIGUE={fatigue:.1f}")
                print(f"[STATE] → HIGH  reason={reason}  "
                      f"gyro={gyro_unstable}  road={road_type}")
            current_alert = "HIGH"

            # ── Buzzer condition ──────────────────────────────────
            # Fires when danger is confirmed on a STRAIGHT road.
            # road_type is now STRAIGHT unless tilt > 18° (real hill),
            # so normal riding no longer suppresses this condition.
            #
            # buzzer_danger: (head_down OR fatigue≥70)
            #                AND gyro_unstable
            #                AND road_type == "STRAIGHT"
            #
            # safe_hill_tilt: gyro_unstable + HILL + head NORMAL
            #   → vehicle leaning on a genuine steep slope, suppress buzzer.
            #
            # BUZZER_COOLDOWN_SEC prevents re-fire every 0.3 s loop tick.
            buzzer_danger = (
                (head_down or fatigue >= FATIGUE_HIGH_MIN) and
                gyro_unstable and
                road_type == "STRAIGHT"
            )
            safe_hill_tilt = (
                gyro_unstable and
                road_type == "HILL" and
                not head_down
            )

            if buzzer_danger:
                if (now - last_buzzer_t) >= BUZZER_COOLDOWN_SEC:
                    reason_b = "head_down" if head_down else f"fatigue={fatigue:.1f}"
                    print(f"[DECISION] BUZZER — {reason_b} + gyro + STRAIGHT")
                    send_buzzer_alert()
                    last_buzzer_t = now

            elif safe_hill_tilt:
                # Hill tilt with head normal — suppress everything
                print("[DECISION] Safe hill tilt — NO alert")
                if current_alert != "NORMAL":
                    stop_alert()

            else:
                # HIGH but no buzzer condition → voice only
                if (now - last_voice_t) >= VOICE_COOLDOWN_SEC:
                    print("[DECISION] HIGH — VOICE only")
                    send_voice_alert()
                    last_voice_t = now

            time.sleep(0.3)
            continue

        # ═════════════════════════════════════════════════════════
        # CASE 2 — MILD FATIGUE  (no stability delay)
        # ═════════════════════════════════════════════════════════
        if fatigue >= FATIGUE_MILD_MIN:
            _normal_first_seen_t = None
            _mild_first_seen_t   = None   # not needed without delay

            if current_alert != "MILD":
                print(f"[STATE] → MILD  score={fatigue:.1f}")
            current_alert = "MILD"

            if (now - last_voice_t) >= VOICE_COOLDOWN_SEC:
                print(f"[STATE] MILD FATIGUE — score={fatigue:.1f} → VOICE")
                send_voice_alert()
                last_voice_t = now

            time.sleep(0.4)
            continue

        # ═════════════════════════════════════════════════════════
        # CASE 1 — NORMAL
        # Stability guard only on the /stop call to prevent rapid
        # on/off toggling when score hovers near the boundary.
        # ═════════════════════════════════════════════════════════
        _mild_first_seen_t = None

        if _normal_first_seen_t is None:
            _normal_first_seen_t = now

        normal_stable = (now - _normal_first_seen_t) >= STATE_STABILITY_SECS

        if normal_stable and current_alert != "NORMAL":
            stop_alert()
            current_alert = "NORMAL"
            print(f"[STATE] NORMAL — score={fatigue:.1f}")

        time.sleep(0.5)

# ──────────────────────────────────────────────────────────────────
# HUD OVERLAY
# ──────────────────────────────────────────────────────────────────

def draw_hud(frame, s):
    """Render on-screen data from shared_state dict s."""
    lines = [
        f"EAR:    {s['ear']:.3f}",
        f"MAR:    {s['mar']:.3f}",
        f"Pitch:  {s['pitch']:.1f}  Yaw: {s['yaw']:.1f}",
        f"PERCLOS:{s['perclos']*100:.1f}%",
        f"Blinks: {s['blinks_pm']:.0f}/min",
        f"Yawns:  {s['yawns_5m']}  (5 min)",
        f"Fatigue:{s['fatigue']:.1f}",
        # [CHANGE 8] Show head_down and gyro state on HUD for diagnostics
        f"Head:   {'DOWN' if s['head_down'] else 'OK'}  "
        f"Gyro: {'SHAKE' if s.get('gyro_unstable') else 'OK'}",
        f"State:  {s['alert_state']}",
    ]
    state_color = {
        "NORMAL": (0, 200, 0),
        "MILD"  : (0, 165, 255),
        "HIGH"  : (0, 0, 255),
    }.get(s["alert_state"], (255, 255, 255))

    for i, line in enumerate(lines):
        color = state_color if "State" in line else (220, 220, 220)
        cv2.putText(frame, line, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {W}×{H} — starting calibration phase...")

    # Start alert decision thread
    t_alert = threading.Thread(target=alert_decision_loop, daemon=True)
    t_alert.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # Default values when no face detected
        ear   = 0.30
        mar   = 0.00
        pitch = 0.00
        yaw   = 0.00
        head_down = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # ── EAR ──────────────────────────────────────────────
            l_ear = compute_ear(lm, LEFT_EYE,  W, H)
            r_ear = compute_ear(lm, RIGHT_EYE, W, H)
            ear   = (l_ear + r_ear) / 2.0

            # ── MAR ──────────────────────────────────────────────
            mar = compute_mar(lm, W, H)

            # ── Head pose ─────────────────────────────────────────
            # estimate_head_pose() now returns a normalised pitch
            # (wrap-around fix applied inside the function).
            pitch, yaw = estimate_head_pose(lm, W, H)

            # head_down triggers on EITHER condition (OR logic).
            # pitch > 15°  covers forward chin-drop.
            # nose_ratio   covers forward slump even before pitch catches up.
            head_down = (pitch > HEAD_PITCH_THRESH or
                         nose_ratio_head_down(lm, W, H))

            # ── Draw mesh ─────────────────────────────────────────
            mp_drawing.draw_landmarks(
                frame,
                res.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_sty
                    .get_default_face_mesh_contours_style(),
            )

        # ── Calibration update ────────────────────────────────────
        calib.update_ear(ear, now)
        calib.update_mar(mar)

        if not calib.done:
            remain = max(0, CALIB_EAR_SECONDS - (
                now - (calib.ear_start_t or now)))
            cv2.putText(frame,
                        f"CALIBRATING... {remain:.0f}s  (keep eyes open)",
                        (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Compute fatigue signals ───────────────────────────────
        perclos     = update_perclos(ear, calib.ear_thresh, now)
        blinks_pm   = update_blink(ear, calib.ear_thresh, now)
        yawns_5m    = update_yawn(mar, now)
        fatigue     = compute_fatigue(perclos, blinks_pm, yawns_5m, pitch)

        # ── Eyes-closed duration (needed here for alert_state HUD) ─
        # eye_closed_start is updated by update_perclos() above, so
        # this check is always fresh on every frame.
        eyes_closed_long = (
            eye_closed_start is not None and
            (now - eye_closed_start) >= EYE_CLOSED_SECS
        )

        # ── Alert state for HUD — head_down takes absolute priority ─
        # This runs in the CV thread so the screen updates the same
        # frame that head_down is detected — zero display lag.
        if head_down or eyes_closed_long:
            alert_state = "HIGH"
        elif fatigue >= FATIGUE_HIGH_MIN:
            alert_state = "HIGH"
        elif fatigue >= FATIGUE_MILD_MIN:
            alert_state = "MILD"
        else:
            alert_state = "NORMAL"

        # ── Update shared state ───────────────────────────────────
        with state_lock:
            shared_state.update({
                "fatigue"    : fatigue,
                "head_down"  : head_down,
                "ear"        : round(ear,   3),
                "mar"        : round(mar,   3),
                "pitch"      : round(pitch, 1),
                "yaw"        : round(yaw,   1),
                "blinks_pm"  : round(blinks_pm, 1),
                "perclos"    : round(perclos, 3),
                "yawns_5m"   : yawns_5m,
                "alert_state": alert_state,
            })

        # ── HUD ───────────────────────────────────────────────────
        with state_lock:
            hud_data = dict(shared_state)
        draw_hud(frame, hud_data)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alert()
    print("[INFO] Exiting.")

if __name__ == "__main__":
    main()