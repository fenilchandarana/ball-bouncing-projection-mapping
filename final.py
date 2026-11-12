import json
import os
import random
import threading

import cv2
import numpy as np
import pygame
import pymunk

# ---------------- settings ----------------
DISPLAY_INDEX = 1      # 1 = secondary screen/projector, 0 = main screen
BALL_RADIUS = 20
SPAWN_X = 0.8          # 0 = left edge, 0.5 = center, 1 = right edge
SPAWN_PER_SEC = 1      # balls created per second
MAX_BALLS = 200
GRAVITY = 900          # pixels/sec^2, positive = down on screen
BOUNCE = 0.75           # 0 = no bounce, 1 = perfect bounce
HOLD_FRAMES = 15      # keep the last note position for this many frames if detection drops
CALIB_MARGIN = 250    # distance of markers 1-4 from the screen edges (pixels), bigger = closer together

CAMERA_INDEX = 1
FOCUS_VALUE = 240
MIN_AREA = 100         # ignore camera blobs smaller than this (pixels)
CALIB_FILE = "calibration.json"
DRAW_OUTLINE = False    # thin blue outline on each detected note (set False if it disturbs detection)

FONT = cv2.FONT_HERSHEY_SIMPLEX
HSV_DEFAULTS = {"H min": 20, "H max": 35, "S min": 60, "S max": 255, "V min": 92, "V max": 255}
HSV_LIMITS = {"H min": 179, "H max": 179, "S min": 255, "S max": 255, "V min": 255, "V max": 255}

# ---------------- pygame + physics ----------------
pygame.init()
displays = pygame.display.get_num_displays()
idx = DISPLAY_INDEX if DISPLAY_INDEX < displays else 0
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=idx)
W, H = screen.get_size()
pygame.mouse.set_visible(True)
clock = pygame.time.Clock()
big_font = pygame.font.Font(None, 120)

space = pymunk.Space()
space.gravity = (0, GRAVITY)
balls = []
note_shapes = []
note_quads = []


def spawn_ball():
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, BALL_RADIUS))
    body.position = (int(W * SPAWN_X), -BALL_RADIUS)
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.elasticity = BOUNCE
    shape.friction = 0.4
    space.add(body, shape)
    balls.append(shape)


def update_colliders(quads):
    """Replace last frame's note colliders with the current ones."""
    for s in note_shapes:
        space.remove(s)
    note_shapes.clear()
    for q in quads:
        try:
            s = pymunk.Poly(space.static_body, q)
        except Exception:
            continue
        s.elasticity = BOUNCE
        s.friction = 0.4
        space.add(s)
        note_shapes.append(s)


# ---------------- camera (reads frames in a background thread) ----------------
class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit("Camera not found. Try another CAMERA_INDEX.")
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def calib_spots():
    m = CALIB_MARGIN
    return [(m, m), (W - m, m), (W - m, H - m), (m, H - m)]


# ---------------- calibration ----------------
def draw_calibration_screen():
    screen.fill((255, 255, 255))
    for i, (x, y) in enumerate(calib_spots()):
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 8)
        t = big_font.render(str(i + 1), True, (0, 0, 0))
        screen.blit(t, t.get_rect(center=(x, y - 60)))
    pygame.display.flip()


def calibrate(cam):
    """Click the 4 corners of the projected screen in the camera view (1 to 4 as numbered)."""
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)
    draw_calibration_screen()

    while True:
        pygame.event.pump()
        frame = cam.get()
        if frame is None:
            cv2.waitKey(30)
            continue
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (p[0] + 8, p[1] - 8), FONT, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Click projected screen corners 1,2,3,4 as numbered", (10, 25), FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "ENTER = done   R = reset   ESC = cancel", (10, 50), FONT, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibration", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == 13 and len(pts) == 4:
            break
        elif k == 27:
            cv2.destroyWindow("calibration")
            return None

    cv2.destroyWindow("calibration")
    with open(CALIB_FILE, "w") as f:
        json.dump(pts, f)
    return pts


def make_matrix(pts):
        dst = np.float32(calib_spots())
        return cv2.getPerspectiveTransform(np.float32(pts), dst)


# ---------------- detection ----------------
def detect_notes(frame, M):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s_min = cv2.getTrackbarPos("S min", "controls")
    v_min = cv2.getTrackbarPos("V min", "controls")
    lower = np.array([HSV_DEFAULTS["H min"], s_min, v_min])
    upper = np.array([HSV_DEFAULTS["H max"], HSV_DEFAULTS["S max"], HSV_DEFAULTS["V max"]])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    debug = frame.copy()
    for c in contours:
        if cv2.contourArea(c) < cv2.getTrackbarPos("Min area", "controls"):
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        cv2.drawContours(debug, [box.astype(int)], 0, (0, 255, 0), 2)
        quad = cv2.perspectiveTransform(box.reshape(1, 4, 2), M)[0]
        cx, cy = quad.mean(axis=0)
        if 0 <= cx <= W and 0 <= cy <= H:  # ignore notes outside the projected area
            quads.append([(float(x), float(y)) for x, y in quad])
    return quads, mask, debug


# ---------------- startup ----------------
cam = Camera(CAMERA_INDEX)

pts = None
if os.path.exists(CALIB_FILE):
    with open(CALIB_FILE) as f:
        pts = json.load(f)
if pts is None:
    pts = calibrate(cam)
if pts is None:
    cam.stop()
    pygame.quit()
    raise SystemExit("Calibration cancelled.")
M = make_matrix(pts)

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 500, 300)
cv2.createTrackbar("S min", "controls", HSV_DEFAULTS["S min"], 255, lambda x: None)
cv2.createTrackbar("V min", "controls", HSV_DEFAULTS["V min"], 255, lambda x: None)
cv2.createTrackbar("Ball size", "controls", BALL_RADIUS, 80, lambda x: None)
cv2.createTrackbar("Bounce %", "controls", int(BOUNCE * 100), 100, lambda x: None)
cv2.createTrackbar("Balls/sec x10", "controls", int(SPAWN_PER_SEC * 10), 100, lambda x: None)
cv2.createTrackbar("Fall X %", "controls", int(SPAWN_X * 100), 100, lambda x: None)
cv2.createTrackbar("Min area", "controls", MIN_AREA, 2000, lambda x: None)

print("Keys: ESC/Q = quit, K = recalibrate")

# ---------------- main loop ----------------
spawn_timer = 0.0
running = True
recalibrate = False 
missed = 0
while running:
    dt = min(clock.tick(60) / 1000.0, 1 / 30)

    BALL_RADIUS = max(5, cv2.getTrackbarPos("Ball size", "controls"))
    BOUNCE = cv2.getTrackbarPos("Bounce %", "controls") / 100
    SPAWN_PER_SEC = max(1, cv2.getTrackbarPos("Balls/sec x10", "controls")) / 10
    SPAWN_X = cv2.getTrackbarPos("Fall X %", "controls") / 100

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif e.key == pygame.K_k:
                recalibrate = True

        # camera: find notes and turn them into colliders
    frame = cam.get()
    if frame is not None:
        quads, mask, debug = detect_notes(frame, M)
        if quads:
            note_quads = quads
            missed = 0
        else:
            missed += 1
            if missed > HOLD_FRAMES:
                note_quads = []
        update_colliders(note_quads)
        cv2.imshow("camera + detection", debug)
        # cv2.imshow("mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        running = False
    elif key == ord("k"):
        recalibrate = True

    if recalibrate:
        recalibrate = False
        new_pts = calibrate(cam)
        if new_pts:
            pts = new_pts
            M = make_matrix(pts)

    # balls
    spawn_timer += dt
    while spawn_timer > 1 / SPAWN_PER_SEC:
        spawn_timer -= 1 / SPAWN_PER_SEC
        if len(balls) < MAX_BALLS:
            spawn_ball()

    space.step(dt)

    for s in balls[:]:
        if s.body.position.y > H + 50:
            space.remove(s, s.body)
            balls.remove(s)

    # draw
    screen.fill((0, 0, 0))
    for s in balls:
        x, y = s.body.position
        pygame.draw.circle(screen, (220, 30, 30), (int(x), int(y)), int(s.radius))
    if DRAW_OUTLINE:
        for q in note_quads:
            pygame.draw.polygon(screen, (0, 120, 255), q, 2)
    pygame.display.flip()

cam.stop()
cv2.destroyAllWindows()
pygame.quit()