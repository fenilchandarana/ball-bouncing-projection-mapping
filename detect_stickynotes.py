import cv2
import numpy as np

CAMERA_INDEX = 1   # change to 1, 2 if the wrong camera opens
MIN_AREA = 400     # ignore blobs smaller than this (pixels)
FOCUS_VALUE = 240   # manual focus, usually 0-255, try different values

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Camera not found. Try another CAMERA_INDEX.")

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # turn autofocus off
cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)  # manual focus

cv2.namedWindow("controls")
# starting guess for yellow (OpenCV hue range is 0-179)
defaults = {"H min": 20, "H max": 35, "S min": 100, "S max": 255, "V min": 100, "V max": 255}
limits = {"H min": 179, "H max": 179, "S min": 255, "S max": 255, "V min": 255, "V max": 255}
for name, val in defaults.items():
    cv2.createTrackbar(name, "controls", val, limits[name], lambda x: None)

print("Keys: P = print values, Q = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    v = {n: cv2.getTrackbarPos(n, "controls") for n in defaults}
    lower = np.array([v["H min"], v["S min"], v["V min"]])
    upper = np.array([v["H max"], v["S max"], v["V max"]])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)

    cv2.imshow("camera + detection", out)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        print("lower =", lower.tolist(), " upper =", upper.tolist())
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()