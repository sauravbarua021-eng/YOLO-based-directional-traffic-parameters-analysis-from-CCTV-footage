import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math
from collections import defaultdict

# ==============================
# SETTINGS
# ==============================
VIDEO_PATH = "cam1-20260105-163230.mkv"
MODEL_PATH = "yolo11n.pt"
CONF = 0.3

VEHICLES = {2, 3, 5, 7}  # COCO vehicle classes

# ==============================
# GLOBAL DRAW STATE
# ==============================
count_lines = []
calib_line = None
drawing = False
mode = "count"
temp_pt = None
pixels_per_meter = None

# ==============================
# MOUSE DRAW
# ==============================
def mouse_draw(event, x, y, flags, param):
    global drawing, temp_pt, count_lines, calib_line, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pt2 = (x, y)

        if mode == "count":
            count_lines.append((temp_pt, pt2))
            print("Count line:", temp_pt, pt2)

        elif mode == "calib":
            calib_line = (temp_pt, pt2)
            print("Calibration line:", calib_line)

# ==============================
# GEOMETRY
# ==============================
def line_length(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def direction(prev, curr):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    else:
        return "Down" if dy > 0 else "Up"

# ==============================
# LOAD
# ==============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (w,h))

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_draw)

print("\nINSTRUCTIONS:")
print("Draw COUNT lines → mouse drag")
print("Press c → calibration mode")
print("Press n → count mode")
print("Press ENTER → finish calibration")
print("Press q → start processing\n")

# ==============================
# DRAW SETUP
# ==============================
calib_ready = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()

    # draw count lines
    for ln in count_lines:
        cv2.line(disp, ln[0], ln[1], (0,255,255), 2)

    # draw calibration line
    if calib_line:
        cv2.line(disp, calib_line[0], calib_line[1], (0,0,255), 3)

    cv2.putText(disp, f"MODE: {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(disp, "ENTER = finish calibration", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Video", disp)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('c'):
        mode = "calib"

    elif key == ord('n'):
        mode = "count"

    elif key == 13:  # ENTER
        if calib_line is None:
            print("Draw calibration line first")
        else:
            calib_ready = True
            break

    elif key == ord('q'):
        break

# ==============================
# CALIBRATION INPUT
# ==============================
if calib_ready:
    meters = float(input("\nEnter real calibration length (meters): "))
    px = line_length(calib_line[0], calib_line[1])
    pixels_per_meter = px / meters
    print("Pixels per meter:", pixels_per_meter)
else:
    pixels_per_meter = None

# ==============================
# RESET VIDEO
# ==============================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ==============================
# TRACKING + FLOW + SPEED
# ==============================
track_last = {}
track_time = {}

line_counts = [0]*len(count_lines)
line_dir_counts = [defaultdict(int) for _ in count_lines]
counted = set()

records = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    t = frame_id / fps

    results = model.track(frame, persist=True, conf=CONF)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        for box, tid, cls in zip(boxes, ids, clss):
            if int(cls) not in VEHICLES:
                continue

            x1,y1,x2,y2 = map(int, box)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            center = (cx,cy)

            speed = 0
            prev = None

            if tid in track_last and pixels_per_meter:
                prev = track_last[tid]
                dt = t - track_time[tid]

                pix = math.hypot(center[0]-prev[0], center[1]-prev[1])
                meters = pix / pixels_per_meter
                speed = (meters/dt)*3.6 if dt>0 else 0

            track_last[tid] = center
            track_time[tid] = t

            # FLOW COUNT
            if prev is not None:
                for i, ln in enumerate(count_lines):
                    if (tid,i) in counted:
                        continue
                    if intersect(prev, center, ln[0], ln[1]):
                        line_counts[i]+=1
                        counted.add((tid,i))

                        dirc = direction(prev, center)
                        line_dir_counts[i][dirc]+=1

                        records.append([
                            frame_id,
                            round(t,2),
                            tid,
                            i,
                            dirc,
                            round(speed,2)
                        ])

            # draw box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame, center, 3, (0,0,255), -1)
            cv2.putText(frame,f"{int(speed)} km/h",(cx,cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # draw lines + counts
    for i,ln in enumerate(count_lines):
        cv2.line(frame, ln[0], ln[1], (0,255,255), 2)
        txt = f"L{i}:{line_counts[i]}"
        cv2.putText(frame, txt, ln[0],
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    out.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1)==27:
        break

# ==============================
# SAVE EXCEL
# ==============================
df = pd.DataFrame(records,
    columns=["frame","time_s","id","line","direction","speed_kmh"])

df.to_excel("traffic_flow_speed.xlsx", index=False)

print("\nFINAL COUNTS")
for i,c in enumerate(line_counts):
    print(f"Line {i}: {c}")
    print("Directional:", dict(line_dir_counts[i]))

cap.release()
out.release()
cv2.destroyAllWindows()