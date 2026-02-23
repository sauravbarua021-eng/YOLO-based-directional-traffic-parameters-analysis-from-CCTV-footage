import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time

# ==============================
# SETTINGS
# ==============================
VIDEO_PATH = "cam1-20260105-163230.mkv"
MODEL_PATH = "yolo11n.pt"
OUTPUT_VIDEO = "flow_output.mp4"
OUTPUT_EXCEL = "flow_counts.xlsx"

CONF = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ==============================
# GLOBALS
# ==============================
drawing = False
ix, iy = -1, -1
count_lines = []
line_counts = []
object_history = {}

# directional counts
line_counts_dir = []   # [[forward, reverse], ...]

# time series storage
records = []

# ==============================
# MOUSE DRAW
# ==============================
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, count_lines, line_counts, line_counts_dir

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        count_lines.append(((ix, iy), (x, y)))
        line_counts.append(0)
        line_counts_dir.append([0, 0])  # forward, reverse


# ==============================
# GEOMETRY
# ==============================
def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def direction(A, B, P):
    # positive = one direction, negative = opposite
    return (B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0])

# ==============================
# LOAD MODEL + VIDEO
# ==============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

ret, frame = cap.read()

cv2.namedWindow("Draw Lines")
cv2.setMouseCallback("Draw Lines", draw_line)

# ==============================
# STEP 1 — DRAW LINES
# ==============================
while True:
    temp = frame.copy()

    for line in count_lines:
        cv2.line(temp, line[0], line[1], (0,255,255), 2)

    cv2.putText(temp, "Draw lines. Press S to start",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Draw Lines", temp)
    key = cv2.waitKey(1)
    if key == ord("s"):
        break

cv2.destroyWindow("Draw Lines")

start_time = time.time()

# ==============================
# STEP 2 — COUNTING
# ==============================
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=CONF, classes=VEHICLE_CLASSES)[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1,y1,x2,y2 = map(int, box)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            center = (cx,cy)

            if obj_id in object_history:
                prev_center = object_history[obj_id]

                for i, line in enumerate(count_lines):
                    if intersect(prev_center, center, line[0], line[1]):
                        line_counts[i] += 1

                        # direction classification
                        d1 = direction(line[0], line[1], prev_center)
                        d2 = direction(line[0], line[1], center)

                        if d1 < 0 and d2 > 0:
                            line_counts_dir[i][0] += 1  # forward
                        else:
                            line_counts_dir[i][1] += 1  # reverse

            object_history[obj_id] = center

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,center,3,(0,0,255),-1)

    # draw lines + counts
    for i, line in enumerate(count_lines):
        cv2.line(frame, line[0], line[1], (0,255,255), 2)
        mid = ((line[0][0]+line[1][0])//2, (line[0][1]+line[1][1])//2)

        fwd = line_counts_dir[i][0]
        rev = line_counts_dir[i][1]

        cv2.putText(frame, f"L{i+1} F:{fwd} R:{rev}",
                    mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ===== record time step =====
    elapsed = time.time() - start_time
    row = {"time_s": elapsed}
    for i in range(len(count_lines)):
        row[f"L{i+1}_F"] = line_counts_dir[i][0]
        row[f"L{i+1}_R"] = line_counts_dir[i][1]
    records.append(row)

    out.write(frame)
    cv2.imshow("Flow Counting", frame)

    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# ==============================
# SAVE EXCEL
# ==============================
df = pd.DataFrame(records)
df.to_excel(OUTPUT_EXCEL, index=False)

print("Saved video:", OUTPUT_VIDEO)
print("Saved Excel:", OUTPUT_EXCEL)