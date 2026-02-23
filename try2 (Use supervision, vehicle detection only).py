from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

# Load pre-trained YOLO model
model = YOLO("yolo11n.pt")  # or "yolov8n.pt" — small & fast

# Video paths
input_video = "cam1-20260105-163230.mkv"
output_video = "annotated_traffic.mp4"

# Open video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Colors for each class (BGR)
colors = {
    2: (0, 255, 0),   # car → green
    3: (0, 0, 255),   # motorcycle → red
    5: (255, 0, 0)    # bus → blue
}

class_names = {2: "car", 3: "motorcycle", 5: "bus"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)[0]

    # Filter only car/bus/motorcycle
    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        if cls in [2, 3, 5]:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf)
            detections.append((xyxy, conf, cls))

    if detections:
        # Convert to supervision format for nice annotation
        xyxy = np.array([d[0] for d in detections])
        confidence = np.array([d[1] for d in detections])
        class_id = np.array([d[2] for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Annotate
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=sv_detections)
        labels = [f"{class_names.get(c, 'other')} {conf:.2f}" for conf, c in zip(confidence, class_id)]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=sv_detections, labels=labels)

    else:
        annotated_frame = frame

    out.write(annotated_frame)

cap.release()
out.release()
print("Annotated video saved:", output_video)