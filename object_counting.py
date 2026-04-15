import cv2
from ultralytics import YOLO
import numpy 

model = YOLO('yolov8m.pt')
cap=cv2.VideoCapture("street.mp4")
unique_ids = set()

while True:
    ret, frame = cap.read()
    results = model.track(frame,classes=[0],persist=True,verbose=False)
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids= results[0].boxes.id.numpy()
        for oid in ids:
            unique_ids.add(oid)

    count = len(unique_ids)
    cv2.putText(annotated_frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    resized_frame = cv2.resize(annotated_frame, (800, 600))
    cv2.imshow("Annotated video", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
