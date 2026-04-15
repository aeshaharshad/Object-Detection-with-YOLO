import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture("street.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
id_map={}
nex_id=1

trail=defaultdict(lambda: deque(maxlen=30))
appear=defaultdict(int)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(640,480))
    res=model.track(frame,classes=[0],persist=True,verbose=False)
    annotated_frame=frame.copy()
    if res[0].boxes.id is not None:
        boxes=res[0].boxes.xyxy.numpy()
        ids=res[0].boxes.id.numpy()
        for box,oid in zip(boxes,ids):
            x1,y1,x2,y2=map(int,box)
            cx,cy=(x1+x2)//2,(y1+y2)//2

            appear[oid]+=1
            if appear[oid]>=5 and oid not in id_map:
                id_map[oid]=nex_id
                nex_id+=1
            if oid in id_map:
                sid=id_map[oid]
                trail[oid].append((cx,cy))
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(annotated_frame,f'ID: {sid}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                cv2.circle(annotated_frame,(cx,cy),5,(0,255,0),-1)
            
    cv2.imshow("Tracking", annotated_frame)
    out.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows() 