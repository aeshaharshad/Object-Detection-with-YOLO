import cv2
from ultralytics import YOLO
import numpy as np

model=YOLO('yolov8n-seg.pt')
cap=cv2.VideoCapture('street.mp4')

def shrink(frame,w=640):
    h,wd=frame.shape[:2]
    if wd<=w: return frame,1.0
    s=w/wd
    return cv2.resize(frame,(w,int(round(h*s))),interpolation=cv2.INTER_AREA),s

while True:
    ret,frame=cap.read()
    if not ret: break
    sf,s=shrink(frame)
    results=model.track(source=sf,classes=[0],persist=True,verbose=False)
    out=frame.copy()
    for r in results:
        if r.masks is None or r.boxes is None or r.boxes.id is None: continue
        masks=r.masks.data.cpu().numpy(); boxes=r.boxes.xyxy.cpu().numpy(); ids=r.boxes.id.cpu().numpy()
        for i,mask in enumerate(masks):
            pid=ids[i]; x1,y1,x2,y2=boxes[i].astype(int)
            x1,x2=int(x1/s),int(x2/s); y1,y2=int(y1/s),int(y2/s)
            m=cv2.resize(mask.astype(np.uint8)*255,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
            c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out,c,-1,(0,255,0),2)
            cv2.putText(out,f'ID:{pid}',(x1,max(y1-10,0)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    cv2.imshow('Object Tracking with Segmentation',out)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release(); cv2.destroyAllWindows()
