import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Load the image
image = cv2.imread('image1.jpg')
if image is None:
    print("Error: Could not load image.jpg")
    exit()

# Resize the image
resized_img = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

# Save the resized image (optional)
cv2.imwrite("image_r.jpg", resized_img)

# Run object detection on the resized image
results = model(resized_img)

# Annotate and display the results
annotated_image = results[0].plot()
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()