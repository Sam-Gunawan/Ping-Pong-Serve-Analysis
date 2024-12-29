import cv2
from ultralytics import YOLO

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # set width
cap.set(4, 720) # set height

# Load YOLOv11 model
model = YOLO("yolo11n.pt")

while True:
    is_live, frame = cap.read()

    if not is_live:
        break

    # Detect objects
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv11 Tracking", annotated_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release() # so it can be used by other applications
cv2.destroyAllWindows()
