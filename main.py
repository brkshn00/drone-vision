import cv2
from ultralytics import YOLO

# Load the YOLOv10n model
# The model will be downloaded automatically if it doesn't exist
model = YOLO("yolov10n.pt")

# Initialize the webcam
# 0 is usually the default camera. If you have multiple cameras, try 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run object detection on the frame
    # stream=True is recommended for video sources
    results = model(frame, stream=True)

    # Visualize the results on the frame
    # Since we are using stream=True, results is a generator
    for result in results:
        annotated_frame = result.plot()

        # Display the annotated frame
        cv2.imshow("YOLOv10n Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
