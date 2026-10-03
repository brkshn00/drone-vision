import cv2
import os
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client
# Using the credentials provided in your guide
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="hIUD1lwwM8bnEKNO7pfy"
)

# Workflow configuration
WORKSPACE_NAME = "target-xue2d"
WORKFLOW_ID = "find-flowers"

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Roboflow Cloud Inference...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Save frame temporarily (Roboflow SDK often works best with file paths or base64)
    # Optimization: We could do in-memory base64, but disk is safer for the SDK's basic usage
    temp_path = 'temp_frame.jpg'
    cv2.imwrite(temp_path, frame)

    try:
        # Run inference
        # The return structure depends on the specific workflow. 
        # Based on your guide, it returns 'outputs' -> list -> 'predictions' -> 'predictions'
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path}
        )
        
        # DEBUG: Print the result to see if we are getting predictions
        print(f"API Response: {result}")
        
        # Parse results and draw boxes
        # We need to navigate the JSON response strictly based on the provided example
        # Example: result['outputs'][0]['predictions']['predictions']
        
        predictions = []
        if 'outputs' in result and len(result['outputs']) > 0:
            output_item = result['outputs'][0]
            # print(f"DEBUG: output_item keys: {output_item.keys()}")
            
            if 'predictions' in output_item:
                 pred_data = output_item['predictions']
                 # print(f"DEBUG: pred_data type: {type(pred_data)}")
                 
                 if isinstance(pred_data, dict) and 'predictions' in pred_data:
                     predictions = pred_data['predictions']
                 elif isinstance(pred_data, list):
                     predictions = pred_data
                 else:
                     print("DEBUG: Could not find 'predictions' list in expected location.")

        print(f"DEBUG: Found {len(predictions)} predictions.")
        
        for pred in predictions:
            # Roboflow usually returns center_x, center_y, width, height
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            class_name = pred['class']
            confidence = pred['confidence']

            # Calculate top-left corner
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)

    except Exception as e:
        print(f"Inference error: {e}")

    # Display the frame
    cv2.imshow('Roboflow Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if os.path.exists('temp_frame.jpg'):
    os.remove('temp_frame.jpg')
