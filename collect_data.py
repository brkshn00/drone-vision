import cv2
import os
import time

# Create directory for raw images
output_dir = "datasets/raw_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Starting data collection. Images will be saved to '{output_dir}'")
print("Press 's' to save an image.")
print("Press 'q' to quit.")

count = 0
# Check existing files to avoid overwriting
existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
if existing_files:
    # Find the highest number in existing files
    nums = []
    for f in existing_files:
        try:
            num = int(f.split('_')[1].split('.')[0])
            nums.append(num)
        except:
            pass
    if nums:
        count = max(nums) + 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image on 's' press
    if key == ord('s'):
        filename = f"{output_dir}/img_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
        # Visual feedback: flash screen or pause briefly
        cv2.imshow("Data Collection", 255 - frame) # Invert colors briefly
        cv2.waitKey(50)

    # Quit on 'q' press
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
