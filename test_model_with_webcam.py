from ultralytics import YOLO
import cv2
import os  # Import os module to check file existence

# Path to the model file

# Path to the model file
model_path = r'C:\Users\hassa\OneDrive\Desktop\project\corn_leaf_model\corn_Leaf_model.pt'

# Verify the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path.")
    exit(1)

print("Model file found. Loading the model...")

# Load the YOLO model
model = YOLO(model_path)

# Open the webcam (default is device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit(1)

print("Webcam accessed. Starting real-time detection...")

# Get webcam properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Webcam properties: {frame_width}x{frame_height}, {fps} FPS")

# Real-time detection loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run the YOLO model on the current frame
        results = model(frame)

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('Real-Time Detection', annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting real-time detection...")
            break
except Exception as e:
    print(f"Error occurred: {e}")

# Release resources
cap.release()
cv2.destroyAllWindows()
