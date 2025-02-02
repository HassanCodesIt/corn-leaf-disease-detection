from ultralytics import YOLO
import cv2
import os

# Path to the model file
model_path = r'C:\Users\hassa\OneDrive\Desktop\project\corn_leaf_model\corn_Leaf_model.pt'

# Path to the test image
image_path = r'C:\Users\hassa\OneDrive\Desktop\project\brown_spot3.jpg'

# Verify the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path.")
    exit(1)

# Verify the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}. Please check the path.")
    exit(1)

print("Model and image file found. Loading the model...")

# Load the YOLO model
model = YOLO(model_path)

# Load the image
image = cv2.imread(image_path)

# Run the YOLO model on the image
results = model(image)

# Annotate the image
annotated_image = results[0].plot()

# Display the annotated image
cv2.imshow("YOLO Image Prediction", annotated_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the annotated image
output_path = r'C:\Users\hassa\OneDrive\Desktop\test_images\annotated_example.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"Annotated image saved to {output_path}")
