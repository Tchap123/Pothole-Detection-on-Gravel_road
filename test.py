import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Parameters
test_data_path = "./ImgData/Test"  # Path to the directory with test images
model_path = "./Trained.h5"  # Path to the trained model
target_size = (150, 150)  # Size to resize images

model = load_model(model_path)

# Function to preprocess images
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at path '{image_path}'")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size) / 255.0
    return image

# Load and preprocess test images
test_images = []
test_image_paths = []
for filename in os.listdir(test_data_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions and error handling
        image_path = os.path.join(test_data_path, filename)
        image = preprocess_image(image_path, target_size)
        if image is not None:
            test_images.append(image)
            test_image_paths.append(image_path)

test_images = np.array(test_images)

# Make predictions
predictions = model.predict(test_images)

# Display results
for i, prediction in enumerate(predictions):
    plt.imshow(test_images[i])
    plt.title(f"Prediction: {'Pothole' if prediction >= 0.5 else 'No Pothole'}")
    plt.show()
    print(f"Image: {test_image_paths[i]} - Prediction: {'Pothole' if prediction >= 0.5 else 'No Pothole'}")
