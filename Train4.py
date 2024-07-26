import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

data_path = "./ImgData"
target_size = (150,150)  
batch_size = 4 


def load_pothole_data(data_path, csv_file, target_size):
    images = []
    labels = []
    with open(data_path + "/" + csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_path = row['image']
            xmin, ymin, xmax, ymax = map(float, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = 1 if row['label'] == 'Pothole' else 0
            labels.append(label)
            image = cv2.imread(data_path + "/Potholes/" + image_path)
            if image is None:
                print(f"Error: Unable to read image at path '{data_path}/{image_path}'")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size) / 255.0
            images.append(image)
    return np.array(images), np.array(labels)


def load_no_pothole_data(data_path, target_size):
  images = []
  for filename in os.listdir(data_path + "/No_Potholes/"):
    image = cv2.imread(data_path + "/No_Potholes/" + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, target_size) / 255.0
    images.append(image)
  labels = np.array([0] * len(images))  
  return np.array(images), labels

#error handling for load file
try:
  pothole_images, pothole_labels = load_pothole_data(data_path, "Potholes/Potholes.csv", target_size)
except FileNotFoundError:
  print("Error: Pothole CSV file not found. Please check the path.")
  exit()

# Load no pothole data
no_pothole_images, no_pothole_labels = load_no_pothole_data(data_path, target_size)

# Combine data for all separate files
all_images = np.concatenate((pothole_images, no_pothole_images))
all_labels = np.concatenate((pothole_labels, no_pothole_labels))

# Data augmentation
train_datagen = ImageDataGenerator(
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  rotation_range=20
)

# Create the CNN model from scratch
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (3,)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (pothole vs no pothole)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with data augmentation
train_generator = train_datagen.flow(all_images, all_labels, batch_size=batch_size)

# Train the model
history = model.fit(train_generator, epochs=500, steps_per_epoch=len(all_images) // batch_size)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(all_images, all_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Train the model with data augmentation
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20
)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)

# Define validation data generator
val_datagen = ImageDataGenerator()  # No augmentation for validation data
val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // batch_size,
    epochs=10, #adjust the epochs according to your choice (optimal would be 10 to avoid overfitting)
    validation_data=val_generator,
    validation_steps=len(val_images) // batch_size
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(val_images, val_labels)
print("Validation Loss:", test_loss)
print("Validation Accuracy:", test_accuracy)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

#save the model
model.save(r"./Trained.h5")