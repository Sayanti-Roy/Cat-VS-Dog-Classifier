import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅ fixed import
import zipfile
import random
import os  # ✅ missing import
from PIL import Image

# Getting the Dataset
dataset_path = 'cats_and_dogs_filtered (1).zip'
extract_path = 'cat and dogs'

# Extract the dataset
if not os.path.exists(extract_path):
    print("Extracting dataset .....")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction Completed")

# Paths
train_dir = os.path.join(extract_path, 'cats_and_dogs_filtered', 'train')
val_dir = os.path.join(extract_path, 'cats_and_dogs_filtered', 'validation')

# Load some sample images
cat_dir = os.path.join(train_dir, 'cats')
image_files = os.listdir(cat_dir)

print("Available Images:", image_files[:5])  # show first 5 file names

# Show one image
image_path = os.path.join(cat_dir, image_files[2])
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

# Preprocessing images
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Building CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator
)

# Save model
model.save("cat_dog_classifier.h5")
print("Model saved successfully")

# Evaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Validation accuracy: {accuracy*100:.2f}%')