
# data_loading.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {'normal': 0, 'tuberculosis': 1}
    
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            for filename in os.listdir(label_folder_path):
                img_path = os.path.join(label_folder_path, filename)
                if os.path.isdir(img_path):
                    continue  # Skip directories
                try:
                    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label_map[label_folder.lower()])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def normalize_images(images):
    # Normalize the images to the range [0, 1]
    images = images / 255.0
    return images

def preprocess_data(images, labels):
    # Normalize images
    images = normalize_images(images)
    # Flatten the images for CNN
    images = images.reshape(len(images), IMG_HEIGHT, IMG_WIDTH, 3)
    return images, labels

def split_data(images, labels):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    return X_train, X_val, y_train, y_val
