import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.models import Model

def get_encoder():
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return model

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = (img / 127.5) - 1
    return img

def get_feature_vector(model, img):
    img_batch = np.expand_dims(img, axis=0)
    features = model.predict(img_batch, verbose=0)
    return features.flatten()

def build_feature_database(model, folder_path):
    features = []
    image_paths = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img = preprocess_image(full_path)
            vector = get_feature_vector(model, img)
            features.append(vector)
            image_paths.append(full_path)

    return np.array(features), image_paths

def find_similar_images(query_vector, feature_db, image_paths, top_k):
    similarities = cosine_similarity([query_vector], feature_db)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [image_paths[i] for i in top_k_indices]

def show_images(image_list):
    plt.figure(figsize=(15, 5))
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(image_list), i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# --- USAGE ---

# Path to your dataset folder (user must create and fill this)
dataset_folder = 'my_dataset/'  # replace with your folder

# Path to your query image
query_image_path = 'query.jpg'  # replace with your image

# Load model and build feature DB
encoder = get_encoder()
feature_db, image_paths = build_feature_database(encoder, dataset_folder)

# Process query image
query_img = preprocess_image(query_image_path)
query_vector = get_feature_vector(encoder, query_img)

# Show original image
print("Original image: \n")
plt.figure(figsize=(3, 3))
img = cv2.imread(query_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()

# Show similar images
print("\nSimilar images: \n")
similar_images = find_similar_images(query_vector, feature_db, image_paths, top_k=5)
show_images(similar_images)
