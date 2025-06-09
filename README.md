# 🖼️ Image Similarity Search with MobileNetV2

This project implements an **image similarity search engine** using a pretrained **MobileNetV2** model for deep feature extraction and **cosine similarity** to find and display visually similar images from a user-provided dataset.

---

## 🚀 Features

- ✅ Uses pretrained MobileNetV2 from Keras (transfer learning)
- ✅ Extracts meaningful feature vectors from images
- ✅ Finds top-k visually similar images using cosine similarity
- ✅ Displays results in an easy-to-understand plot

---

## 📁 Project Structure

```
image-similarity-search/
├── main.ipynb # Main notebook with all code
├── requirements.txt # Dependencies list
├── README.md # This file
```

> ⚠️ You must provide your own image dataset folder. See instructions below.

---

## 📦 Installation

1. Clone this repository:

> ⚠️ You must provide your own image dataset folder. See instructions below.

---

## 📦 Installation

1. Clone this repository:

```bash
git clone https://github.com/souvikde1/image-similarity-search.git
cd image-similarity-search
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Launch the notebook:
```
jupyter notebook main.ipynb
```
📂 How to Provide Your Dataset
Create a folder (e.g., my_dataset/) and place your .jpg / .png images in it.

Example:
```
my_dataset/
├── img1.jpg
├── img2.jpg
├── img3.jpg
```
Then set the dataset folder path in the notebook:
```
dataset_folder = 'my_dataset/'
```
---
## 🔍 How It Works

>MobileNetV2 (with ImageNet weights) extracts feature vectors from all dataset images.
A user-specified query image is also encoded to a feature vector.
Cosine similarity is used to find the closest matches.
The top-k most similar images are displayed.
---
## 🧪 Output Example

>Replace with your own dataset for actual results.
Original Image:
Similar Images:
---
## ⚙️ Configurable Parameters
>You can customize:
```
dataset_folder = 'my_dataset/'
query_image_path = 'path/to/query.jpg'
top_k = 5
```
## 🧾 Dependencies
>Listed in requirements.txt
Install them with:
```
pip install -r requirements.txt
```
## ✍️ Author
Developed by Souvik De
If this project helped you, consider ⭐ starring the repo or sharing it with others!
