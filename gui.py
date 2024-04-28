import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import base64
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load pre-computed descriptors
with open('saved_dictionary.pkl', 'rb') as f:
    dl_descriptors = pickle.load(f)

def find_similar_images(query_image_path, dictionary=dl_descriptors, dir_path='./archive/dataset'):
    similarity_scores = []
    query_features = dictionary['./dataset/'+query_image_path]
    for key in dictionary:
        similarity = cosine_similarity([query_features], [dictionary[key]])[0][0]
        similarity_scores.append((key, similarity))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores[:5]

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize the image to fit in the GUI
        img = ImageTk.PhotoImage(img)
        query_label.config(image=img)
        query_label.image = img  # Keep reference to the image to prevent garbage collection
        print("HEY")
        similar_images = find_similar_images(file_path.split(sep="/")[-1])
        # Display similar images
        print(similar_images)
        display_similar_images(similar_images)

def display_similar_images(similar_images):
    for i, (image_path, similarity_score) in enumerate(similar_images[1:]):
        img = Image.open('./archive/dataset/'+image_path.split(sep = '/')[-1])
        img.thumbnail((100, 100))  # Resize the image to fit in the GUI
        img = ImageTk.PhotoImage(img)
        similar_label = tk.Label(root, image=img)
        similar_label.grid(row=1, column=i)
        similar_label.image = img 
# Create the main application window
root = tk.Tk()
root.title("Image Similarity Search")

# Create a label to display the selected image
# Create a label to display the selected image
query_label = tk.Label(root)
query_label.grid(row=0, column=0, padx=10, pady=10)

# Create a button to open the file dialog
button = tk.Button(root, text="Select Image", command=open_file)
button.grid(row=0, column=1, padx=10, pady=10)

# Run the application
root.mainloop()
