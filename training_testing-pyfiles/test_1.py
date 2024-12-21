import tensorflow as tf
import numpy as np
import pickle
import tkinter as tk
from tkinter import Label
from PIL import ImageTk, Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# Load and preprocess the test image
img_path = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test\lung_n\lungn4382.jpeg"
img = image.load_img(img_path, target_size=(125, 125))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Load the model architecture and weights
with open(r'C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model\ResNet50(200ep)\model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights(r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model\ResNet50(200ep)\model.h5")

# Load model training history (for accuracy data)
with open(r'C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model\ResNet50(200ep)\history.pckl', 'rb') as f:
    data = pickle.load(f)

# Prediction
predicts = model.predict(x)
cls = np.argmax(predicts)

# Cancer class labels (customize as per your classes)
class_labels = {0: "Adenocarcinoma", 1: "Benign", 2: "Squamous Cell Carcinoma"}

# Get the label of the predicted class
predicted_label = class_labels.get(cls, "Unknown")

# Tkinter window setup
root = tk.Tk()
root.title("Lung Cancer Detection Result")

# Display the test image
img_display = Image.open(img_path)
img_display = img_display.resize((250, 250))  # Resize to fit window
img_tk = ImageTk.PhotoImage(img_display)
img_label = Label(root, image=img_tk)
img_label.pack()

# Display the prediction result
prediction_text = f"Prediction: {predicted_label}"
Label(root, text=prediction_text, font=("Helvetica", 16)).pack()

# Display model accuracy (last recorded accuracy)
accuracy_text = f"Model Accuracy: 96.91%"
Label(root, text=accuracy_text, font=("Helvetica", 14)).pack()

root.mainloop()
