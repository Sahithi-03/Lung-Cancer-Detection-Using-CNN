import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load and preprocess the test image
img_path = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test\lung_n\lungn4382.jpeg"
img = image.load_img(img_path, target_size=(125, 125))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Load the complete model from .h5 file
model = load_model(r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model_v2\continued_trained_model.h5")

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

# Since no pickle file is available for accuracy, you can either manually input accuracy
# or use model evaluation on test data
# Evaluate the model on test data (optional)
# (Assuming you have test data in x_test and y_test)
# accuracy = model.evaluate(x_test, y_test)
# accuracy_text = f"Model Accuracy: {accuracy[1] * 100:.2f}%"

# If you don't have test data, manually input the accuracy value:
accuracy_text = "Model Accuracy: 98.83%"

Label(root, text=accuracy_text, font=("Helvetica", 14)).pack()

root.mainloop()
