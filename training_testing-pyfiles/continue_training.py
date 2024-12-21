import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import pickle

# Paths for the dataset
train_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Train"
test_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test"
val_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Val"
batch_size = 32
target_size = (125, 125)

# Data generators
train = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.40)

test = ImageDataGenerator(rescale=1/255.0)

train_generator = train.flow_from_directory(
    directory=train_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training')

valid_generator = train.flow_from_directory(
    directory=val_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)

test_generator = test.flow_from_directory(
    directory=test_data,
    target_size=target_size,
    batch_size=1)

# Load the previously trained model
model = load_model(r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model\ResNet50(200ep)\model.h5")

# Recompile the model with a new optimizer
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

# Define the early stopping callback for continued training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Continued training for additional 20 epochs
hist_continued = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,  # Additional epochs
    callbacks=[early_stopping])

# After continued training, you can access the final training and validation accuracy
final_train_accuracy = hist_continued.history['accuracy'][-1]
final_val_accuracy = hist_continued.history['val_accuracy'][-1]

# Print the final accuracy after continued training
print(f"Final training accuracy (after continued training): {final_train_accuracy * 100:.2f}%")
print(f"Final validation accuracy (after continued training): {final_val_accuracy * 100:.2f}%")

# Saving the continued model and training history
model.save("model/ResNet50(200ep)/model_continued.h5")

model_json = model.to_json()
with open("model/ResNet50(200ep)/model_continued.json", "w") as json_file:
    json_file.write(model_json)

with open("model/ResNet50(200ep)/history_continued.pckl", "wb") as f:
    pickle.dump(hist_continued.history, f)
