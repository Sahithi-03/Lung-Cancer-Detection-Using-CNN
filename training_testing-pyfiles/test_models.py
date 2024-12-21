import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define the test dataset path and image size
test_data_dir = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test"
img_size = (125, 125)
batch_size = 32

# Class labels for interpretation
class_labels = {0: "Adenocarcinoma", 1: "Benign", 2: "Squamous Cell Carcinoma"}

# Load the test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Define paths to both models
model_paths = {
    "Model_V1": r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model\ResNet50(200ep)\model.h5",
    "Model_V2": r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model_v2\continued_trained_model.h5"
}

# Evaluate each model on the test set and save results
for model_name, model_path in model_paths.items():
    # Load the model directly from .h5 file
    model = load_model(model_path)

    # Compile the model for evaluation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Predictions and true labels
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Classification Report
    report = classification_report(true_classes, predicted_classes, target_names=list(class_labels.values()))
    
    # Model evaluation metrics
    loss, accuracy = model.evaluate(test_generator, verbose=0)

    # Save results to a file
    output_file = f"{model_name}_evaluation.txt"
    with open(output_file, "w") as f:
        f.write(f"=== Evaluation Metrics for {model_name} ===\n\n")
        
        # Write Confusion Matrix
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm, separator=", ") + "\n\n")
        
        # Write Classification Report
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        
        # Write Accuracy and Loss
        f.write(f"Model Loss: {loss:.4f}\n")
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")

    print(f"Results saved to {output_file}")

