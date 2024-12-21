# Lung Cancer Detection Using ResNet50

This project implements lung cancer detection using a ResNet50 convolutional neural network. The dataset includes images of three classes: Adenocarcinoma, Benign, and Squamous Cell Carcinoma. 

A Tkinter-based interface allows users to detect lung cancer using the trained model.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Dataset](#dataset)
4. [Getting Started](#getting-started)
5. [Model Details](#model-details)
6. [Results](#results)
7. [How to Use the Application](#how-to-use-the-application)
8. [License](#license)

---

## **Project Overview**
This project aims to:
- Detect and classify lung cancer types: Adenocarcinoma, Benign, and Squamous Cell Carcinoma.
- Provide a graphical interface (GUI) using Tkinter to facilitate lung cancer detection.

Key features:
- Utilizes ResNet50 pre-trained model for image classification.
- Includes evaluation metrics for two versions of the model: initial and continued training.
- Supports further model training and testing with well-organized scripts.

---

## **File Structure**
```plaintext
├──training_testing-pyfiles/
│   ├── lung/
│   │   ├── Test/              # Test dataset
│   |   │     ├── lung_aca/       #adenocarcinoma
│   |   │     ├── lung_n/         #benign
│   |   │     ├── lung_scc/       #squamous cell carcinoma
│   │   ├── Train/             # Training dataset
│   |   │     ├── lung_aca/       #adenocarcinoma
│   |   │     ├── lung_n/         #benign
│   |   │     ├── lung_scc/       #squamous cell carcinoma
│   │   ├── Val/               # Validation dataset
│   |   │     ├── lung_aca/       #adenocarcinoma
│   |   │     ├── lung_n/         #benign
│   |   │     ├── lung_scc/       #squamous cell carcinoma
├── continue_training.py   # Script for additional training epochs
├── test_1.py              # First testing script
├── test_2.py              # Second testing script
├── test_models.py         # Script to test and evaluate models
├── train(ResNet50).py     # Script for training ResNet50 model
├── Model_V1_evaluation.txt# Evaluation results of initial model
├── Model_V2_evaluation.txt# Evaluation results of extended training model
├── README.md              # Project documentation
├── main.py                # Tkinter-based GUI for lung cancer detection
├── requirements.txt       # Python dependencies
```
---

## **Dataset**
The dataset used is a combination of lung and colon cancer datasets from [Kaggle](https://www.kaggle.com/code/mohamedbakrey/colon-cancer-classification-using-deep-learning/input). The dataset has already been separated and pre-processed into three categories:
1. Adenocarcinoma
2. Benign
3. Squamous Cell Carcinoma

The dataset is available within the repository.

---

## **Getting Started**
Follow these steps to set up and run the project:

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd training_testing-pyfiles
```

### **2. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Train the Model**
To train the ResNet50 model:
```bash
python train(ResNet50).py
```

To continue training the model for additional epochs:
```bash
python continue_training.py
```

### **4. Test the Model**
Run testing scripts to evaluate the trained model:
```bash
python test_models.py
```

### **5. Launch the GUI**
To use the graphical interface for lung cancer detection:
```bash
python main.py
```
---

## **Model Details**
### **ResNet50**
- Pre-trained on ImageNet.
- Fine-tuned for lung cancer classification.

### **Evaluation Metrics**
#### **Model V1** (Initial Training):
- **Accuracy**: 96.91%
- **Loss**: 0.0892

#### **Model V2** (Continued Training):
- **Accuracy**: 98.83%
- **Loss**: 0.0259

---

## **Results**
### **Model V1 (Initial Training):**
- **Confusion Matrix**:
  ```
  [[594,   7,  24],
   [  0, 625,   0],
   [ 27,   0, 598]]
  ```
- **Classification Report**:
  | Class                   | Precision | Recall | F1-Score |
  |-------------------------|-----------|--------|----------|
  | Adenocarcinoma          | 0.96      | 0.95   | 0.95     |
  | Benign                 | 0.99      | 1.00   | 0.99     |
  | Squamous Cell Carcinoma | 0.96      | 0.96   | 0.96     |

### **Model V2 (Extended Training):**
- **Confusion Matrix**:
  ```
  [[611,   0,  14],
   [  0, 625,   0],
   [  8,   0, 617]]
  ```
- **Classification Report**:
  | Class                   | Precision | Recall | F1-Score |
  |-------------------------|-----------|--------|----------|
  | Adenocarcinoma          | 0.99      | 0.98   | 0.98     |
  | Benign                 | 1.00      | 1.00   | 1.00     |
  | Squamous Cell Carcinoma | 0.98      | 0.99   | 0.98     |

---

## **How to Use the Application**
1. Launch `main.py` to open the GUI.
2. Upload an image of the lung cancer sample.
3. The application will classify the image into one of the three categories:
   - Adenocarcinoma
   - Benign
   - Squamous Cell Carcinoma
4. The GUI displays the prediction of the type of lung cancer.

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

