import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import filedialog, ttk, Text, Message, Toplevel
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import cv2


main = tkinter.Tk()
main.title("LUNG CANCER DETECTION USING CNN")  # designing main screen
main.geometry("1300x1200")

global test_filename
test_filename = None
classes = ['adenocarcinoma', 'benign', 'squamous']

def test_upload():
    global test_filename
    test_filename = askopenfilename(initialdir="data/test")
    if test_filename:
        # Create a popup window
        popup = Toplevel(main)
        popup.title("Upload Success")
        popup.geometry("300x200")
        Label(popup, text="Image uploaded successfully!", font=('times', 12, 'bold')).pack(pady=20)
        
        # Automatically close the popup after 2 seconds
        popup.after(2000, popup.destroy)

def predict():
    global test_filename
    if not test_filename:
        text.insert(END, "Please upload an image first.\n")
        text.see(END)
        return

    model = load_model(r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\model_v2\continued_trained_model.h5")
    lab = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma']
    img = load_img(test_filename, target_size=(125, 125))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, [0])
    
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = lab[y]

    img = cv2.imread(test_filename, 1)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, res, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow('Disease Identified as : ' + res, img)
    cv2.waitKey(0)

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='LUNG CANCER DETECTION USING CNN')
title.config(bg='Lavender', fg='black', font=font, height=3, width=120)
title.place(x=0, y=3)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 14, 'bold')
imageButton = Button(main, text="Upload Test Image", command=test_upload, width=35, height=2, background='lavender')
imageButton.place(x=460, y=550)
imageButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer", command=predict, width=35, height=2, background='lavender')
predictButton.place(x=245, y=660)
predictButton.config(font=font1)

closeButton = Button(main, text="Close", command=close, width=35, height=2, background='lavender')
closeButton.place(x=700, y=660)
closeButton.config(font=font1)

abstract = Message(main, font="times 16", text=(
    "Abstract:\nLung Cancer is one of the leading life-taking cancers worldwide. "
    "Early detection and treatment are crucial for patient recovery. Medical professionals "
    "use histopathological images of biopsied tissue from potentially infected areas of lungs for diagnosis. "
    "Most of the time, the diagnosis regarding the types of lung cancer are error-prone and time-consuming. "
    "Convolutional Neural networks can identify and classify lung cancer types with greater accuracy in a shorter period, "
    "which is crucial for determining patients' right treatment procedure and their survival rate. "
    "Benign tissue, Adenocarcinoma, and squamous cell carcinoma are considered in this research work.\n\n"
    "Keywords:\nConvolutional Neural Network (CNN), Deep Learning, Lung Cancer, Histopathological Image"
), width=1000, justify="left", foreground='black', background='white')
abstract.pack(padx='20px')
abstract.place(x=150, y=190)

main.config(bg='teal')
main.mainloop()
