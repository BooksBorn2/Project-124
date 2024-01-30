import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if not os.environ.get("PYTHONHTTPSVERIFY",'') and getattr(ssl, "_create_unverified_context", None):
    ssl._create_default_https_context = ssl._create_unverified_context



X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.1

cnf = LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scale,y_train)
y_pred = cnf.predict(X_test_scale)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
cap = cv2.VideoCapture(0)

while (True):
    try: 
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GREY)
        height, width = grey.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 +56), int(height/2 +56))
        cv2.rectangle(grey, upper_left, bottom_right, (0,255,0), 2)
        roi = grey[upper_left[1]: bottom_right[1],upper_left[0]: bottom_right[0]]
        im_pim = Image.fromarray(roi)
        Image_bw = im_pil.convert("L")
        Image_bw_resize = Image_bw.resize((28, 28),Image.ANTIALIAS)


