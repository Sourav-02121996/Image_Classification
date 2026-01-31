from ast import increment_lineno

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

 plt.imshow(img)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

face_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/'
                                     'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/'
                                    'haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)
