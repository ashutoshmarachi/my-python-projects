#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('/Users/azrael/haarcascade_frontalface_default.xml')

# Load an image
img = cv2.imread('20220719_164517-3.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with the rectangles
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

