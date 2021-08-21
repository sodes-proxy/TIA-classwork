import cv2
import numpy as np
from matplotlib import pyplot as plt
LOWER_TRESHOLD=140
UPPER_TRESHOLD=200
octagono= cv2.imread("octagono.png", 0) #reads image in grayscale
# blurring the image
octagono= cv2.GaussianBlur(octagono, ksize=(5,5), sigmaX=1)
# derivatives
dx= cv2.Sobel(octagono,cv2.CV_64F, 1,0)
dy= cv2.Sobel(octagono,cv2.CV_64F, 0,1)
# derivatives and normal
'''plt.figure(figsize=(12,8), dpi=100)
plt.subplot(1,3,1),plt.imshow(octagono, cmap="gray")
plt.subplot(1,3,2), plt.imshow(dx, cmap="gray")
plt.subplot(1,3,3), plt.imshow(dy, cmap="gray")
plt.show()'''
#Gradient
'''
G= np.sqrt(dx**2+dy**2)
plt.imshow(G, cmap="gray")
plt.show()'''
#canny edge detectetor 
bordes=cv2.Canny(octagono, LOWER_TRESHOLD, UPPER_TRESHOLD, apertureSize=3)
plt.imshow(bordes, cmap="gray")
plt.show()