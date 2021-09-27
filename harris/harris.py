import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, floor, cos, sin, sqrt, zeros, argmin
import pprint


def window(image, i, j, dx, dy):
    # creates window for each pixel and assigns a matrix
    windowed = []
    matrix = [dx[i][j] ** 2, dx[i][j] * dy[i][j], dx[i][j] * dy[i][j], dy[i][j] ** 2]
    for i in range(0, 5):
        row_windowed = []
        for j in range(0, 5):
            row_windowed.append(matrix)
        windowed.append(row_windowed)
    # sum of the matrices
    window_array = np.array(windowed.pop())
    window_add = window_array.sum(axis=0)
    return window_add


def det_traza(matrix, k):
    det = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2])
    traza = matrix[0] + matrix[-1]
    return det - k * (traza ** 2)


imagen = cv2.imread("figura1.png", 0)
dx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0)
dy = cv2.Sobel(imagen, cv2.CV_64F, 0, 1)

plt.figure(figsize=(12, 8), dpi=100)
plt.subplot(1, 3, 1), plt.imshow(imagen, cmap="gray")
plt.subplot(1, 3, 2), plt.imshow(dx, cmap="gray")
plt.subplot(1, 3, 3), plt.imshow(dy, cmap="gray")
plt.show()

imagen_temp = imagen.copy()
for i, element in enumerate(imagen_temp):
    for j, element2 in enumerate(imagen_temp[i]):
        M = window(imagen_temp, i, j, dx, dy)
        imagen_temp[i][j] = det_traza(M, 0.04)

plt.figure(figsize=(12, 8), dpi=100)
plt.subplot(1, 3, 1).set_title("Original"), plt.imshow(imagen, cmap="gray")
plt.subplot(1, 3, 2).set_title("Det - traza"), plt.imshow(imagen_temp, cmap="gray")
plt.show()
