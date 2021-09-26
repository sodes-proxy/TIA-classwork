# author : Sebastián Ochoa Osuna and Marcelo Alvarez
# reference:  https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, floor, cos, sin, sqrt, zeros, argmin


def hough(imagen, cat, name):
    border_imagen = cv2.Canny(imagen, 100, 200)
    plt.imshow(border_imagen, cmap="gray")
    cv2.imwrite("..\\images\\" + name, border_imagen)
    plt.show()

    altura, anchura = border_imagen.shape
    diagonal = sqrt(altura ** 2 + anchura ** 2)
    votos = zeros((cat, cat), dtype=int)
    angulos = [x * pi / cat for x in range(cat)]
    rhos = [((x * 2 * diagonal) / cat) - diagonal for x in range(cat)]

    for y in range(altura):
        for x in range(anchura):
            if border_imagen[y, x]:
                punto = [y - (altura / 2), x - (anchura / 2)]
                for nangulo, angulo in enumerate(angulos):
                    rho = (punto[1] * cos(angulo)) + (punto[0] * sin(angulo))
                    n_rho = argmin(abs(rhos - rho))
                    votos[n_rho, nangulo] += 1
    plt.imshow(votos)
    plt.show()
    cv2.imwrite("..\\images\\votos_" + name, votos)
    return votos


cat = 100
imagen1 = cv2.imread("1triangulo.png", 0)
imagen2 = cv2.imread("2lineas.png", 0)
imagen3 = cv2.imread("entrada.jpeg", 0)
votos_triangulos = hough(imagen1, cat, "1triangulo.png")
votos_lineas = hough(imagen2, cat, "2lineas.png")
votos_entradas = hough(imagen3, cat, "entrada.jpeg")
