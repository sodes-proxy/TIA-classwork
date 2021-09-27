# author : Sebastián Ochoa Osuna and Marcelo Alvarez
# reference:  https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, floor, cos, sin, sqrt, zeros, argmin


def hough(imagen, cat):
    border_imagen = cv2.Canny(imagen, 100, 200)
    plt.imshow(border_imagen, cmap="gray")
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
    return votos


def bordesHough(imagen, votos, cat, umbral, lineSize):
    altura, anchura = imagen.shape
    diagonal = sqrt(altura ** 2 + anchura ** 2)
    votos_temp = votos.copy()
    angulos = [x * pi / cat for x in range(cat)]
    rhos = [((x * 2 * diagonal) / cat) - diagonal for x in range(cat)]

    ## MÁXIMOS LOCALES DE LOS VECINOS
    maximos = np.zeros([anchura, altura], dtype=float)
    for i, element in enumerate(votos_temp):
        for j, element2 in enumerate(votos_temp[i]):
            try:
                if votos_temp[i][j] > votos_temp[i - 1][j + 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass
            try:
                if votos_temp[i][j] > votos_temp[i][j + 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass
            try:
                if votos_temp[i][j] > votos_temp[i + 1][j + 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass

            try:
                if votos_temp[i][j] > votos_temp[i - 1][j]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass
            try:
                if votos_temp[i][j] > votos_temp[i + 1][j]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass

            try:
                if votos_temp[i][j] > votos_temp[i - 1][j - 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass
            try:
                if votos_temp[i][j] > votos_temp[i][j - 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass
            try:
                if votos_temp[i][j] > votos_temp[i + 1][j - 1]:
                    maximos[i][j] = votos_temp[i][j]
            except:
                pass

    # FILTRAR LOS MÁXIMOS CON UN UMBRAL
    for i, element in enumerate(maximos):
        for j, element in enumerate(maximos[i]):
            if maximos[i][j] < umbral:
                maximos[i][j] = 0

    # COORDENADAS A VALOR DE RHO Y DE THETA
    for i, element in enumerate(maximos):
        for j, element in enumerate(maximos[i]):
            if maximos[i][j] != 0:
                rho = rhos[i]
                theta = angulos[j]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = (a * rho) + (anchura / 2)
                y0 = (b * rho) + (altura / 2)
                x1 = int(x0 + lineSize * (-b))
                y1 = int(y0 + lineSize * (a))
                x2 = int(x0 - lineSize * (-b))
                y2 = int(y0 - lineSize * (a))
                plt.plot((x1, x2), (y1, y2), marker="x")
    plt.imshow(imagen)
    plt.show()


cat = 100
imagen1 = cv2.imread("1triangulo.png", 0)
imagen2 = cv2.imread("2lineas.png", 0)
imagen3 = cv2.imread("entrada.jpeg", 0)

votos_triangulos = hough(imagen1, cat)
votos_lineas = hough(imagen2, cat)
votos_entradas = hough(imagen3, cat)

bordesHough(imagen1, votos_triangulos, cat, 164, 100)
bordesHough(imagen2, votos_lineas, cat, 700, 250)
bordesHough(imagen3, votos_entradas, cat, 450, 100)
