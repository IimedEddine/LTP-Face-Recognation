import cv2
import numpy as np
from PIL import Image

tab = np.array([[9,1,4,2,6],[7,8,9,2,7],[6,6,5,3,3],[8,1,4,7,1],[4,6,2,1,3]])

#Fonction pour afficher le code binaire pour chaque pixel en appliquant LTP
def Binary(mat):
    L = []
    L.append(mat[0][0])
    L.append(mat[0][1])
    L.append(mat[0][2])
    L.append(mat[1][2])
    L.append(mat[2][2])
    L.append(mat[2][1])
    L.append(mat[2][0])
    L.append(mat[1][0])
    #L.reverse()
    return L


#Fonction pour convertir du binaire au dicimal
def BinToDec(mat):
    mat.reverse()
    decimal=0
    for i in range(len(mat)):

            decimal+= mat[i]*(2**i)
    return decimal


#Fontion qui affiche les resultats de LTP
def LTP(img):
    mat = np.zeros((3, 3), int)

    listUpper = []
    t = 5
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            a = 0
            for x in range(i - 1, i + 2):
                b = 0
                for y in range(j - 1, j + 2):
                    if img[x, y] > img[i, j] + t:
                        mat[a, b] = 1
                    elif img[x, y] < img[i, j] - t:
                        mat[a, b] = -1
                    else:
                        mat[a, b] = 0

                    #print(mat)
                    b += 1
                    #print(mat)
                a += 1


            listUpper.append(Binary(mat))
    return listUpper


# Fonction qui calcule Upper patter l'histograme des nombre positifs les 1
def LTPUpperPattern(img):
    mat = np.zeros((3, 3), int)

    listUpper = []
    t = 5
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            a = 0
            for x in range(i - 1, i + 2):
                b = 0
                for y in range(j - 1, j + 2):
                    if img[x, y] > img[i, j] + t:
                        mat[a, b] = 1
                    elif img[x, y] < img[i, j] - t:
                        mat[a, b] = 0
                    else:
                        mat[a, b] = 0

                    #print(mat)
                    b += 1
                    #print(mat)
                a += 1


            listUpper.append(Binary(mat))
    return listUpper


# Fonction qui calcule et affiche les lower patterns l'histograme des nombres negatives -1
def LTPLowerPattern(img):
    t = 5
    mat = np.zeros((3, 3), int)
    image = np.empty((img.shape[0] - 1, img.shape[1] - 1), int)
    listLower = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            a = 0
            for x in range(i - 1, i + 2):
                b = 0
                for y in range(j - 1, j + 2):
                    if img[x, y] > img[i, j] + t:
                        mat[a, b] = 0
                    elif img[x, y] < img[i, j] - t:
                        mat[a, b] = 1
                    else:
                        mat[a, b] = 0


                    b += 1

                a += 1
            #print(BinToDec(Binary(mat)))
            #image[i, j] = BinToDec(mat)
            listLower.append(Binary(mat))
    print( listLower)


#afficher les resultat LTP
print(LTP(tab))
#afficher les resultat LTPUpperPattern
print(LTPUpperPattern(tab))
#afficher les resultat LTPLowerPatter
print(LTPLowerPattern(tab))