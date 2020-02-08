"""
PROJET REALISER PAR :
BNEOUDJIT IMED EDDINE
BOUCETTA NOUR EL YAKINE


Dans ce projet on a fait d'abord les pretraitement.
On a appliquer la gamme corretion puis le dog(Difference of gaussian) et en fin contrast equalisation.
On les a appliquer sur tout les images puis on a mis les resultats dans un ficher (results) qui est dans le fichier.
Apres on a appliquer la fonction LTP qui nous donne deux vecteur UPPER PATTERN et LOWERPATTERN.
On a calculer les histogrames des deux vecteur upper et lower patterns.
On a aussi construit la fonction qui verifie si le nombre est uniforme ou non mais on l'a pas appliquer sur le programme cas elle augmente la complexité du programme.
Apres tout ça on a fait une fonction qui calcule la distance entre une image de testes et les images indexées et en dessous
de cette fonction on trouve la fonction qui affiche les resultats des plus petites distances et en fin la fonction qui affiches les images resultats.

On a aussi le ficher Excel qui nous montre compbien de resultat juste on a trouver en K=3 puis k=5 puis k=7 et en fin k=9.
"""

import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image

#Fonction pour lire l'image
def readImage(ImageName):

    image = cv2.imread(ImageName,0)
    return image

#Fonction pour l'affichage de l'image
def showImage(ImageName):
    cv2.imshow('image', ImageName)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Fonction pour enregistrer l'image traité
def saveImage(savedImgName,newImg):
    outputFile =  savedImgName
    cv2.imwrite(outputFile, newImg)



#Fonction pour appliquer la correction de gamme
def applygammacorrection(image):

    img = np.empty((image.shape[0], image.shape[1]), np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i, j] = ((image[i, j] / 255) ** 0.2) * 255
    return img


#Fonction pour appliquer la difference de gaussien
def DOG(image):
    img=image/255
    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur5 = cv2.GaussianBlur(img, (0, 0), 1)
    blur3 = cv2.GaussianBlur(img, (0, 0), 2)
    num = blur5 - blur3
    num=num*255
    return num


#Cette fonction nous donnes des resultat qui sont proche des resultat du paper mais les images ne sont pa pareils
def ContrastEqualization(image):
    gray = cv2.normalize(image, dst=image, alpha=1.0, beta=290, norm_type=cv2.NORM_MINMAX)
    return gray

#Cette fonction retourne des resultats qui sont plus proche que la premiere et les images sont pareils
def ContEqualization1(image):
    return image+130

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
    return L

#Fonction pour convertir du binaire au dicimal
def BinToDec(mat):
    mat.reverse()
    decimal=0
    for i in range(len(mat)):
            decimal+= mat[i]*(2**i)
    return decimal

# Fonction qui calcule et affiche la liste des Upper patter
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


                    b += 1

                a += 1

            #ici on a applliquer la fonction qui affiche notre numero binaire sur notre matrice binaire puis on a appliquer la fonction qui converte binaire en dicimal
            listUpper.append(BinToDec(Binary(mat)))
    return listUpper


# Fonction qui calcule et affiche la liste des lower patterns
def LTPLowerPattern(img):
    t = 5
    mat = np.zeros((3, 3), int)
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
            # ici on a applliquer la fonction qui affiche notre numero binaire sur notre matrice binaire puis on a appliquer la fonction qui converte binaire en dicimal
            listLower.append(BinToDec(Binary(mat)))
    return listLower

# Fonction qui calcule l'histograme (pour les upper et les lower )
def CalculeHist(List):
    histogram = []
    for i in range(0, 256):
        histogram.append(0)
    for i in range(len(List)):
        histogram[List[i]] += 1
    return histogram
#Fonction pour concatiner les deux histogrames de LTP upper + lower et qui retourne le l'histograme de LTP
def HistLTP(Image):
    HistUpper = CalculeHist(LTPUpperPattern(Image))
    HistLower = CalculeHist(LTPLowerPattern(Image))
    L= HistUpper+HistLower
    return L

#Fonction pour appliquer les 3 pretraitement pour toutes les images de la base et sauvgarder toutes ces images avec les pretraitements
def IndxPreTraitement():
    for i in range(1,401):
        img=readImage('faces/'+str(i)+'.jpg')
        gamma = applygammacorrection(img)
        dog = DOG(gamma)
        ContEq = dog+130
        saveImage('results/'+str(i)+'.jpg', ContEq)
        print(i)
#cette fonction verifie si les nombre sont uniforme ou non-uniforme
def VerifUniform(Numero):
    result = False
    UinformList=[0,1,2,3,4,6,7,8,12,14,15,16,24,28,30,31,32,48,56,60,62,63,64,96,112,120,124,126,127,128,129,131,135,143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255]
    if Numero in UinformList:
        result=True
    return result
#Fonction pour calcule la distance entre l'image de test et les image indexé
def calculeDistance(HistImgTest):
    Tab=[]
    for i in range(1,361):
         image = readImage('results/'+str(i)+'.jpg')
         Distance = distance.euclidean(HistImgTest,HistLTP(image))
         Tab.append(Distance)
         print(i)

    return Tab

#Fonction qui affiche les indexes des images qui ont la plus petites distance
def resultats(Listdist):
    list = sorted(Listdist)[:9]
    ListMin=[]
    for i in range(len(Listdist)):

        if Listdist[i] in list :
            ListMin.append(i+1)
    return ListMin

#Fontion qui affiche les images resultats
def DisplayImgResults(resultsList):
    for i in range(len(resultsList)):
        o = readImage('faces/' + str(resultsList[i]) + '.jpg')
        im = Image.open('faces/' + str(resultsList[i]) + '.jpg')
        im.show()

"""
___main____
"""
img=readImage('results/'+str(399)+'.jpg')
HistogramLTP=HistLTP(img)


Listdist = calculeDistance(HistogramLTP)
resultsList = resultats(Listdist)
print(resultats(Listdist))
DisplayImgResults(resultsList)







#IndxPreTraitement()
