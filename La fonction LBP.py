import cv2
import numpy as np
from PIL import Image

def dec(mat):
    L = []
    s = L.append(mat[0][0])
    s = L.append(mat[0][1])
    s = L.append(mat[0][2])
    s = L.append(mat[1][2])
    s = L.append(mat[2][2])
    s = L.append(mat[2][1])
    s = L.append(mat[2][0])
    s = L.append(mat[1][0])
    L.reverse()
    dec=0
    for i in range(len(L)):
        dec+= L[i]*(2**i)
    return dec


img=Image.open('1.jpg').convert('L')
img=np.array(img)
print(img)
#img= cv2.imread('19.jpg')
mat = np.zeros((3,3),np.uint8)

image = np.empty((img.shape[0]-1,img.shape[1]-1),int)
list=[]
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape [1]-1):

        a=0
        for x in range(i-1,i+2):
            b=0
            for y in range(j-1,j+2):
                if img[x,y] >= img[i,j]:
                    mat[a,b]=1
                else:
                    mat[a,b]=0

                b+=1
                print(mat)
            a+=1
        image[i,j] = dec(mat)
        list.append(dec(mat))
print(list)
