# This is the code from https://www.youtube.com/watch?v=_B9kP_N8YjQ

import cv2
import numpy as np, sys

A = cv2.imread('examples/apple.jpg')
B = cv2.imread('examples/orange.jpg')

val = A.shape[1]
posAx = 30
posAy = 30
posBx = posAx + val
posBy = posAy

cv2.namedWindow('img1')
cv2.moveWindow('img1', posAx, posAy)
cv2.imshow('img1', A)
cv2.namedWindow('img2')
cv2.moveWindow('img2', posBx, posBy)
cv2.imshow('img2', B)

N_levels = 5

lstA_G = []     # apple image, gaussian
lstB_G = []     # orange image, gaussian

G1 = A.copy()
G2 = B.copy()

## Gaussian Pyramids: At each level, the image dimensions are reduced in half
for i in range(N_levels):
    lstA_G.append(G1)
    lstB_G.append(G2)

    G1 = cv2.pyrDown(G1)
    G2 = cv2.pyrDown(G2)

    posAx = posAx + val // (2**(i + 1))
    posAy = posAy + val // (2**(i + 1))
    posBx = posBx + val // (2**(i + 1))
    posBy = posBy + val // (2**(i + 1))

    cv2.namedWindow("img1 scaled down at i = " + str(i))
    cv2.moveWindow("img1 scaled down at i = " + str(i), posAx, posAy)
    cv2.imshow("img1 scaled down at i = " + str(i), G1)

    cv2.namedWindow("img2 scaled down at i = " + str(i))
    cv2.moveWindow("img2 scaled down at i = " + str(i), posBx, posBy)
    cv2.imshow("img2 scaled down at i = " + str(i), G2)

cv2.waitKey(0)
cv2.destroyAllWindows()

posAx = 30
posAy = 30
posBx = posAx + val
posBy = posAy


## Laplacian Pyramids: At each level, the Laplacian Pyramid image is the difference between the
# Gaussian Pyramid image at the level and expanded Gaussian Pyramid image at the level above it
lstA_L = [lstA_G[N_levels - 1]]     # apple image, laplacian
lstB_L = [lstB_G[N_levels - 1]]     # orange image, laplacian

for i in range(N_levels - 1, 0, -1):
    L1 = cv2.subtract(lstA_G[i-1], cv2.pyrUp(lstA_G[i]))
    lstA_L.append(L1)
    L2 = cv2.subtract(lstB_G[i-1], cv2.pyrUp(lstB_G[i]))
    lstB_L.append(L2)

    cv2.namedWindow("10*img1 Laplacian at i = " + str(i - 1))
    cv2.moveWindow("10*img1 Laplacian at i = " + str(i - 1), posAx, posAy)
    cv2.imshow("10*img1 Laplacian at i = " + str(i - 1), 10*L1)

    cv2.namedWindow("10*img2 Laplacian at i = " + str(i - 1))
    cv2.moveWindow("10*img2 Laplacian at i = " + str(i - 1), posBx, posBy)
    cv2.imshow("10*img2 Laplacian at i = " + str(i - 1), 10*L2)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Concatenating the half images in a list
LS = []
for la, lb in zip(lstA_L, lstB_L):      # zip() function can basically iterate through multiple lists as its parameters, and then returns a tuple
    cols = la.shape[1]      # get the value of the columns

    # hstack is horizontal stack, concatenates two inputs horizontally
    # so, in this, we concatenate horizontally each image from both lists at the same level
    ls = np.hstack((la[ : , 0:cols//2], lb[ : , cols//2 : ]))       # la[row_x : row_y, column_x : column_y] => these are ranges
    
    LS.append(ls)

# Visualize LS
for i, level in enumerate(LS):
    cv2.imshow(f'Level {i}', level)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## Constructing the blended image from its Laplacian Pyramid: Starting with the bottom of the
# Laplacian Pyramid images and then adding the expanded Laplacian Pyramid image at the next
# level to the current level Laplacian Pyramid image
blend_with_invisible_edge = LS[0]
for i in range(1, N_levels):
    blend_with_invisible_edge = cv2.pyrUp(blend_with_invisible_edge)
    blend_with_invisible_edge = cv2.add(blend_with_invisible_edge, LS[i])

# Blending without pyramids
cols = A.shape[1]
blend_with_visible_edge = np.hstack((A[:,:cols//2], B[:,cols//2:]))

cv2.namedWindow("Pyramid_blending")
cv2.moveWindow("Pyramid_blending", 30, 30)
cv2.imshow("Pyramid_blending", blend_with_invisible_edge)

cv2.namedWindow("Direct_blending")
cv2.moveWindow("Direct_blending", 30 + 512, 30)
cv2.imshow("Direct_blending", blend_with_visible_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()