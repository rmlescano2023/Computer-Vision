import cv2
import numpy
import os

# Make an array of 480,000 random bytes.
randomByteArray = bytearray(os.urandom(480000))
flatNumpyArray = numpy.array(randomByteArray)

# Convert the array to make a 800x600 grayscale image.
grayImage = flatNumpyArray.reshape(800, 600)
cv2.imwrite('out/RandomGray.png', grayImage)

# Convert the array to make a 400x400 color image.
bgrImage = flatNumpyArray.reshape(400, 400, 3)
cv2.imwrite('out/RandomColor.png', bgrImage)

cv2.imshow('Random Gray Image', grayImage)
cv2.imshow('Random Color  Image', bgrImage)

cv2.waitKey()
cv2.destroyAllWindows()


