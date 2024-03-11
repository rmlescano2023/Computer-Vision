import sys
import cv2
import numpy as np

def cross_correlation(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    
    #   Get local copy of img
    image = img
    
    #   Extract image dimensions height and width
    image_height, image_width = image.shape[:2]
    
    #   Extract kernel dimensions, assuming that the kernel is m x n
    kernel_height, kernel_width = kernel.shape

    #   Initialize number of image channels
    image_channels = 0
    
    #   Check if the image is grayscale or RGB
    if len(image.shape) == 3:             #     Cover the case where the image is colored
        image_channels = image.shape[2]
    else:                                 #     Cover the case where the image is greyscale
        image_channels = 1
        image = image[:, :, np.newaxis]   #     Convert grayscale image to 3-channel (for consistency)

    #   Initialize blank image for result of operation
    processed_image = np.zeros_like(image)
                
    #   Pad image with zeroes
    padded_image = np.zeros((image_height+kernel_height-1,image_width+kernel_width-1, image_channels))
    for c in range(image_channels):
        for i in range(image_height):
            for j in range(image_width):
                padded_image[i + int((kernel_height - 1) / 2), j + int((kernel_width - 1) / 2), c] = image[i, j, c]  #  Copy image to padded array

    print('Padding')
    preview_image('Padded', padded_image)

    #   Loop for computing cross-correlation with kernel
    for i in range(image_height):
        for j in range(image_width):
            for c in range(image_channels):
                #   Extract the window from the padded image for the current color channel
                window = padded_image[i : i + kernel_height, j : j + kernel_width, c]
                #   Perform element-wise multiplication between padded image and kernel
                processed_image[i, j, c] = np.sum(window * kernel)

    return processed_image

def preview_image(title, image):

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    # Load the images
    image_1 = cv2.imread('image_1.png')     # Smiling
    image_2 = cv2.imread('image_2.png')     # Crying

    # Preview of the images
    """ preview_image('Smiling', image_1)
    preview_image('Crying', image_2) """

    # Mean Filter Kernel
    kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    # Test cross-correlation
    result_image = cross_correlation(image_1, kernel)

    preview_image('Result', result_image)

if __name__ == "__main__":
    main()