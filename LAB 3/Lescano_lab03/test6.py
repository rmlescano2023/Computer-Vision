# In this code, I will resort to using pyramids instead of stacks

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
PYRAMID_LEVELS = 5                # Number of levels in the Gaussian and Laplacian pyramids
GAUSSIAN_KERNEL_SIZE = 5        # Size of the Gaussian kernel


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN PYRAMID
def generate_gaussian_pyramid(image, levels, kernel_size):

    image_pyramid = [image]       # Original image is at index 0

    for i in range(levels):
        blurred_image = gaussian_blur(image_pyramid[-1], kernel_size)
        blurred_image = cv2.resize(blurred_image, (blurred_image.shape[1] // 2, blurred_image.shape[0] // 2))
        image_pyramid.append(blurred_image)

    return image_pyramid


# --------------------------------------------------------------------------------------------------------------- LAPLACIAN PYRAMID
def generate_laplacian_pyramid(gaussian_pyramid, levels):    # Levels = 5

    image_pyramid = [gaussian_pyramid[levels - 1]]

    for i in range(levels - 1, 0, -1):
        # Upscale the images manually
        upscaled_img = cv2.resize(gaussian_pyramid[i], (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))

        # Construct Laplacian images
        result = cv2.subtract(gaussian_pyramid[i - 1], upscaled_img)
        image_pyramid.append(result)

    return image_pyramid


# --------------------------------------------------------------------------------------------------------------- BLENDING
def concat_images(pyramid_1, pyramid_2):       # pyramid_1 is apple, pyramid_2 is orange

    concatenated_images = []

    # left_img is from pyramid_1, right_img is from pyramid_2
    for left_img, right_img in zip(pyramid_1, pyramid_2):
        columns = left_img.shape[1]

        result = np.hstack((left_img[ : , 0 : columns//2], right_img[ : , columns//2 : ]))

        concatenated_images.append(result)
    
    return concatenated_images


# --------------------------------------------------------------------------------------------------------------- HELPER FUNCTIONS
def gaussian_blur(image, kernel_size):

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def preview_images(image_1, image_2):

    value = image_1.shape[1]
    image_1_x_position = 30
    image_1_y_position = 30
    image_2_x_position = image_1_x_position + value
    image_2_y_position = image_1_y_position

    cv2.namedWindow('Image 1')
    cv2.moveWindow('Image 1', image_1_x_position, image_1_y_position)
    cv2.imshow('Image 1', image_1)

    cv2.namedWindow('Image 2')
    cv2.moveWindow('Image 2', image_2_x_position, image_2_y_position)
    cv2.imshow('Image 2', image_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_pyramid(pyramid):

    for i, level in enumerate(pyramid):
        cv2.imshow(f'Level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------------------- MAIN
def main():

    # Load the images
    image_1 = cv2.imread('examples/apple.jpg')
    image_2 = cv2.imread('examples/orange.jpg')

    # Image preview
    preview_images(image_1, image_2)

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Generate Gaussian pyramid of the two images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels, kernel_size)     # apple
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels, kernel_size)     # orange

    # Generate Laplacian pyramid of the two images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid, levels)    # apple
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid, levels)    # orange

    # Visualize Gaussian & Laplacian pyramids
    visualize_pyramid(img1_gaussian_pyramid)
    visualize_pyramid(img2_gaussian_pyramid)
    visualize_pyramid(img1_laplacian_pyramid)
    visualize_pyramid(img2_laplacian_pyramid)

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_pyramid, img2_laplacian_pyramid)
    visualize_pyramid(concat_result)

    # Blending
    blend_with_invisible_edge = concat_result[0]

    # Iterate over the Laplacian images and add them to the blended image
    for i in range(1, levels):
        # Upscale the blended image manually to match the size of the Laplacian image
        upscaled_blend = cv2.resize(blend_with_invisible_edge, (concat_result[i].shape[1], concat_result[i].shape[0]))
        
        # Add the upscaled blended image and the Laplacian image at the current level
        blend_with_invisible_edge = upscaled_blend + concat_result[i]

    cv2.namedWindow("Pyramid_blending")
    cv2.moveWindow("Pyramid_blending", 30, 30)
    cv2.imshow("Pyramid_blending", blend_with_invisible_edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# STEPS:
    
# In Gaussian Pyramid, blur only the image (don't subsample).
    # You typically use the Gaussian filter.

# Laplacian stack needs to be modified.
    
# Generate Gaussian stack of the images first, because that will be used as parameters for
    # generating the Laplacian stacks. At the end, when blending the two images, only the
    # Laplacian stack will be utilized.