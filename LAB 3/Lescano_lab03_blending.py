""" 
    File: Computer Vision Laboratory 03
    Author: Renmar M. Lescano
    Date Modified: 03/13/2024

    Description:        
        This Python script demonstrates how to blend images using Gaussian and Laplacian pyramids. It offers two
        options: blending images vertically and blending them with a custom mask. The vertical blending method involves
        splitting images into pyramids and combining their left halves. On the other hand, blending with a custom mask
        involves creating a mask to blend images selectively. These techniques showcase effective ways to seamlessly
        blend images for various purposes.
"""

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
PYRAMID_LEVELS = 6                  # Number of levels in the Gaussian and Laplacian pyramids


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN PYRAMID
def generate_gaussian_pyramid(img, levels):

    lower = img.copy()

    gaussian_pyr = [lower]          # index 0 is the copy of the original image

    for i in range(levels):
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        gaussian_pyr.append(img)

    return gaussian_pyr


# --------------------------------------------------------------------------------------------------------------- LAPLACIAN PYRAMID
def generate_laplacian_pyramid(gaussian_pyr):

    laplacian_pyr = [gaussian_pyr[-1]]

    levels = len(gaussian_pyr) - 1

    for i in range(levels, 0, -1):
        gaussian_expanded = cv2.resize(gaussian_pyr[i], (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0]))
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)

        laplacian_pyr.append(laplacian)

    return laplacian_pyr

# --------------------------------------------------------------------------------------------------------------- BLENDING
def concat_images(pyramid_1, pyramid_2):

    concatenated_images = []

    for left_img, right_img in zip(pyramid_1, pyramid_2):
        columns = left_img.shape[1]
        result = np.hstack((left_img[ : , 0 : columns//2], right_img[ : , columns//2 : ]))

        concatenated_images.append(result)
    
    return concatenated_images


def blend_images(concat_result, levels):

    blended_image = concat_result[0]

    # Iterate over the Laplacian images and add them to the blended image
    for i in range(1, levels):
        # Upscale the blended image manually to match the size of the Laplacian image
        upscaled_blend = cv2.resize(blended_image, (concat_result[i].shape[1], concat_result[i].shape[0]))
        
        # Add the upscaled blended image and the Laplacian image at the current level
        blended_image = cv2.add(upscaled_blend, concat_result[i])

    return blended_image


def blend_crazy_images(laplacian_pyramid_1, laplacian_pyramid_2, mask_pyramid_final):

    LS = []

    for la, lb, mask in zip(laplacian_pyramid_1, laplacian_pyramid_2, mask_pyramid_final):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)

    return LS


def reconstruct_crazy_images(blended_images):

    laplacian_top = blended_images[0]
    reconstructed_images = [laplacian_top]

    levels = len(blended_images) - 1

    for i in range(levels):
        laplacian_expanded = cv2.resize(laplacian_top, (blended_images[i + 1].shape[1], blended_images[i + 1].shape[0]))
        laplacian_top = cv2.add(laplacian_expanded, blended_images[i + 1])
        
        reconstructed_images.append(laplacian_top)

    return reconstructed_images


# --------------------------------------------------------------------------------------------------------------- HELPER FUNCTIONS
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


def visualize_pyramid(pyramid, pyramid_type):

    image_x_position = 50
    image_y_position = 50

    for i, level in enumerate(pyramid):

        value = pyramid[i].shape[1]     # width of image

        cv2.namedWindow('Level {} of {}'.format(i, pyramid_type))
        cv2.moveWindow('Level {} of {}'.format(i, pyramid_type), image_x_position, image_y_position)
        cv2.imshow('Level {} of {}'.format(i, pyramid_type), pyramid[i])

        # image_x_position += value

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result):

    visualize_pyramid(img1_gaussian_pyramid, pyramid_type="Gaussian")
    visualize_pyramid(img2_gaussian_pyramid, pyramid_type="Gaussian")
    visualize_pyramid(img1_laplacian_pyramid, pyramid_type="Laplacian")
    visualize_pyramid(img2_laplacian_pyramid, pyramid_type="Laplacian")
    visualize_pyramid(concat_result, pyramid_type="Concatenated")


# --------------------------------------------------------------------------------------------------------------- MAIN
def blend_vertical(levels):

    # Load the two images
    image_1 = cv2.imread('sample-images/Lescano_lab03_left.png')
    image_2 = cv2.imread('sample-images/Lescano_lab03_right.png')

    # Image preview
    preview_images(image_1, image_2)

    # Generate Gaussian pyramid of the two images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels)         # apple
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels)         # orange

    # Generate Laplacian pyramid of the two images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid)      # apple
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid)      # orange

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_pyramid, img2_laplacian_pyramid)

    # Visualizations
    # test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result)
    
    # Blending
    blended_image = blend_images(concat_result, levels)

    # Save the final image to the disk
    cv2.imwrite('output/Lescano_lab03_blendvert.png', blended_image)
    cv2.imshow('Vertical Blending Result', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blend_crazy(levels):

    # Load the two images
    img1 = cv2.imread('sample-images/Lescano_lab03_crazyone.png')    # Sky
    img1 = cv2.resize(img1, (1800, 1000))
    img2 = cv2.imread('sample-images/Lescano_lab03_crazytwo.jpg')    # Plane
    img2 = cv2.resize(img2, (1800, 1000))

    # Image preview
    preview_images(img1, img2)

    # Create the mask
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[250:500,640:1440,:] = (1,1,1)                          # turn row 250-550 & column 640-1440 into white (1,1,1)
        
    # Generate Gaussian pyramid of the two images
    gaussian_pyramid_1 = generate_gaussian_pyramid(img1, levels)
    gaussian_pyramid_2 = generate_gaussian_pyramid(img2, levels)

    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyramid_final = generate_gaussian_pyramid(mask, levels)
    mask_pyramid_final.reverse()

    # Generate Laplacian pyramid of the two images
    laplacian_pyramid_1 = generate_laplacian_pyramid(gaussian_pyramid_1)
    laplacian_pyramid_2 = generate_laplacian_pyramid(gaussian_pyramid_2)

    # Blend the images
    blended_images = blend_crazy_images(laplacian_pyramid_1, laplacian_pyramid_2, mask_pyramid_final)

    # Reconstruct the images
    final_image = reconstruct_crazy_images(blended_images)

    # Save the final image to the disk
    cv2.imwrite('output/Lescano_lab03_blendcrazy.png', final_image[levels])
    cv2.imshow('Crazy Blending Result', cv2.imread('output/Lescano_lab03_blendcrazy.png'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    # Define parameters
    levels = PYRAMID_LEVELS

    # Vertical blend
    blend_vertical(levels)

    # Blend two images using an irregular mask
    blend_crazy(levels)


if __name__ == '__main__':
    main()