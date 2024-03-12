# In this code, I will resort to using pyramids instead of stacks

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
PYRAMID_LEVELS = 6                  # Number of levels in the Gaussian and Laplacian pyramids
GAUSSIAN_KERNEL_SIZE = 7            # Size of the Gaussian kernel


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN PYRAMID
def generate_gaussian_pyramid(image, levels, kernel_size):

    image_pyramid = [image]       # Original image is at index 0

    for i in range(levels):
        blurred_image = gaussian_blur(image_pyramid[-1], kernel_size)   # blurs the last image in every iteration in the list
        blurred_image = cv2.resize(blurred_image, (blurred_image.shape[1] // 2, blurred_image.shape[0] // 2))
        image_pyramid.append(blurred_image)

    return image_pyramid

# Visualization:
# Iteration 1 = blurs original image, resizes to half, saved as L1 at index 2
# Iteration 2 = blurs L1, resizes to half, saved as L2 at index 3

# --------------------------------------------------------------------------------------------------------------- LAPLACIAN PYRAMID
def generate_laplacian_pyramid(gaussian_pyramid, levels):    # Levels = 5

    image_pyramid = [gaussian_pyramid[levels - 1]]          # L4 from gaussian_pyramid

    for i in range(levels - 1, 0, -1):          # Iteration: i = 4, i = 3, i = 2, i = 1
        # Upscale the images manually
        upscaled_img = cv2.resize(gaussian_pyramid[i], (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))

        # Construct Laplacian images
        result = cv2.subtract(gaussian_pyramid[i - 1], upscaled_img)
        image_pyramid.append(result)

    return image_pyramid

"""""
cv2.resize(src, (desired width, desired height)) = cv2.resize(L4, (L3 width, L3 height)) => upscaled
ang result kay upscaled image sang L4, same size na sa L3

L = Level

gaussian_pyramid = [    L0 = original image,
                        L1,
                        L2,
                        L3,
                        L4,
                        L5 = blurriest image
]


image_pyramid = [       L4,
                        i = 4; result = L3 - L4; L3 = L3 - L4
                        i = 3; result = L2 - L3; L2 = L2 - L3
                        i = 2; result = L1 - L2; L1 = L1 - L2
                        i = 1; result = L0 - L1; L0 = L0 - L1
]

"""""

# --------------------------------------------------------------------------------------------------------------- BLENDING
def concat_images(pyramid_1, pyramid_2):       # pyramid_1 is apple, pyramid_2 is orange

    concatenated_images = []

    # left_img is from pyramid_1, right_img is from pyramid_2
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

def visualize_pyramid(pyramid, pyramid_type):

    # value = pyramid[0].shape[1]     # width of image
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

def show_blended_image(blended_image):

    cv2.namedWindow("Pyramid Blending")
    cv2.moveWindow("Pyramid Blending", 30, 30)
    cv2.imshow("Pyramid Blending", blended_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------------------- MAIN
def blend_apple_orange(levels, kernel_size):

    # Load the images
    image_1 = cv2.imread('examples/Lescano_lab03_left.png')
    image_2 = cv2.imread('examples/Lescano_lab03_right.png')

    # Image preview
    preview_images(image_1, image_2)

    # Generate Gaussian pyramid of the two images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels, kernel_size)         # apple
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels, kernel_size)         # orange

    # Generate Laplacian pyramid of the two images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid, levels)      # apple
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid, levels)      # orange

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_pyramid, img2_laplacian_pyramid)

    # Visualizations
    # test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result)
    
    # Blending
    blended_image = blend_images(concat_result, levels)
    cv2.imwrite('output/test.png', blended_image)
    show_blended_image(blended_image)

def blend_crazy(levels, kernel_size):

    # Load the images
    image_1 = cv2.imread('examples/Lescano_lab03_crazyone.png')
    image_2 = cv2.imread('examples/Lescano_lab03_crazytwo.png')

    # Create the mask
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[250:500,640:1440,:] = (1,1,1)

    # Generate Gaussian pyramid of all the images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels, kernel_size)
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels, kernel_size)
    mask_gaussian_pyramid = generate_gaussian_pyramid(mask, levels, kernel_size)

    # Try using my program
    gaussian_pyr_1 = generate_gaussian_pyramid(image_1, levels, kernel_size)
    laplacian_pyr_1 = generate_laplacian_pyramid(gaussian_pyr_1, levels)

    # Visualize
    # visualize_pyramid(gaussian_pyr_1, 'Gaussian One')
    # visualize_pyramid(laplacian_pyr_1, 'Laplacian One')

    # Try using my program
    gaussian_pyr_2 = generate_gaussian_pyramid(image_2, levels, kernel_size)
    laplacian_pyr_2 = generate_laplacian_pyramid(gaussian_pyr_2, levels)

    # Visualize
    visualize_pyramid(gaussian_pyr_2, 'Gaussian Two')
    visualize_pyramid(laplacian_pyr_2, 'Laplacian Two')


def main():

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Vertical blend
    blend_apple_orange(levels, kernel_size)

    # Blend two images using an irregular mask
    # blend_crazy(levels, kernel_size)

if __name__ == "__main__":
    main()