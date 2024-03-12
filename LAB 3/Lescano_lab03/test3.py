# In this code, I will convert everything back to stacks

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
PYRAMID_LEVELS = 5                  # Number of levels in the Gaussian and Laplacian pyramids
GAUSSIAN_KERNEL_SIZE = 5            # Size of the Gaussian kernel


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN PYRAMID
def generate_gaussian_pyramid(image, levels, kernel_size):

    image_pyramid = [image]       # Original image is at index 0

    for i in range(levels):
        blurred_image = gaussian_blur(image_pyramid[-1], kernel_size)
        image_pyramid.append(blurred_image)

    return image_pyramid


# --------------------------------------------------------------------------------------------------------------- LAPLACIAN PYRAMID
def generate_laplacian_pyramid(gaussian_pyramid, levels):    # Levels = 5

    image_pyramid = [gaussian_pyramid[levels - 1]]      # Order: [L4, L3, L2, L1, L0]

    for i in range(levels - 1, 0, -1):      # Iteration: i = 4, i = 3, i = 2, i = 1

        # Construct Laplacian images
        result = cv2.subtract(gaussian_pyramid[i - 1], gaussian_pyramid[i])
        image_pyramid.append(result)

    return image_pyramid

"""""
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

        result = np.hstack((left_img[ : , 0 : int(columns/2)], right_img[ : , int(columns/2) : ]))

        concatenated_images.append(result)
    
    return concatenated_images

def blend_images(concat_result, levels):

    blended_image = concat_result[0]

    # Iterate over the Laplacian images and add them to the blended image
    for i in range(1, levels):
        
        # This is in pyramids
        # blended_image = cv2.pyrUp(blended_image)                    # upsample current image. so in iteration 1, current image is concat_result[0]
        # blended_image = cv2.add(concat_result[i], blended_image)

        
        
        
        # Upscale the blended image manually to match the size of the Laplacian image
        # upscaled_blend = cv2.resize(blended_image, (concat_result[i].shape[1], concat_result[i].shape[0]))
        
        # Add the upscaled blended image and the Laplacian image at the current level
        # blended_image = upscaled_blend + concat_result[i]
        blended_image = cv2.add(blended_image, concat_result[i])

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

    for i, level in enumerate(pyramid):
        cv2.imshow('Level {} of {}'.format(i, pyramid_type), level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result):
    
    visualize_pyramid(img1_gaussian_pyramid, pyramid_type="Gaussian")
    visualize_pyramid(img2_gaussian_pyramid, pyramid_type="Gaussian")
    visualize_pyramid(img1_laplacian_pyramid, pyramid_type="Laplacian")
    visualize_pyramid(img2_laplacian_pyramid, pyramid_type="Laplacian")
    visualize_pyramid(concat_result, pyramid_type="Concatenated")

def show_blended_image(blended_image):

    cv2.namedWindow("Pyramid_blending")
    cv2.moveWindow("Pyramid_blending", 30, 30)
    cv2.imshow("Pyramid_blending", blended_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------------------- MAIN
def main():

    # Load the images
    image_1 = cv2.imread('examples/Lescano_lab03_left.png')
    image_2 = cv2.imread('examples/Lescano_lab03_right.png')

    # Image preview
    preview_images(image_1, image_2)

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Generate Gaussian pyramid of the two images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels, kernel_size)     # apple
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels, kernel_size)     # orange

    # visualize_pyramid(img1_gaussian_pyramid, 'Gaussian Apple')
    # visualize_pyramid(img2_gaussian_pyramid, 'Gaussian Orange')

    # Generate Laplacian pyramid of the two images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid, levels)    # apple
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid, levels)    # orange

    # visualize_pyramid(img1_laplacian_pyramid, 'Laplacian Apple')
    # visualize_pyramid(img2_laplacian_pyramid, 'Laplacian Orange')

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_pyramid, img2_laplacian_pyramid)

    # visualize_pyramid(concat_result, 'Concatenated Apple & Orange')

    # Blending
    blended_image = blend_images(concat_result, levels)
    # cv2.imwrite('output/stack_blend.png', blended_image)
    show_blended_image(blended_image) 


    """ 



    # Visualizations
    # test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result)
    
    """

if __name__ == "__main__":
    main()