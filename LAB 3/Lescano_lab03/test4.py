import cv2
import numpy as np

# from Lescano_lab03_blending import *

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
PYRAMID_LEVELS = 6                  # Number of levels in the Gaussian and Laplacian pyramids
GAUSSIAN_KERNEL_SIZE = 7            # Size of the Gaussian kernel


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN PYRAMID
def gaussian_pyramid(img, num_levels):

    lower = img.copy()

    gaussian_pyr = [lower]          # index 0 is the copy of the original image

    for i in range(num_levels):
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        gaussian_pyr.append(img)

    return gaussian_pyr


# --------------------------------------------------------------------------------------------------------------- LAPLACIAN PYRAMID
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.resize(gaussian_pyr[i], size)
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

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


def blend_crazy_images(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct_crazy_images(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.resize(laplacian_top, size)
        laplacian_top = cv2.add(laplacian_expanded, laplacian_pyr[i + 1])
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


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
    img1_gaussian_pyramid = gaussian_pyramid(image_1, levels)         # apple
    img2_gaussian_pyramid = gaussian_pyramid(image_2, levels)         # orange

    # Generate Laplacian pyramid of the two images
    img1_laplacian_pyramid = laplacian_pyramid(img1_gaussian_pyramid)      # apple
    img2_laplacian_pyramid = laplacian_pyramid(img2_gaussian_pyramid)      # orange

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_pyramid, img2_laplacian_pyramid)

    # Visualizations
    # test_visualizations(img1_gaussian_pyramid, img2_gaussian_pyramid, img1_laplacian_pyramid, img2_laplacian_pyramid, concat_result)
    
    # Blending
    blended_image = blend_images(concat_result, levels)
    cv2.imwrite('output/test.png', blended_image)
    show_blended_image(blended_image)

def blend_crazy():

    # Load the two images
    img1 = cv2.imread('examples/sky.png')
    img1 = cv2.resize(img1, (1800, 1000))
    img2 = cv2.imread('examples/plane.jpg')
    img2 = cv2.resize(img2, (1800, 1000))

    # Create the mask
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[250:500,640:1440,:] = (1,1,1)
    
    num_levels = 7
    
    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)

    # Visualize
    # visualize_pyramid(gaussian_pyr_1, 'Gaussian One')
    # visualize_pyramid(laplacian_pyr_1, 'Laplacian One')

    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    # Try using my program
    # gaussian_pyr_2 = generate_gaussian_pyramid(img2, num_levels, 5)
    # laplacian_pyr_2 = generate_laplacian_pyramid(gaussian_pyr_2, num_levels)

    # Visualize
    # visualize_pyramid(laplacian_pyr_2, 'Laplacian Two')

    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()

    # Blend the images
    add_laplace = blend_crazy_images(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)

    # Reconstruct the images
    final = reconstruct_crazy_images(add_laplace)

    # Save the final image to the disk
    cv2.imwrite('output/crazyblend_test_4.png', final[num_levels])


def main():

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Vertical blend
    blend_apple_orange(levels, kernel_size)

    # Blend two images using an irregular mask
    blend_crazy()


if __name__ == '__main__':
    main()