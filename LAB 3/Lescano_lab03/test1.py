import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------- GLOBAL VARIABLES
STACK_LEVELS = 5                # Number of levels in the Gaussian and Laplacian stacks
GAUSSIAN_KERNEL_SIZE = 7        # Size of the Gaussian kernel


# --------------------------------------------------------------------------------------------------------------- GAUSSIAN STACK
def generate_gaussian_stack(image, levels, kernel_size):

    image_stack = [image]       # Original image is at index 0

    for i in range(levels):
        blurred_image = gaussian_blur(image_stack[-1], kernel_size)
        image_stack.append(blurred_image)

    return image_stack

    # Order:
    # Level 0 = Original Image
    # Level 5 = Blurriest Gaussian
    # image_stack = [L0, L1, L2, L3, L4, L5]    => 5 levels, 6 images


# --------------------------------------------------------------------------------------------------------------- LAPLACIAN STACK
def generate_laplacian_stack(stack, levels):    # Levels = 5

    image_stack = []   

    for i in range(levels):
        img_1 = stack[i]
        img_2 = stack[i + 1]
        up_sample_image = img_1 - img_2
        image_stack.append(up_sample_image)

    image_stack.append(stack[levels - 1])

    return image_stack

    # L0 = L0 - L1
    # L1 = L1 - L2
    # L2 = L2 - L3
    # L3 = L3 - L4
    # L4 = L4 - L5
    # L5 = L5


# --------------------------------------------------------------------------------------------------------------- BLENDING
def concat_images(stack_1, stack_2):       # stack_1 is apple, stack_2 is orange

    concatenated_images = []

    # left_img is from stack_1, right_img is from stack_2
    for left_img, right_img in zip(stack_1, stack_2):
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

def visualize_stack(stack):

    for i, level in enumerate(stack):
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
    levels = STACK_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Generate Gaussian stacks of the two images
    img1_gaussian_stack = generate_gaussian_stack(image_1, levels, kernel_size)     # apple
    img2_gaussian_stack = generate_gaussian_stack(image_2, levels, kernel_size)     # orange

    # Generate Laplacian stacks of the two images
    img1_laplacian_stack = generate_laplacian_stack(img1_gaussian_stack, levels)    # apple
    img2_laplacian_stack = generate_laplacian_stack(img2_gaussian_stack, levels)    # orange

    # Visualize Gaussian & Laplacian stacks
    # visualize_stack(img1_gaussian_stack)
    # visualize_stack(img2_gaussian_stack)
    # visualize_stack(img1_laplacian_stack)
    # visualize_stack(img2_laplacian_stack)

    # Concatenating the half images
    concat_result = concat_images(img1_laplacian_stack, img2_laplacian_stack)
    visualize_stack(concat_result)

    # Blending
    """ blend_with_invisible_edge = concat_result[0]
    for i in range(1, levels):
        # blend_with_invisible_edge = cv2.pyrUp(blend_with_invisible_edge)
        blend_with_invisible_edge = cv2.add(blend_with_invisible_edge, concat_result[i])

    cv2.namedWindow("Pyramid_blending")
    cv2.moveWindow("Pyramid_blending", 30, 30)
    cv2.imshow("Pyramid_blending", blend_with_invisible_edge)      

    cv2.waitKey(0)
    cv2.destroyAllWindows() """


if __name__ == "__main__":
    main()


# STEPS:
    
# In Gaussian Pyramid, blur only the image (don't subsample).
    # You typically use the Gaussian filter.

# Laplacian stack needs to be modified.
    
# Generate Gaussian stack of the images first, because that will be used as parameters for
    # generating the Laplacian stacks. At the end, when blending the two images, only the
    # Laplacian stack will be utilized.