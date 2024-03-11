# This program creates a Gaussian stack. The stack consists of 5 levels, with the application of
# Gaussian blurring in each level.

import cv2
import numpy as np

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def generate_gaussian_stack(image, stack_levels, kernel_size=5):

    image_stack = [image]       # Original image is appended at index 0

    for i in range(stack_levels):
        blurred_image = gaussian_blur(image_stack[-1], kernel_size)
        image_stack.append(blurred_image)

    return image_stack

def generate_laplacian_stack(gaussian_stack, stack_levels):

    image_stack = []

    for i in range(stack_levels):
        img_1 = gaussian_stack[i]
        img_2 = gaussian_stack[i + 1]
        up_sample_image = img_1 - img_2
        image_stack.append(up_sample_image)

    image_stack.append(gaussian_stack[stack_levels - 1])

    return image_stack

def concatenate_laplacian_images(laplacian_stack):

    concatenated_images = []



# Example usage:
image_1 = cv2.imread('examples/apple.jpg')
stack_levels = 5
gaussian_kernel_size = 5

gaussian_stack = generate_gaussian_stack(image_1, stack_levels, gaussian_kernel_size)       # This is an array
laplacian_stack = generate_laplacian_stack(gaussian_stack, stack_levels)

concatenate_laplacian_images(laplacian_stack)

# Visualize Gaussian stack
for i, level in enumerate(gaussian_stack):
    cv2.imshow(f'Level {i}', level)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize Laplacian stack
for i, level in enumerate(laplacian_stack):
    cv2.imshow(f'Level {i}', level)
    cv2.waitKey(0)
    cv2.destroyAllWindows()