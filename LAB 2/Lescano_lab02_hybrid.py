""" 
    File: Computer Vision Laboratory 02
    Author: Renmar M. Lescano
    Date Modified: 03/11/2024

    Description:        
        This Python script utilizes the cv2 package to implement an image filtering function. It applies this function to create
        hybrid images, a concept inspired by the SIGGRAPH 2006 paper authored by Oliva, Torralba, and Schyns. Hybrid images are
        static images whose interpretation varies with viewing distance. The principle behind them lies in how high-frequency
        details dominate perception up close, while only low-frequency features are discernible from afar. By blending the
        high-frequency components of one image with the low-frequency elements of another, hybrid images induce different 
        interpretations at different viewing distances.
"""

import sys
import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------- CROSS-CORRELATION
def cross_correlation(image, kernel):

    print('Processing cross-correlation...')

    image = image

    # Get image resolution
    image_height, image_width = image.shape[:2]
    
    # Get kernel dimensions
    kernel_height, kernel_width = kernel.shape

    # Check if the image is Grayscale or RGB
    image_channels = 0
    if len(image.shape) == 2:                 # Grayscale
        image_channels = 1
    elif len(image.shape) == 3:               # Colored
        image_channels = image.shape[2]
        
    # Padding
    image_padded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1, image_channels))     # ((height, width, channels))
    for k in range(image_channels):
        for i in range(image_height):
            for j in range(image_width):
                image_padded[i + int((kernel_height - 1) / 2), j + int((kernel_width - 1) / 2), k] = image[i, j, k]  #  Copy image to padded array

    # Correlation
    image_result = np.zeros_like(image)      # Initialize a blank image for the result of the correlation
    for i in range(image_height):
        for j in range(image_width):
            for k in range(image_channels):

                # Extracts a window from the padded image for the current pixel position (i, j) and color channel (k)
                window = image_padded[i : i + kernel_height, j : j + kernel_width, k]

                # Numpy does element-wise multiplication on arrays
                image_result[i, j, k] = np.sum(window * kernel)
    
    print('Cross-correlation process DONE!')

    return image_result


# ----------------------------------------------------------------------------------------------------------------------------- CONVOLUTION
def convolution(img, kernel):               # Convolution will use Cross-Correlation

    print('Processing convolution...')

    # Same as cross-correlation, it's just that the kernel is flipped
    image_result = cross_correlation(img, np.flip(kernel))

    print('Convolution process DONE!')

    return image_result
    

# ----------------------------------------------------------------------------------------------------------------------------- GAUSSIAN BLUR
def gaussian_blur(sigma, height, width):    # Implement Gaussian Blur to return a kernel, then pass to Convolution along with an image
    
    print('Generating gaussian kernel...')

    kernel = np.zeros((height, width))

    # Center
    center_x = width // 2
    center_y = height // 2
    
    # Generate kernel
    for i in range(height):
        for j in range(width):
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - center_y)**2 + (j - center_x)**2) / (2 * sigma**2))

    # Normalize values so that sum is 1.0
    kernel = kernel / np.sum(kernel)

    print('Gaussian kernel generated!')

    return kernel


# ----------------------------------------------------------------------------------------------------------------------------- LOW PASS
def low_pass(img, sigma, size):             # using Gaussian blurring, implement High Pass and Low Pass filters
    
    print('Generating low pass image...')

    # Either convolution or correlation using Gaussian kernel will do 
    # Generate kernel
    kernel = gaussian_blur(sigma, size, size)

    print('Low pass image generated!')

    return cross_correlation(img, kernel)

def high_pass(img, sigma, size):            # Original image - Low pass image
    
    print('Generating high pass image...')

    # Generate low pass image
    low_pass_image = low_pass(img, sigma, size)

    print('High pass image generated!')

    return img - low_pass_image


# ----------------------------------------------------------------------------------------------------------------------------- HYBRID IMAGE
def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''

    print('Generating hybrid image...')
    
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor

    print('Hybrid image generated!')

    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------------------------- HELPER FUNCTIONS
def save_image(title, image):
    cv2.imwrite(title, image)

def preview_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_parameters(sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio):

    if high_low1 == 'low':
        frequencies = "Image 1's lower frequencies are used."
    else:
        frequencies = "Image 1's higher frequencies are used."

    with open('parameters.txt', 'w') as f:
        f.write("PARAMETERS:\n")
        f.write(f"Sigma1: {sigma1}\n")
        f.write(f"Size1: {size1}\n")
        f.write(f"Frequencies from Image 1: {high_low1}\n")
        f.write(f"Sigma2: {sigma2}\n")
        f.write(f"Size2: {size2}\n")
        f.write(f"Frequencies from Image 2: {high_low2}\n")
        f.write(f"Mix-in Ratio: {mixin_ratio}\n")
        f.write(f"{frequencies}\n")


# ----------------------------------------------------------------------------------------------------------------------------- TESTING
def test_runs(image_1, image_2, mean_filter_kernel):

    # Test Cross-Correlation
    image_result_1 = cross_correlation(image_1, mean_filter_kernel)
    save_image('1_correlation_result.png', image_result_1)

    # Test Convolution
    image_result_2 = convolution(image_1, mean_filter_kernel)
    save_image('2_convolution_result.png', image_result_2)

    # Test Gaussian Blur
    gaussian_kernel = gaussian_blur(3, 5, 5)       # (sigma, height, width)
    image_result_3 = convolution(image_1, gaussian_kernel)
    save_image('3_gaussian_result.png', image_result_3)

    # Test Low Pass
    image_result_4 = low_pass(image_1, 3, 5)
    save_image('4_low_pass_result.png', image_result_4)

    # Test High Pass
    image_result_5 = high_pass(image_1, 3, 5)
    save_image('5_high_pass_result.png', image_result_5)


# ----------------------------------------------------------------------------------------------------------------------------- MAIN
def main():

    # Load the images
    image_1 = cv2.imread('sample-images/Lescano_lab02_left.png')
    image_2 = cv2.imread('sample-images/Lescano_lab02_right.png')

    # Preview of the images
    preview_image('Cow', image_1)           # height = 853px, width = 853px
    preview_image('Sheep', image_2)

    # Mean Filter Kernel
    mean_filter_kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    # Testing
    # test_runs(image_1, image_2, mean_filter_kernel)

    # Define parameters for generating hybrid image
    sigma1 = 5
    size1 = 50
    high_low1 = 'high'          # High pass

    sigma2 = 10
    size2 = 10
    high_low2 = 'low'           # Low Pass

    mixin_ratio = 0.5           # Proportion of each input image that contributes to the final hybrid image
    scale_factor = 1.0          # Can be used to adjust the overall brightness or contrast of the resulting image

    # Save parameters into a README file
    save_parameters(sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio)

    # Generate hybrid image
    hybrid_image = create_hybrid_image(image_1, image_2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)
    save_image('output/Lescano_lab02_hybrid.png', hybrid_image)

    preview_image('Hybrid Image', hybrid_image)


if __name__ == "__main__":
    main()