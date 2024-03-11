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
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolution(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
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
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

=HINTS====================================
You may find the following code snippets useful

# mean filter kernel
kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9) # 3x3

#Gaussian kernel
size=5
sigma=3
center=int(size/2)
kernel=np.zeros((size,size))
for i in range(size):
	for j in range(size):
          kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))
kernel=kernel/np.sum(kernel)	#Normalize values so that sum is 1.0

#dimensions of the image and the kernel
image_height, image_width = imageGray.shape
kernel_height, kernel_width = ______________________

#Padding
imagePadded = np.zeros((image_height+kernel_height-1,________________________)) # zero-padding scheme, you may opt for other schemes
for i in range(image_height):
	for j in range(image_width):
		imagePadded[i+int((kernel_height-1)/2), j+_____________________] = imageGray[i,j]  #copy Image to padded array


#correlation
for i in range(________):
	for j in range(image_width):
		window = imagePadded[____________________, j:j+kernel_width]
		imageGray[i,j] = np.sum(window*kernel)  #numpy does element-wise multiplication on arrays

#convolution
		#np.flip(kernel)  # flips horizontally and vertically
		#correlation

#low pass filter
	#Either convolution or correlation using Gaussian kernel will do 
	#since Gaussian kernel is all-axis symmetric, either correlation or convolution can be used
	
#high pass filter
	# original image - low pass image

#merge two images (low pass image + high pass image)
	alpha*Image1 + (1-alpha)Image2   # alpha is the amount of blending between the two images
