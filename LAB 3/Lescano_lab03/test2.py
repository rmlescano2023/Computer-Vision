import cv2
import numpy as np

from Lescano_lab03_blending import *

# Step-2
# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr

# Step-3
# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

# Step-4
# Now blend the two images wrt. the mask
def blend(laplacian_A,laplacian_B,mask_pyr):
    LS = []
    for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS

# Step-5
# Reconstruct the original image
def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst

# Now let's call all these functions
if __name__ == '__main__':
    # Step-1
    # Load the two images
    img1 = cv2.imread('examples/Lescano_lab03_crazyone.png')
    img1 = cv2.resize(img1, (1800, 1000))
    img2 = cv2.imread('examples/Lescano_lab03_crazytwo.png')
    img2 = cv2.resize(img2, (1800, 1000))
    mask = cv2.imread('examples/mask.png')
    mask = cv2.resize(mask, (1800, 1000))

    """ cv2.imshow('Left', img1)
    cv2.imshow('Right', img2)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    num_levels = 7
    
    # For image-1, calculate Gaussian and Laplacian
    """ gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1) """

    # Try using my program
    gaussian_pyr_1 = generate_gaussian_pyramid(img1, num_levels, GAUSSIAN_KERNEL_SIZE)
    laplacian_pyr_1 = generate_laplacian_pyramid(gaussian_pyr_1, num_levels)

    """ for i, level in enumerate(gaussian_pyr_1):
        cv2.imshow(f'Gaussian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    for i, level in enumerate(laplacian_pyr_1):
        cv2.imshow(f'Laplacian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """

    # For image-2, calculate Gaussian and Laplacian
    # gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    # laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    # Try using my program
    gaussian_pyr_2 = generate_gaussian_pyramid(img2, num_levels, GAUSSIAN_KERNEL_SIZE)
    laplacian_pyr_2 = generate_laplacian_pyramid(gaussian_pyr_2, num_levels)

    """ for i, level in enumerate(gaussian_pyr_2):
        cv2.imshow(f'Gaussian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    for i, level in enumerate(laplacian_pyr_2):
        cv2.imshow(f'Laplacian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """

    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()

    """ for i, level in enumerate(mask_pyr_final):
        cv2.imshow(f'Gaussian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """
    

    # Blend the images
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)

    # Reconstruct the images
    final  = reconstruct(add_laplace)

    # Save the final image to the disk
    cv2.imwrite('output/crazyblend.png', final[num_levels])