import cv2
import numpy as np

from Lescano_lab03_blending import *

# Step-2    SAME WITH ORIGINAL
# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, num_levels):

    gaussian_pyr = [img]          # index 0 is the copy of the original image

    for i in range(num_levels):
        img = gaussian_blur(gaussian_pyr[-1], GAUSSIAN_KERNEL_SIZE)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        gaussian_pyr.append(img)

    return gaussian_pyr

# Step-3
# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):

    laplacian_top = gaussian_pyr[-1]        # the last image in the gaussian_pyr, the blurriest one

    num_levels = len(gaussian_pyr) - 1      # num_levels = 6, so now it's 5
    
    laplacian_pyr = [laplacian_top]         # use 2nd to the last image

    for i in range(num_levels, 0, -1):      #
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.resize(gaussian_pyr[i], size)
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
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
        laplacian_expanded = cv2.resize(laplacian_top, size)
        laplacian_top = cv2.add(laplacian_expanded, laplacian_pyr[i + 1])
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


# Now let's call all these functions
if __name__ == '__main__':
    # Step-1
    # Load the two images
    img1 = cv2.imread('examples/sky.png')
    img1 = cv2.resize(img1, (1800, 1000))
    img2 = cv2.imread('examples/plane.jpg')
    img2 = cv2.resize(img2, (1800, 1000))

    # Create the mask
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[250:500,640:1440,:] = (1,1,1)
    
    num_levels = 6
    


    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)

    # Visualize
    # visualize_pyramid(gaussian_pyr_1, 'Gaussian One')
    # visualize_pyramid(laplacian_pyr_1, 'Laplacian One')




    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    # Visualize
    # visualize_pyramid(gaussian_pyr_2, 'Gaussian Two')
    # visualize_pyramid(laplacian_pyr_2, 'Laplacian Two')





    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()

    # Blend the images
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)

    # Reconstruct the images
    final  = reconstruct(add_laplace)

    # Save the final image to the disk
    cv2.imwrite('output/crazyblend_test_3.png', final[num_levels])