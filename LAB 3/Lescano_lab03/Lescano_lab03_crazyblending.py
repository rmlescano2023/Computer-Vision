from Lescano_lab03_blending import *


# --------------------------------------------------------------------------------------------------------------- MAIN
def main():

    # Load the images
    image_1 = cv2.imread('examples/Lescano_lab03_crazyone.png')
    image_2 = cv2.imread('examples/Lescano_lab03_crazytwo.png')
    mask_image = cv2.imread('examples/mask.png')

    # Convert the image to grayscale
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Image preview
    preview_images(gray_image_1, gray_image_2)
    cv2.imshow("Mask", mask_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # Testing
    img1_copy = image_1.copy()
    img2_copy = image_2.copy()
    mask_copy = mask_image.copy()

    img2_minus_mask = cv2.subtract(mask_copy, img2_copy)
    cv2.imshow("Test", img2_minus_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Generate Gaussian pyramid of all the images
    img1_gaussian_pyramid = generate_gaussian_pyramid(gray_image_1, levels, kernel_size)
    img2_gaussian_pyramid = generate_gaussian_pyramid(gray_image_2, levels, kernel_size)
    mask_gaussian_pyramid = generate_gaussian_pyramid(gray_mask, levels, kernel_size)

    # Generate Laplacian pyramid of all the images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid, levels)
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid, levels)
    mask_laplacian_pyramid = generate_laplacian_pyramid(mask_gaussian_pyramid, levels)

    # Reverse contents of the mask array
    mask_gaussian_pyramid.reverse()

    # Visualize Gaussian & Laplacian pyramids
    # visualize_pyramid(img1_gaussian_pyramid, pyramid_type="Gaussian")
    # visualize_pyramid(img2_gaussian_pyramid, pyramid_type="Gaussian")
    # visualize_pyramid(mask_gaussian_pyramid, pyramid_type="Gaussian")
    # visualize_pyramid(img1_laplacian_pyramid, pyramid_type="Laplacian")
    # visualize_pyramid(img2_laplacian_pyramid, pyramid_type="Laplacian")
    # visualize_pyramid(mask_laplacian_pyramid, pyramid_type="Laplacian")

    # Blend images using the mask
"""     image_pyramid = []
    for left_img, right_img, mask_img in zip(img1_laplacian_pyramid, img2_laplacian_pyramid, mask_gaussian_pyramid):
        
        result = mask_img * left_img
        # result = mask_img * left_img + (1.0 - mask_img) * right_img

        image_pyramid.append(result)

    visualize_pyramid(image_pyramid, pyramid_type="Pre-blending") """

"""     blended_image = blend_images(image_pyramid, levels)
    show_blended_image(blended_image) """


""" 
    # Reconstruct the blended image
    blended_image = image_pyramid[0]
    for i in range(1, 6):
        blended_image = cv2.resize(blended_image, (blended_image[i].shape[1], blended_image[i].shape[0]))
        blended_image += image_pyramid[i]

    # Display or save the result
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 """
if __name__ == '__main__':
    main()