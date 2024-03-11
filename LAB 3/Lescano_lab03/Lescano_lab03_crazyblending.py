from Lescano_lab03_blending import *


def pyramids(image, mask=False, reconstruct=False):
	kernal = np.array(((1.0/256, 4.0/256,  6.0/256,  4.0/256,  1.0/256), (4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256), (6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256), (4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256), (1.0/256, 4.0/256,  6.0/256,  4.0/256,  1.0/256)))
	if not reconstruct:
		G, L = [image], []
		while image.shape[0] >= 2 and image.shape[1] >= 2:
			image = scipy.signal.convolve2d(image, kernal, mode='same', fillvalue=1)[::2, ::2]
			G.append(image)
		for i in range(len(G) - 1):
			next_img = np.zeros((2 * G[i + 1].shape[0], 2 * G[i + 1].shape[1]))
			next_img[::2, ::2] = G[i + 1]
			L.append(G[i] - scipy.signal.convolve2d(next_img, 4 * kernal, mode='same', fillvalue=1))
		return G[:-1] if mask else L
	else:
		for i in range(len(image)):
			for j in range(i):
				next_img = np.zeros((2 * image[i].shape[0], 2 * image[i].shape[1]))
				next_img[::2, ::2] = image[i]
				image[i] = scipy.signal.convolve2d(next_img, 4 * kernal, mode='same', fillvalue=1)
		tot_sum = np.sum(image, axis=0)
		tot_sum[tot_sum < 0.0] = 0.0
		tot_sum[tot_sum > 255.0] = 255.0
		return tot_sum
	
# Helper function, follows formula described in paper
def blend_pyramids(im1_pyramid, im2_pyramid, mask_pyramid):
	blended = []
	for i in range(len(mask_pyramid)):
		blended.append(im1_pyramid[i] * (1.0 - mask_pyramid[i]) + im2_pyramid[i] * mask_pyramid[i])
	return blended

# --------------------------------------------------------------------------------------------------------------- MAIN
def main():

    # Load the images
    image_1 = cv2.imread('examples/Lescano_lab03_crazyone.png')
    image_2 = cv2.imread('examples/Lescano_lab03_crazytwo.png')
    mask_image = cv2.imread('examples/mask.png')

    # Image preview
    preview_images(image_1, image_2)
    cv2.imshow("Mask", mask_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # Define parameters
    levels = PYRAMID_LEVELS
    kernel_size = GAUSSIAN_KERNEL_SIZE

    # Generate Gaussian pyramid of all the images
    img1_gaussian_pyramid = generate_gaussian_pyramid(image_1, levels, kernel_size)
    img2_gaussian_pyramid = generate_gaussian_pyramid(image_2, levels, kernel_size)
    mask_gaussian_pyramid = generate_gaussian_pyramid(mask_image, levels, kernel_size)

    # Generate Laplacian pyramid of all the images
    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid, levels)
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid, levels)
    mask_laplacian_pyramid = generate_laplacian_pyramid(mask_gaussian_pyramid, levels)

    # Visualize Gaussian & Laplacian pyramids
    # visualize_pyramid(img1_gaussian_pyramid)
    # visualize_pyramid(img2_gaussian_pyramid)
    # visualize_pyramid(mask_gaussian_pyramid)
    # visualize_pyramid(img1_laplacian_pyramid)
    # visualize_pyramid(img2_laplacian_pyramid)
    # visualize_pyramid(mask_laplacian_pyramid)

    # Blending
    result = pyramids(blend_pyramids(img1_laplacian_pyramid, img2_laplacian_pyramid, mask_laplacian_pyramid))
    cv2.imshow(result)


if __name__ == '__main__':
    main()