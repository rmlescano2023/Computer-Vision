import cv2
import numpy as np

def gaussian_stack(image, levels):
    gaussian_images = [image.astype(np.float32)]
    for i in range(levels):
        image = cv2.GaussianBlur(image, (5, 5), 0)
        gaussian_images.append(image)
    return gaussian_images

def laplacian_stack(gaussian_images):
    laplacian_images = []
    for i in range(len(gaussian_images) - 1):
        expanded = cv2.pyrUp(gaussian_images[i + 1], dstsize=(gaussian_images[i].shape[1], gaussian_images[i].shape[0]))
        laplacian = cv2.subtract(gaussian_images[i], expanded)
        laplacian_images.append(laplacian)
    laplacian_images.append(gaussian_images[-1])  # The last image is just the last Gaussian level
    return laplacian_images

def blend_images(image1, image2, mask):
    blended_image = (image1 * (1 - mask)) + (image2 * mask)
    return blended_image.astype(np.uint8)

def main():
    # Load input images
    image1 = cv2.imread('examples/apple.jpg')
    image2 = cv2.imread('examples/orange.jpg')

    # Convert images to float32
    image1_float32 = image1.astype(np.float32)
    image2_float32 = image2.astype(np.float32)

    # Number of levels for the stack
    levels = 5

    # Create Gaussian stacks for both images
    gaussian_stack1 = gaussian_stack(image1_float32, levels)
    gaussian_stack2 = gaussian_stack(image2_float32, levels)

    # Create Laplacian stacks for both images
    laplacian_stack1 = laplacian_stack(gaussian_stack1)
    laplacian_stack2 = laplacian_stack(gaussian_stack2)

    # Blend the images using Laplacian stacks
    blended_stack = []
    for lap1, lap2 in zip(laplacian_stack1, laplacian_stack2):
        mask = np.zeros_like(lap1)
        mask[:, :lap1.shape[1]//2] = 1  # Simple mask splitting the images in half
        blended = blend_images(lap1, lap2, mask)
        blended_stack.append(blended)

    # Reconstruct the blended image
    blended_image = blended_stack[0]
    for i in range(1, levels + 1):
        expanded = cv2.pyrUp(blended_image, dstsize=(blended_stack[i].shape[1], blended_stack[i].shape[0]))
        blended_image = cv2.add(expanded, blended_stack[i])

    # Visualize the outcomes
    cv2.imshow('Blended Image', blended_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
