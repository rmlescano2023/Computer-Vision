# Computer Vision Laboratory by Reah Mae Sardañas



import numpy as np
import cv2

# ------MAIN FUNCTION------------------------------------------------------------------------------------
def main():
    # Load images
    img1 = cv2.imread("input/dog.png")
    img2 = cv2.imread("input/cat2.png")

    # Get user input for sigma1 and size1 as a single array separated by comma
    print("\nENTER THE NEEDED VALUES (separated by commas)")

    first_pic = [float(x) if i == 0 
                    else x for i,
                         x in enumerate(input("Sigma, Size, and Frequency(low/high) of the 1ST IMAGE: ").split(','))]

    # Assign the values to sigma1, size1, and high_low1
    sigma1 = first_pic[0]
    size1 = int(first_pic[1])
    high_low1 = first_pic[2].lower()

    second_pic = [float(x) if i == 0 
                    else x for i,
                         x in enumerate(input("Sigma, Size, and Frequency(low/high) of the 2ND IMAGE: ").split(','))]

    # Assign the values to sigma2, size2, and high_low2
    sigma2 = second_pic[0]
    size2 = int(second_pic[1])
    high_low2 = second_pic[2].lower()
    
    # Ask the user if they want to grayscale the images
    grayscale_option = input("\nDo you want to grayscale the images? (yes/no): ").lower()
    if grayscale_option == "yes":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Parameters
    mixin_ratio = 0.5
    scale_factor = 1.0  # Scale factor to map the image back to [0, 255]

    
    # Determine which image's frequencies are used
    if high_low1 == 'low':
        used_frequencies = "Image 1's lower frequencies are used."
    else:
        used_frequencies = "Image 1's higher frequencies are used."

    # Write the parameters to a .txt file
    with open('Sardañas_lab02_README.txt', 'w') as f:
        f.write("High Pass and Low Pass Filter Parameters:\n")
        f.write(f"Kernel Size for Image 1: {size1}\n")
        f.write(f"Kernel Sigma for Image 1: {sigma1}\n")
        f.write(f"Used frequencies from Image 1: {high_low1}\n")
        f.write(f"Kernel Size for Image 2: {size2}\n")
        f.write(f"Kernel Sigma for Image 2: {sigma2}\n")
        f.write(f"Used frequencies from Image 2: {high_low2}\n")
        f.write(f"Mix-in Ratio: {mixin_ratio}\n")
        f.write(f"{used_frequencies}\n")

    print("\nFile 'Sardañas_lab02_README.txt' created successfully.")

    # Create hybrid image
    hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                                     high_low2, mixin_ratio, scale_factor)


    print("Hybrid processing DONE!")
    # Display or save your hybrid image here
    cv2.imshow('Hybrid Image', hybrid_img)
    # Save the hybrid image
    cv2.imwrite("output/hybrid.png", hybrid_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio, scale_factor):
    print("\nProcessing Images...")

     # Preprocess images if necessary
    if isinstance(img1, np.ndarray) and img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    #  # Resize images to have the same dimensions
    # img1_height, img1_width = img1.shape[:2]
    # img2 = cv2.resize(img2, (img1_width, img1_height))
    
    # Apply filters according to user specifications
    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    # Blend images using the mixing ratio
    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor

     # Clip and convert the image to uint8 if necessary
    if isinstance(hybrid_img, np.ndarray) and hybrid_img.dtype == np.float32:
        hybrid_img = (hybrid_img * 255).clip(0, 255).astype(np.uint8)

    return hybrid_img

def low_pass(img, sigma, size):
    # Generate Gaussian blur kernel
    kernel = gaussian_blur(sigma, size)
    
    # Perform cross-correlation of the image with the Gaussian blur kernel
    return convolution(img, kernel)

def high_pass(img, sigma, size):
    # Get the low-pass filtered image
    low_passed_img = low_pass(img, sigma, size)
    
    # Compute the high-pass filtered image
    return img - low_passed_img

def gaussian_blur(sigma, size):
    # Create Gaussian kernel
    center = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize the kernel

    return kernel

def convolution(img, kernel):
    # Ensure kernel dimensions are odd
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    # Use the cross_correlation function to perform convolution
    return cross_correlation(img, np.flipud(np.fliplr(kernel)))   

def cross_correlation(img, kernel):
    # Check if the image is grayscale or color and get its dimensions and kernel

    if len(img.shape) == 2:  # Grayscale image
        img_height, img_width = img.shape
        channels = 1
    elif len(img.shape) == 3:  # Color image
        img_height, img_width, channels = img.shape
    else:
        raise ValueError("Invalid image shape.")


    kernel_height, kernel_width = kernel.shape

    # Padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2


     # Initialize an empty output image
    output = np.zeros_like(img)

    if channels == 1:  # Grayscale image
        # Add padding to the image
        img_padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Iterate over each pixel in the image
        for y in range(img_height):
            for x in range(img_width):
                # Perform cross-correlation
                output[y, x] = np.sum(img_padded[y:y + kernel_height, x:x + kernel_width] * kernel)

    elif channels == 3:  # Color image
        # Add padding to each channel of the image
        img_padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

        # Iterate over each pixel in the image
        for y in range(img_height):
            for x in range(img_width):
                # Perform cross-correlation separately for each channel
                for c in range(channels):
                    output[y, x, c] = np.sum(img_padded[y:y + kernel_height, x:x + kernel_width, c] * kernel)

    return output


if __name__ == "__main__":
    main()
