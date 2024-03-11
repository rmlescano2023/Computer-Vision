import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def divide_into_strips(image, num_strips):
    height, width, _ = image.shape
    strip_height = height // num_strips
    
    strips = []
    for i in range(num_strips):
        strip = image[i * strip_height : (i + 1) * strip_height, :, :]
        strips.append(strip)
    
    return strips

def assemble_alternate(strips):
    assembled_image1 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 == 0], axis=0)
    assembled_image2 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 != 0], axis=0)
    return assembled_image1, assembled_image2

# Load the input image
input_image = mpimg.imread("MyPic.png")

# Specify the number of strips to divide the image into
num_strips = 50  # Change this value as needed

# Divide the image into equally spaced strips
strips = divide_into_strips(input_image, num_strips)

# Save each strip as a separate image
for i, strip in enumerate(strips):
    output_filename = f"LAB1_Outputs\strip_{i + 1}.jpg"
    plt.imsave(output_filename, strip)

# Assemble every other strip into two images
assembled_image1, assembled_image2 = assemble_alternate(strips)

print(f"{num_strips} strips saved successfully.")

# Save the assembled images
plt.imsave("assembled_image1.jpg", assembled_image1)
plt.imsave("assembled_image2.jpg", assembled_image2)