import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# -------------------------------------------------------------------------------------------------------------- FUNCTIONS

def divide_into_horizontal_strips(image, num_strips):

    height, width, _ = image.shape              # get the height and width of the image
    strip_height = height // num_strips         # calculate the height of each strip
    
    strips = []
    for i in range(num_strips):
        strip = image[i * strip_height : (i + 1) * strip_height, : , :]      # array slicing, syntax -> image[start index : stop index, : , :]
        strips.append(strip)                                                # the first parameter is the row
    
    return strips


def divide_into_vertical_strips(image, num_strips):

    height, width, _ = image.shape              # get the height and width of the image
    strip_width = width // num_strips           # calculate the width of each strip
    
    strips = []
    for i in range(num_strips):
        strip = image[: , i * strip_width : (i + 1) * strip_width, :]         # array slicing, syntax -> image[: , start index : stop index, :]
        strips.append(strip)
    
    return strips


def assemble_horizontal_alternate(strips):      # along axis 0, or the vertical axis
    assembled_horizontal_image1 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 == 0], axis=0)     # concatenates even indexes
    assembled_horizontal_image2 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 != 0], axis=0)     # concatenates odd indexes
    return assembled_horizontal_image1, assembled_horizontal_image2


def assemble_vertical_alternate(strips):        # along axis 1, or the horizontal axis
    assembled_vertical_image1 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 == 0], axis=1)       # concatenates even indexes
    assembled_vertical_image2 = np.concatenate([strips[i] for i in range(len(strips)) if i % 2 != 0], axis=1)       # concatenates odd indexes
    return assembled_vertical_image1, assembled_vertical_image2


def merge_images_side_by_side(image1, image2):
    merged_image = np.concatenate((image1, image2), axis=1)
    return merged_image



# -------------------------------------------------------------------------------------------------------------- MAIN

image = mpimg.imread("MyPic.png")

# The number of strips to divide the image into
num_strips = 50


# >>>>> 3.1 Divide the image horizontally into equally-spaced strips
strips = divide_into_horizontal_strips(image, num_strips)

# Save each strip as an image
for i, strip in enumerate(strips):
    plt.imsave(f"horizontal_strip_{i + 1}.jpg", strip)


# >>>>> 3.2 Assemble into two images by taking every other strip to form one image
assembled_horizontal_image1, assembled_horizontal_image2 = assemble_horizontal_alternate(strips)

# Save the assembled images
plt.imsave("assembled_horizontal_image1.jpg", assembled_horizontal_image1)
plt.imsave("assembled_horizontal_image2.jpg", assembled_horizontal_image2)


# >>>>> 3.3 Merge the two images
merged_image1 = merge_images_side_by_side(assembled_horizontal_image1, assembled_horizontal_image2)

# Save the merged image
plt.imsave("merged_image1.jpg", merged_image1)


# >>>>> 3.4 Divide the merged image vertically into equally-spaced strips
vertical_strips = divide_into_vertical_strips(merged_image1, num_strips)

# Save each vertical strip as an image
for i, strip in enumerate(vertical_strips):
    plt.imsave(f"vertical_strip_{i + 1}.jpg", strip)


# >>>>> 3.5 Assemble into two images again by taking every other strip to form one image
assembled_vertical_image1, assembled_vertical_image2 = assemble_vertical_alternate(vertical_strips)

# Save the assembled images
plt.imsave("assembled_vertical_image1.jpg", assembled_vertical_image1)
plt.imsave("assembled_vertical_image2.jpg", assembled_vertical_image2)


# >>>>> 3.6 Merge the two images
merged_vertical_image = merge_images_side_by_side(assembled_vertical_image1, assembled_vertical_image2)

# Save the merged vertical image
plt.imsave("merged_vertical_image.jpg", merged_vertical_image)


plt.imshow(merged_vertical_image)
plt.axis('off')  # Turn off axis
plt.show()