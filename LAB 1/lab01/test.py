from PIL import Image

def divide_horizontally(image):
    width, height = image.size
    half_height = height // 2
    strip_height = height // 10  # Divide into 10 strips
    
    strips = []
    for i in range(10):
        box = (0, i * strip_height, width, (i + 1) * strip_height)
        strip = image.crop(box)
        strips.append(strip)
    
    return strips

def divide_vertically(image):
    width, height = image.size
    half_width = width // 2
    strip_width = width // 10  # Divide into 10 strips
    
    strips = []
    for i in range(10):
        box = (i * strip_width, 0, (i + 1) * strip_width, height)
        strip = image.crop(box)
        strips.append(strip)
    
    return strips

def assemble_alternate(strips):
    assembled_image = Image.new("RGB", (strips[0].width * len(strips), strips[0].height))
    for i, strip in enumerate(strips):
        if i % 2 == 0:
            assembled_image.paste(strip, (i * strip.width, 0))
    return assembled_image

def merge_images(image1, image2):
    merged_image = Image.new("RGB", (image1.width + image2.width, image1.height))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (image1.width, 0))
    return merged_image

# Load the input image
input_image = Image.open("MyPic.png")

# Step 3.1: Divide the image horizontally into equally-spaced strips
horizontal_strips = divide_horizontally(input_image)

# Step 3.2: Assemble into two images by taking every other strip to form one image
first_assembled_image = assemble_alternate(horizontal_strips)

# Step 3.3: Merge the two images
merged_image = merge_images(input_image, first_assembled_image)

# Step 3.4: Divide the merged image vertically into equally-spaced strips
vertical_strips = divide_vertically(merged_image)

# Step 3.5: Assemble into two images again by taking every other strip to form one image
second_assembled_image = assemble_alternate(vertical_strips)

# Step 3.6: Merge the two images
final_image = merge_images(merged_image, second_assembled_image)

# Save or display the final image
final_image.save("output_image.jpg")
final_image.show()
