import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import imutils

# ------------------------------------------------------------------------------ GLOBAL VARIABLES
KERNEL_SIZE_BLUR = (7, 7)
THRESHOLD_VALUE = 50
MAX_VALUE = 255
KERNEL_SIZE_MORPH = (5, 5)

# ------------------------------------------------------------------------------ LOAD IMAGES
def load_images(images_folder):

    loaded_images = []

    for img in os.listdir(images_folder):                                         # access each image within each folder
        if img.lower().endswith('.jpg'):
            image_path = os.path.join(images_folder, img)                         # generate the exact path of each image; sample-output: data-lab04/100mL/IMG_20240214_101231.jpg
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is not None:
                image = cv2.resize(image, (500, 500))
                loaded_images.append(image)

    return loaded_images

# ------------------------------------------------------------------------------ PROCESS IMAGES
def process_image(img):

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                            # convert the input image to grayscale

    blurred_image = cv2.GaussianBlur(gray_image, KERNEL_SIZE_BLUR, 0)             # apply Gaussian Blur to the grayscale image to smooth it out and reduce noise

    _, binary_image = cv2.threshold(blurred_image, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY_INV)         # perform inverse binary thresholding

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_MORPH)                        # create a rectangular structuring element for morphological operations

    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, structuring_element)                       # apply morphological opening to the binary image

    contours = cv2.findContours(cleaned_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)             # find contours in the cleaned binary image; extract outlines
    contours = imutils.grab_contours(contours)                                    # extract actual list of contours

    if contours:                                                                  # determine the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)                      # find the largest contour
        x, y, width, height = cv2.boundingRect(largest_contour)                   # calculates the bounding rectangle around the largest contour
        contour_area = width * height                                             # solve for contour area
        return contour_area, height                                               # return the area and height of the largest contour

    return 0, 0

# ------------------------------------------------------------------------------ GUESS VOLUMES
def guess_volumes(guess_paths, volume_predictor):

    num_subfolders = len(guess_paths)

    fig, axs = plt.subplots(1, num_subfolders, figsize=(5 * num_subfolders, 5))   # create array of subplots

    for ax, guess_path in zip(axs, guess_paths):
        guess_images = load_images(guess_path)                                    # load the images from the guess folder
        guess_features = [process_image(img) for img in guess_images]             # extract features every image from the guess folder
        guess_features = np.array(guess_features).reshape(-1, 2)                  # reshape to ensure it's a 2D array

        if guess_images:
            predicted_volumes = volume_predictor.predict(guess_features)          # use the trained model to predict the volumes based on the extracted features
            average_volume = np.mean(predicted_volumes)                           # computes for the average predicted volumes

            ax.imshow(cv2.cvtColor(guess_images[0], cv2.COLOR_BGR2RGB))           # display first image
            ax.set_title(f"Subfolder {os.path.basename(guess_path)}\nAvg. Volume: {average_volume:.2f}mL")
            ax.axis('off')
        else:
            ax.axis('off')
            ax.set_title(f"No images in Subfolder {os.path.basename(guess_path)}")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------ MAIN
def main():
    sample_labels = [
        '50mL',
        '100mL',
        '150mL',
        '200mL',
        '250mL',
        '300mL',
        '350mL'
    ]
    sample_paths = [
        'data-lab04/50mL',
        'data-lab04/100mL',
        'data-lab04/150mL',
        'data-lab04/200mL',
        'data-lab04/250mL',
        'data-lab04/300mL',
        'data-lab04/350mL'
    ]
    guess_paths = [
        'data-lab04/guess/A',
        'data-lab04/guess/B',
        'data-lab04/guess/C'
    ]

    features = []
    volume_labels = []

    for label, path in zip(sample_labels, sample_paths):
        loaded_images = load_images(path)                                         # load images from the current folder
        processed_images = [process_image(img) for img in loaded_images]          # extract features from the images
        features.extend(processed_images)                                         # append to features list for every iteration
        volume_labels.extend([int(label[:-2])] * len(processed_images))           # make sure volume labels are according to the images

    features = np.array(features)                                                 # convert features to a 2D array

    volume_predictor = make_pipeline(StandardScaler(), LinearRegression())
    volume_predictor.fit(features, volume_labels)

    guess_volumes(guess_paths, volume_predictor)

if __name__ == "__main__":
    main()
