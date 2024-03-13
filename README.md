# Computer Vision

### This is a repository for Computer Vision (CMSC 174) laboratory exercises using OpenCV Python.

<br>

## LAB 1

In this laboratory exercise, you are given a square image. Then, create a program that replicates the process as shown in this video: https://fb.watch/pWLNqOIQPE/

That is, 
* 3.1 divide the image horizontally into equally-spaced strips
* 3.2 assemble into two images by taking every other strip to form one image
* 3.3 merge the two images
* 3.4 divide the merged image vertically into equally-spaced strips
* 3.5 assemble into two images again by taking every other strip to form one image
* 3.6 merge the two 

<br>

## LAB 2

The goal of this laboratory exercise is to write an image filtering function and use it to create hybrid images using a simplified version of the SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances.

You will use your own solution to create your own hybrid images.

The provided file hybrid.py contains functions that you need to implement.

### Implementation Details 

This laboratory exercise is intended to familiarize you with Python, NumPy and image filtering. Once you have created an image filtering function, it is relatively straightforward to construct hybrid images.

This project requires you to implement 5 functions each of which builds onto a previous function:
* cross_correlation
* convolution
* gaussian_blur
* low_pass
* high_pass

### Image Filtering

Image filtering (or convolution) is a fundamental image processing tool. See chapter 3.2 of Szeliski.
Numpy has numerous built in and efficient functions to perform image filtering, but you will be writing your own such function from scratch for this assignment. More specifically, you will implement cross_correlation_2d, followed by convolve_2d which would use cross_correlation_2d.

### Gaussian Blur

There are a few different way to blur an image, for example taking an unweighted average of the neighboring pixels. Gaussian blur is a special kind of weighted averaging of neighboring pixels. To implement Gaussian blur, you will implement a function gaussian_blur_kernel_2d that returns a kernel of a given height and width which can then be passed to convolve_2d from above, along with an image, to produce a blurred version of the image.

### High and Low Pass Filters

Recall that a low pass filter is one that removed the fine details from an image (or, really, any signal), whereas a high pass filter only retains the fine details, and gets rid of the coarse details from an image. 
Thus, using Gaussian blurring as described above, implement high_pass and low_pass functions.

### Hybrid Images

A hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image. 
There is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. This is called the "cutoff-frequency". In the paper it is suggested to use two cutoff frequencies (one tuned for each image) and you are free to try that, as well. In the starter code, the cutoff frequency is controlled by changing the standard deviation (sigma) of the Gausian filter used in constructing the hybrid images.

### Forbidden functions

For just this laboratory exercise, you are forbidden from using any Numpy, Scipy, OpenCV, or other preimplemented functions for filtering. You are allowed to use basic matrix operations like np.shape, np.zeros, and np.transpose. This limitation will be lifted in future laboratory exercises , but for now, you should use for loops or Numpy vectorization to apply a kernel to each pixel in the image. The bulk of your code will be in cross_correlation_2d, and gaussian_blur_kernel_2d with the other functions using these functions either directly or through one of the other functions you implement.

Your pair of images needs to be aligned using an image manipulation software, e.g., Photoshop, Gimp. 
Alignments can map the eyes to eyes and nose to nose, edges to edges, etc. It is encouraged to create additional examples (e.g. change of expression, morph between different objects, change over time, etc.). See the hybrid images project page (http://olivalab.mit.edu/hybrid_gallery/gallery.html) for some inspiration. The project page also contains materials from their Siggraph presentation.

<br>

## LAB 3

The goal of this laboratory exercise is to blend two images seamlessly using a multi resolution blending as described in the 1983 paper* by Burt and Adelson. An image spline is a smooth seam joining two image together by gently distorting them. Multiresolution blending computes a gentle seam between the two images seperately at each band of image frequencies, resulting in a much smoother seam. You are to create and visualize the Gaussian and Laplacian stacks of the input images, blending together images with the help of the completed stacks, and explore creative outcomes.

You will implement Gaussian and Laplacian stacks, which are kind of like pyramids but without the downsampling. This will prepare you for the next step for Multi-resolution blending.

### Details

Implement a Gaussian and a Laplacian stack. The difference between a stack and a pyramid is that in each level of the pyramid the image is downsampled, so that the result gets smaller and smaller. In a stack, the images are never downsampled so the results are all the same dimension as the original image, and can all be saved in one 3D matrix (if the original image was a grayscale image). To create the successive levels of the Gaussian stack, just apply the Gaussian filter at each level, but do not subsample. In this way we will get a stack that behaves similarly to a pyramid that was downsampled to half its size at each level. If you would rather work with pyramids, you may implement pyramids other than stacks. However, in any case, you are NOT allowed to use existing pyramid (pyrDown, pyrUp) functions. You must implement your stacks from scratch.
1. Apply your Gaussian and Laplacian stacks to your input pair of images to recreate the outcomes similar to Figure 3.42 in Szelski (2nd ed).
2. Using another pair of images, blend them together using some crazy ideas of your own. See Figure 8 of the 1983 paper by Burt and Adelson for some inspiration. You should always use an irregular mask and also, create a Gaussian stack for your mask image as well as for the two input images. The Gaussian blurring of the mask in the pyramid will smooth out the transition between the two images.

<br>

## LAB 4

The goal of this laboratory exercise is to estimate the amount of liquid contained in a bottle.

The accompanying directories (https://drive.google.com/drive/folders/1rLQPUpJejYdw77dnDq5RdISKF-ndL0lY?usp=sharing)
contain the images of the bottle with a specified amount of contained liquid.

Thus, the directory 50mL contains pictures of the bottle with 50 mL liquid in it. And so on...
The directory 'guess' contains images of the bottle with unknown amounts of liquid. You are to guess these amounts.

OpenCV image filtering, thresholding, or morphology operations are allowed.

Hints:
Count the number of pixels corresponding to the liquid and have it mapped to the labelled amount.
OR
Compute the area (or some other measure) of the region occupied by the liquid in the image and have it mapped to the labelled amount.


Use interpolation or regression to guess the unknown amounts.



<br>

## LAB 5

### Image Stitching
(individual or groups of two)

1. Read https://medium.com/@paulsonpremsingh7/image-stitching-using-opencv-a-step-by-step-tutorial-9214aa4255ec

2. Using the code in #1 as basis, stitch the images in the directory named 'data'.

3. Using the video named 'spike.mp4' in directory 'data', generate an actionshot image. 

Actionshot is a method of capturing an object in action and displaying it in a single image with multiple sequential appearances of the object.
Extra credits for using your own video for the actionshot image.
(doing the cartwheel, running forehand in tennis, grand jeté in ballet, somersault in acrobatics or diving, forward flip in skateboarding)
SAFETY FIRST. Be sure you know what you are doing should you embark in this adventure.



<br>

## LAB 6

### Do-it-yourself (DIY) Pinhole Camera
(individual or groups of two or three)

Create a classroom model pinhole camera.