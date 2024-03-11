Read SZ 3.5.5 Application: Image blending

The goal of this laboratory exercise is to blend two images seamlessly using a multi resolution blending as described in the 1983 paper* by Burt and Adelson. An image spline is a smooth seam joining two image together by gently distorting them. Multiresolution blending computes a gentle seam between the two images seperately at each band of image frequencies, resulting in a much smoother seam. You are to create and visualize the Gaussian and Laplacian stacks of the input images, blending together images with the help of the completed stacks, and explore creative outcomes.

*Burt, P. J. and Adelson, E. H. (1983b). A multiresolution spline with applications to image mosaics. ACM Transactions on Graphics, 2(4):217–236.

Overview
You will implement Gaussian and Laplacian stacks, which are kind of like pyramids but without the downsampling. This will prepare you for the next step for Multi-resolution blending.

Details
Implement a Gaussian and a Laplacian stack. The difference between a stack and a pyramid is that in each level of the pyramid the image is downsampled, so that the result gets smaller and smaller. In a stack, the images are never downsampled so the results are all the same dimension as the original image, and can all be saved in one 3D matrix (if the original image was a grayscale image). To create the successive levels of the Gaussian stack, just apply the Gaussian filter at each level, but do not subsample. In this way we will get a stack that behaves similarly to a pyramid that was downsampled to half its size at each level. If you would rather work with pyramids, you may implement pyramids other than stacks. However, in any case, you are NOT allowed to use existing pyramid (pyrDown, pyrUp) functions. You must implement your stacks from scratch.

1. Apply your Gaussian and Laplacian stacks to your input pair of images to recreate the outcomes similar to Figure 3.42 in Szelski (2nd ed).

2. Using another pair of images, blend them together using some crazy ideas of your own. See Figure 8 of the 1983 paper by Burt and Adelson for some inspiration. You should always use an irregular mask and also, create a Gaussian stack for your mask image as well as for the two input images. The Gaussian blurring of the mask in the pyramid will smooth out the transition between the two images.

Extra points (Optional)
Try using color to enhance the blend effects.


Submission
LastName_lab03_blending.py
LastName_lab03_left.png,LastName_lab03_right.png: Submit the left and right images you used to create the vertically blended image.
LastName_lab03_blendvert.png: Submit the blended image produced by using your implementation on left and right images
LastName_lab03_crazyone.png,LastName_lab03_crazytwo.png: Submit the pair of images you used to create the creatively blended image.
LastName_lab03_blendcrazy.png: Submit the blended image produced by using your implementation on crazyone and crazytwo images
LastName_lab03_README.txt: What is the coolest/most interesting thing you learned from this laboratory exercise?



