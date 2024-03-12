gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    for i, level in enumerate(gaussian_pyr_2):
        cv2.imshow(f'Gaussian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    for i, level in enumerate(laplacian_pyr_2):
        cv2.imshow(f'Laplacian level {i}', level)
        cv2.waitKey(0)
        cv2.destroyAllWindows()