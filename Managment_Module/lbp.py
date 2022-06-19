import cv2
import numpy as np

'''
    This is a function used to calculate local binary pattern
    from scratch for a given image.

    Input:  @param image: image to find the hist on
            @param kernel_size: lbp local window size.

    Output: lbp_image
'''


def local_binary_pattern(image, kernel_size=3):

    # start allocation for lbp image
    lbp_image = np.zeros_like(image)
    kernel_size = kernel_size
    center_pixel = int(kernel_size // 2)
    for ih in range(0, image.shape[0] - kernel_size):
        for iw in range(0, image.shape[1] - kernel_size):

            # move the kernel along the image.
            img = image[ih:ih + kernel_size, iw:iw + kernel_size]

            # simple filter that only leaves out the ones bigger than the center pixel.
            center = img[center_pixel, center_pixel]
            # 3 * 3, processed kernel.
            filtered_kernel = (img >= center) * 1.0

            flat_kernel = filtered_kernel.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()

            # remove the center, e.g the 5th element of a size 9 kernel (3 * 3).
            flat_kernel = np.delete(flat_kernel, 4)
            # example: [1. 0. 0. 1. 0. 1. 1. 1.]

            non_zero_locations = np.where(flat_kernel)[0]
            if len(non_zero_locations) >= 1:
                num = np.sum(2 ** non_zero_locations)
            else:
                num = 0
            # adjust the center value.
            lbp_image[ih + center_pixel, iw + center_pixel] = num
    return lbp_image
