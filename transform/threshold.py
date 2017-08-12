import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transform.camera_calibration import undistort


# image = mpimg.imread('../test_images/test5.jpg')


def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def select_white(image):
    lower = np.array([202, 202, 202])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    return mask


def threshold_image(img, s_thresh=(170, 255), l_thresh=(225, 255), b_thresh=(155, 200), sx_thresh=(20, 100), plot=False):
    img = np.copy(img)

    # s channel from HLS space
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # b channel from Lab space
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    # l channel from LUV space
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    white_mask = select_white(img)
    yellow_mask = select_yellow(img)

    combined_binary = np.zeros_like(white_mask)
    combined_binary[((white_mask != 0) | (yellow_mask != 0)) & ((sxbinary == 1) | (b_binary == 1))] = 1

    if plot:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(14, 6))
        f.tight_layout()

        ax1.set_title('Original Image', fontsize=14)
        ax1.imshow(img)

        ax2.set_title('s_binary', fontsize=14)
        ax2.imshow(s_binary, cmap='gray')

        ax3.set_title('l_binary', fontsize=14)
        ax3.imshow(l_binary, cmap='gray')

        ax4.set_title('white_mask', fontsize=14)
        ax4.imshow(white_mask, cmap='gray')

        ax5.set_title('yellow_mask', fontsize=14)
        ax5.imshow(yellow_mask, cmap='gray')

        ax6.set_title('Combined color thresholds', fontsize=14)
        ax6.imshow(combined_binary, cmap='gray')
        plt.savefig('thresholded_exp.jpg')

    return combined_binary


# result = threshold_image(image, plot=True)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
#
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)
#
# ax2.imshow(result, cmap='gray')
# ax2.set_title('Pipeline Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
