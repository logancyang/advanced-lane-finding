import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from transform.camera_calibration import undistort

# Read in an image
img = cv2.imread('../test_images/test6.jpg')


def bird_view(img, plot=False, output_file="warp"):
    # 4 source points
    src = np.float32([[490, 480], [810, 480],
                      [1250, 720], [40, 720]])
    # 4 destination points
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Undistorted Image', fontsize=20)
        ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted and Warped Image', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('../output_images/' + output_file + '.jpg')

    return warped, M


for index, fname in enumerate(glob.glob('../test_images/test*.jpg')):
    image = cv2.imread(fname)
    bird_view(image, True, 'warp' + str(index))
