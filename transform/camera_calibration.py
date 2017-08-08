import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in calibration images
images = glob.glob('../camera_cal/calibration*.jpg')
# test_images = glob.glob('../test_images/test*.jpg')


def get_calibration_params(images, save=False):
    """
    :param images: Calibration image files
    :return: objpoints and imgpoints for undistortion. Each objpoint is produced by one calibration image
    """
    # Create objpoints and imgpoints
    # 3d points in real world space
    objpoints = []
    # 2d points in image plane.
    imgpoints = []
    calibration_params = {
        'objpoints': objpoints,
        'imgpoints': imgpoints
    }
    nx = 9
    ny = 6
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for index, fname in enumerate(images):
        img = cv2.imread(fname)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If corners found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display corners
            img = cv2.drawChessboardCorners(img, (ny, nx), corners, ret)
            if save:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
                ax1.imshow(cv2.cvtColor(mpimg.imread(fname), cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Image', fontsize=18)
                ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax2.set_title('With Corners', fontsize=18)
                plt.savefig('comparison' + str(index) + '.jpg')
    return calibration_params


# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, params):
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(params['objpoints'], params['imgpoints'], (img.shape[1], img.shape[0]), None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def undistort_test_images(images, calib_param):
    for index, fname in enumerate(images):
        test_image = cv2.imread(fname)
        undistorted = cal_undistort(test_image, calib_param)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('../output_images/undist_images/chess_undist_' + str(index) + '.jpg')
        # plt.show()


global_param = get_calibration_params(images)


def undistort(image, param=global_param):
    return cal_undistort(image, param)


# undistort_test_images(images, global_param)
