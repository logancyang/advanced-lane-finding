import numpy as np
import cv2
from transform.camera_calibration import get_calibration_params, undistort
from transform.perspective_transform import bird_view
from transform.threshold import threshold_image
import glob
from transform.line import Line
import matplotlib.image as mpimg

calib_images = glob.glob('camera_cal/calibration*.jpg')
# calib_images = glob.glob('../camera_cal/calibration*.jpg')

# Undistort
calib_params = get_calibration_params(calib_images)
Left = Line("left")
Right = Line("right")


def process_single(image):
    img_size = (image.shape[1], image.shape[0])

    # Calibrate camera and undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(calib_params["objpoints"], calib_params["imgpoints"], img_size,
                                                       None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Warp
    warped, M = bird_view(undist)

    # Threshold
    combined_binary = threshold_image(warped, plot=True)


def process(image):
    img_size = (image.shape[1], image.shape[0])

    # Calibrate camera and undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(calib_params["objpoints"], calib_params["imgpoints"], img_size,
                                                       None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Warp
    warped, M = bird_view(undist)

    # Threshold
    combined_binary = threshold_image(warped)

    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(combined_binary))

    if Left.found:  # Search for left lane pixels around previous polynomial
        leftx, lefty, Left.found = Left.found_search(x, y)

    if Right.found:  # Search for right lane pixels around previous polynomial
        rightx, righty, Right.found = Right.found_search(x, y)

    if not Right.found:  # Perform blind search for right lane lines
        rightx, righty, Right.found = Right.blind_search(x, y, combined_binary)

    if not Left.found:  # Perform blind search for left lane lines
        leftx, lefty, Left.found = Left.blind_search(x, y, combined_binary)

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    # Calculate left polynomial fit based on detected pixels
    left_fit = np.polyfit(lefty, leftx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int, left_top = Left.get_intercepts(left_fit)

    # Average intercepts across n frames
    Left.x_int.append(leftx_int)
    Left.top.append(left_top)
    leftx_int = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    Left.lastx_int = leftx_int
    Left.last_top = left_top

    # Add averaged intercepts to current x and y vals
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)

    # Sort detected pixels based on the yvals
    leftx, lefty = Left.sort_vals(leftx, lefty)

    Left.X = leftx
    Left.Y = lefty

    # Recalculate polynomial with intercepts and average across n frames
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0),
                np.mean(Left.fit1),
                np.mean(Left.fit2)]

    # Fit polynomial to detected pixels
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    Left.fitx = left_fitx

    # Calculate right polynomial fit based on detected pixels
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int, right_top = Right.get_intercepts(right_fit)

    # Average intercepts across 5 frames
    Right.x_int.append(rightx_int)
    rightx_int = np.mean(Right.x_int)
    Right.top.append(right_top)
    right_top = np.mean(Right.top)
    Right.lastx_int = rightx_int
    Right.last_top = right_top
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)

    # Sort right lane pixels
    rightx, righty = Right.sort_vals(rightx, righty)
    Right.X = rightx
    Right.Y = righty

    # Recalculate polynomial with intercepts and average across n frames
    right_fit = np.polyfit(righty, rightx, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]

    # Fit polynomial to detected pixels
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]
    Right.fitx = right_fitx

    # Compute radius of curvature for each lane in meters
    left_curverad = Left.radius_of_curvature(leftx, lefty)
    right_curverad = Right.radius_of_curvature(rightx, righty)

    # Only print the radius of curvature every 3 frames
    if Left.count % 3 == 0:
        Left.radius = left_curverad
        Right.radius = right_curverad

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_int + leftx_int) / 2
    distance_from_center = abs((640 - position) * 3.7 / 700)

    # TODO: Should avoid hardcoding the src and dst
    src = np.float32([[490, 480], [810, 480], [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])
    pts = np.hstack((pts_left, pts_right))

    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (combined_binary.shape[1], combined_binary.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    # Print distance from center on video
    if position > 640:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left.radius + Right.radius) / 2)), (120, 140),
                fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    Left.count += 1
    return result

# test_image = mpimg.imread('../test_images/test5.jpg')
# process_single(test_image)
