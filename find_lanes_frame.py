import glob

import imageio
# imageio.plugins.ffmpeg.download()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from transform.camera_calibration import get_calibration_params, undistort
from transform.perspective_transform import bird_view
from transform.threshold import threshold_image
from transform.line import Line
from transform.process import process


calib_images = glob.glob('camera_cal/calibration*.jpg')
image = mpimg.imread('test_images/test6.jpg')
video = None

# # Undistort
# calib_params = get_calibration_params(calib_images)
#
# undist = undistort(image, calib_params)
#
# # Warp
# warped, M = bird_view(undist)
#
# # Threshold
# binary_image = threshold_image(warped)

# Plot the result
# result = binary_image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
#
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)
#
# ax2.imshow(result, cmap='gray')
# ax2.set_title('Pipeline Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('result.jpg')
# plt.show()


# Process video
Left = Line("left")
Right = Line("right")
video_output = 'result.mp4'
clip1 = VideoFileClip("project_video.mp4") #.subclip(40, 43)
white_clip = clip1.fl_image(process)
white_clip.write_videofile(video_output, audio=False)


