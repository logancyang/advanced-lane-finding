## Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_images/comparison1.jpg "Chessboard corner markings"
[image2]: ./output_images/calibration_images/comparison3.jpg "Chessboard corner markings"
[image3]: ./output_images/calibration_images/comparison9.jpg "Chessboard corner markings"
[image4]: ./output_images/undist_images/chess_undist.jpg "Undistorted"

[image5]: ./output_images/undist_images/test_undist_0.jpg "Undistorted 1"
[image6]: ./output_images/undist_images/test_undist_1.jpg "Undistorted 2"
[image7]: ./output_images/undist_images/test_undist_2.jpg "Undistorted 3"

[image8]: ./output_images/thresholded_sx.jpg "thresholded images with s and sobel x binaries"
[image9]: ./output_images/warped_binary_sx.jpg "thresholded images with s and sobel x binaries (warped)"

[image10]: ./output_images/thresholded.jpg "thresholded images with s and l binaries"
[image11]: ./output_images/warped_binary.jpg "thresholded images with s and l binaries (warped)"

[image12]: ./output_images/warp/warp.jpg "Warp Example"

[image13]: ./output_images/fit_lane_lines.png "Fit Visual"

[image14]: ./output_images/highlight_lane.png "Output"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `transform/camera_calibration.py`. The main function is `get_calibration_params` which is used to obtain the calibration parameters, and `cal_undistort` to undistort the images using the parameters obtained from the former step.

To calibrate the camera, there are 20 images taken by the same camera available for computing the calibration parameters needed. The parameters include "object points" and "image points". The "object points" are the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Here are some examples of the chessboard with corners marked:

![alt text][image1]
![alt text][image2]
![alt text][image3]

Note that the calibration parameters, aka `objpoints` and `imgpoints` are obtained from the set of calibration images and will be constant for the same camera. So there is no need to obtain these parameters each time we call the `undistort` function. The right approach is to store the parameters and call `undistort` with them on the target images.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result for the chessboard: 

![alt text][image4]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Applying the calibration parameters and `undistort` function to the test images gave these undistorted results:

![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of different color transformations and thresholding to generate a binary image. The code is in `transform/threshold.py` under the function `threshold_image`. Initially I used the S channel and the Sobel x gradient to do the thresholding and obtained the results below:

![alt text][image8]
![alt text][image9]

The results are not bad but the Sobel x gradient binary is somewhat noisy. Then I tried different combinations of the color channels and finally decided to go with the combination of S channel from HLS space and the L channel from the LUV space. The thresholds are chosen by manual experimentations and can be found in `threshold.py` as default parameters. The final output looks like these:

![alt text][image10]
![alt text][image11]

They look less noisy compared with the previous output and are used as a benchmark in my project. However, there definitly are more optimized ways to approach this step and it could be one potential way to improve the performance of the whole lane finding program.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `bird_view`, which appears in lines 11 through 30 in the file `transform/perspective_transform.py`. The `bird_view` function takes as inputs an image `img`, as well as the configurations `plot` and `output_file`. I chose the hardcode the source and destination points in the following manner:

```python
# 4 source points
src = np.float32([[490, 480], [810, 480], [1250, 720], [40, 720]])
# 4 destination points
dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image12]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code can be found in `transform/process.py` and `transform/line.py`.

I followed the suggestions in the lectures to fit a polynomial to each lane line. After some research I also added the `Line` class to memorize the previous frame and avoid the blind search for every frame. First, identify peaks in a histogram of each frame image. Then identify all non-zero pixels around the peaks and fit a polynomial to each line.

![alt text][image13]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the position of the vehicle with respect to the center of the lane, the center of the image is used as the position of the vehicle. The difference between the midpoint of the lanes and the center of the image is defined as the relative position. The distance from center is converted from pixels to meters by multiplying the number of pixels by `3.7/700`. The code can be found in `transform/process.py` at line 134.

To calculate the curvature, I referenced this useful [link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The final radius of curvature was taken by average the left and right curve radiuses. The code The code can be found in `transform/line.py` from line 94-102.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step uses an inverse perspective transform. The code for this step can be found in `transform/process.py` line 238-264. 

![alt text][image14]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/ag90yBINNS0). In most cases, the program works perfectly as expected. There are 1 or 2 split seconds where the highlighting behaved a little strange and twisted a bit.

---

### Discussion

From this project, I learned the effective approach to find the lane lines in real world images. The steps include

1. Calibrate the camera for **distortion correction**
2. Warp the original image using **perspective transform** to obtain a bird-view image
3. Find the proper combination of **color transformations** and/or **Sobel gradients** to transform the image to binary
4. **Threshold** the binary image to get clear lane lines in the result
5. **Fit a polynomial** on the lane lines and find the **curvature** of the lines
6. Find the vehicle position by calculating the difference between the image center and the lane center
7. Optimization: find the lane lines in each video frame given the knowledge from the previous frame

Among all steps, the key is step 3 and 4. The performance robustness is determined by how effective the lane lines are separated out in the image. Even humans have a hard time seeing lane lines in certain conditions such as dark night, uneven pavement and texture on the road, bad weather, etc. The camera performs better than human in some conditions but worse in others. The lane line detection algorithm should be robust and tested against human performance as a benchmark.
