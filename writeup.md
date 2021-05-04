# ** Writeup of "Advanced Lane Finding Project" **

[//]: # (Image References)

---

**Advanced Lane Finding Project**

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/original_lane_image.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

  I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

---

### Writeup / README

### Camera Calibration

Camera calibration process is following,

* convert to grayscale
* Prepare object points for (x, y, z) like (0, 0, 0), (1, 0, 0)...
* Generate co-ordinate for the given grid size, reshapes to 2x2 matrix for x and y
* Detect chessboard corners of the distorted image in grayscale image
* Append the corners to the image points
* Draw the detected corners on an image
* Calibrate the camera to obtain calibration matrix and distortion coefficients

![alt text][image1]


### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `find-lane-pixels.py`).  

The code for my perspective transform includes a function called `transform_perspective()`, which appears in the file `perspective_transform.py`.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

I did this in function `find_lane_line_from_poolynomial()` in my code in `find_lane_pixels.py`

I implemented this step in lines # through # in my code in `P2.ipynb` in the function `original_lane_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

---


### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

.
