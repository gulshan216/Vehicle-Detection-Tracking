**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car-not_car.png
[image2]: ./examples/HOG_example1.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window_examples.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/Heatmap_example.png
[video1]: ./output_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1.Extracting HOG features from the training images.

The code for this step is contained in the ninth code cell of the IPython notebook(Vehicle Detection and Tracking.ipynb). I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:
![alt text][image1]

I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.
Here is an example using the LUV color space and HOG parameters of orientations=6, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):

![alt text][image2]

#### 2. Choosing the right HOG parameters.

I tried various combinations of parameters as shown below:
Color_space = RGB, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), hog_channel = 0     Accuracy = 0.945
Color_space = LUV, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), hog_channel = 1    Accuracy = 0.95
Color_space = YCrCb, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), hog_channel = 1    Accuracy = 0.955

 Then I looked at the suggestion from the HOG paper which showed that using hog features from all channels gave better accuracy and hence tried the following combination of parameters which gave the best accuracy for my case as well: 
Color_space = YCrCb, orientations=11, pixels_per_cell=(8, 8), cells_per_block=(2, 2), hog_channel = ‘ALL’      Accuracy = 0.9865

#### 3. Training a classifier using selected HOG features and color features.

The code for this step is contained in the tenth and eleventh code cell of the IPython notebook. I trained a linear SVM using spatial binning, color histogram and HOG based features. The classifier gives an accuracy of 98.65% on the test set.

### Sliding Window Search

#### 1.Implementing a sliding window search.

The code for this step is contained in the fourteenth code cell of the IPython notebook. I decided to implement the HOG subsampling based sliding window technique, wherein instead of performing feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. 

#### 2. Examples of test images to demonstrate how your pipeline is working.  Optimizing the performance of the classifier.

Ultimately I searched on the following four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
y_start_stop = [400,460]  scale = 0.8, y_start_stop = [400,480]  scale = 1.2, y_start_stop = [400,460]  scale = 0.8, 
y_start_stop = [400,460]  scale = 0.8
 Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Link to final video output. 
Here's a [link to my video result](./output_video.mp4)


#### 2. Filtering for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. In order to smoothen out the bounding boxes over the video I keep appending the heatmaps to a queue of maximum length 8 and then take a mean of all the samples in the queue to get the final heatmap. Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the pipeline does reasonably well on filtering the false positives. Any more thresholding actually can get rid of correct car detections. I think though that for images , a convolutional neural network might be better in learning the features of car, rather than manually tuning and picking the right parameters for the HOG feature. And hence we might end up with a better classifier with lesser false positive rates with better object detection.

