# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline I implemented is a total of 6 steps.

First, convert the color image to a gray scale image.
* Second, remove noise with gaussian smoothing in a gray scale image.
* Three times, find the edge with the canny edge detection algorithm.
* Fourth, find the line pixel using the hough transform in the edge map.
* Fifth, obtain the region maksing map of the position where the lane is located at the camera position of the car.
* Finally, we combine the line pixel map with the region masking map to compute the final result.

In order to draw a single line on the left and right lanes, I modified draw_lines() in such a way that lines with similar orientations connect the midpoints of each line.


### 2. Identify potential shortcomings with your current pipeline

* Because it is based on line detection using Hough Transform, curved lanes are difficult to detect.
* Since I manually tuned the parameters, the recognition rate of lane detection may be lower for various images.

### 3. Suggest possible improvements to your pipeline

* Hough transform-based algorithms could be replaced by algorithms that find MSER and connected components. MSER(Maximally stable extremal regions) will have the advantage of a curved lane because it detects not only the straight line but also the whole area with similar color in the image.





