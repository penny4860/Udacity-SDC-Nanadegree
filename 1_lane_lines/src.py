# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from utils import grayscale, canny, gaussian_blur, region_of_interest, hough_lines, weighted_img


TEST_DIR = "test_images"
files = os.listdir(TEST_DIR)

def fine_lane_pipeline(image):
    imshape = image.shape
    xlength = imshape[1]
    ylength = imshape[0]
    gray = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size=5)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 25
    min_line_length = 10
    max_line_gap = 5
    
    # Run Hough on edge detected image
    line_image = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    vertices = np.array([[(0, ylength),
                          (xlength/2-ylength/10, ylength*0.625),
                          (xlength/2+ylength/10, ylength*0.625),
                          (xlength, ylength)]], dtype=np.int32)
    
    combo = region_of_interest(line_image, vertices)
    combo = weighted_img(combo, image)
    return combo

for filename in files:
    image = mpimg.imread(os.path.join(TEST_DIR, filename))
    im_out = fine_lane_pipeline(image)
    mpimg.imsave(os.path.join("test_images_output", filename), im_out)


