# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    def slop(line):
        x1,y1,x2,y2 = line[0, :]
        return np.arctan2((y2-y1), (x2-x1))
    
    def offset(line):
        x1,y1,x2,y2 = line[0, :]
        m = (y2-y1) / max((x2-x1), 1e-6)
        b = y1 - m * x1
        return b
    
    def get_sorted_lines(lines):
        slops = []
        for line in lines:
            slops.append(slop(line))
        indexes = np.argsort(slops)
        sorted_lines = [lines[i] for i in indexes]
        return sorted_lines

    def average_point(line):
        x1, y1, x2, y2 = line[0, :]
        return (x1+x2)/2, (y1+y2)/2

    def merge_line(line1, line2):
        
        x1, y1 = average_point(line1)
        x2, y2 = average_point(line2)

        line = [x1, y1, x2, y2]
        return np.array(line).astype(int).reshape(-1, 4)

    def is_merge(line1, line2, thd_theta=np.pi/180*2, thd_b=5):
        s1 = slop(line1)
        s2 = slop(line2)
        
        b1 = offset(line1)
        b2 = offset(line2)
        
        if abs(s1-s2) <= thd_theta and abs(b1-b2) <= thd_b:
            return True
        else:
            return False
    
    sorted_lines = get_sorted_lines(lines)
    
    for i in range(len(sorted_lines)-1):
        line1 = sorted_lines[i]
        line2 = sorted_lines[i+1]
           
        if is_merge(line1, line2):
            sorted_lines.append(merge_line(line1, line2))
            
    for line in sorted_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

#def weighted_img(img, initial_img, ��=0.8, ��=1., ��=0.):
def weighted_img(img, initial_img, alpha=0.8, beta=1., lambda_=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * �� + img * �� + ��
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambda_)

