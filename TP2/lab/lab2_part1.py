"""
AVG Lab 2 - Part 1 : Epipolar Geometry, Using Fundamental Matrix
Example use:
> cd lab2_folder
> python lab2_part1.py --input_path chapel_F.txt

Submission: Please submit this code completed and commented. You need to comment
everything you add and explain why you add it.
"""

import os
import shutil
import argparse
import logging
from matplotlib import pyplot as plt
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import cv2
import numpy as np

def get_points(image, n=1):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = plt.ginput(n)
    plt.close(fig)  # Close the figure after getting points
    return np.array(points)

def main(args):
    """
    In this part, we will compute the epipolar lines and the epipole given the F matrix 
    and plot them on the second image.

    Input: 
    - chapel_0.png: the first image
    - chapel_1.png: the second image
    - F_gt: the ground truth fundamental matrix
    - User input for the points on the first image 
     (you should prompt the user to input the points !!!)

    Output: a single plot showing the chapel_1.png image with the epipolar lines 
    and the epipole (a dot).
    """
    from constants import F_gt as F
    logging.info("args = %s", args)

    #TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--img_0', help='', default='chapel_0.png')
    parser.add_argument('--img_1', help='', default='chapel_1.png')
    args = parser.parse_args()

    main(args)