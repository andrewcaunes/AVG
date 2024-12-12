"""
AVG Lab 2 - Part 2 : Epipolar Geometry, Computing fundamental matrix
Example use:
> cd lab2_folder
> python lab2_part2.py 

Submission: Please submit this code completed and commented. You need to comment
everything you add and explain why you add it.
"""

import argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
from constants import F_gt
# The following imports are for printing with file name and line numbers
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    # center = # TODO: find center of the set of points by computing mean of x & y
    # dist = # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    # s = # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans

def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

def main(args):
    """
    In this part, we will estimate the fundamental matrix F from point correspondences
    between two images using the normalized 8-point algorithm.

    Input:
    - Two images of the same scene taken from different viewpoints

    Output:
    - The estimated fundamental matrix F
    """
    logging.info("args = %s", args)

    # Read image & put them in grayscale
    img1 = cv2.imread(#TODO, 0)  # queryImage
    img2 = cv2.imread(#TODO, 0)  # trainImage
    

    # The following part is for automatically getting pairs of corresponding points from two images
    # Detect keypoints & compute descriptors automatically (Don't hesitate to google/LLM what this does)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(#TODO, None)
    kp2, des2 = orb.detectAndCompute(#TODO, None)
    

    # Match keypoints using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(#TODO, #TODO)


    # Organize key points into matrix, each row is a point
    query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
    train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

    # # plot matches
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img_matches)
    # plt.title('Matched Keypoints')
    # plt.show()

    # # The following follows Algorithm 1 from the lab subject
    # # Normalize keypoints
    # T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
    # T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts

    # # Homogenize and transform keypoints
    # # ... you may need to add more lines here
    # normalized_query_kpts = #TODO
    # normalized_train_kpts = #TODO


    # # Construct homogeneous linear equation to find fundamental matrix
    # # ... you may add lines here
    # A = # TODO: construct A according to Eq.(3) in lab subject


    # # Find vector f by solving A f = 0 using SVD
    # # ... you may add lines here
    # f = # TODO: find f


    # # Arrange f into 3x3 matrix to get fundamental matrix F
    # F = f.reshape(3, 3)
    # print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

    # # Force F to have rank 2
    # # ... you may need to add more lines here
    # F = #TODO
    # assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

    # # De-normalize F
    # # hint: last line of Algorithme 1 in the lab subject
    # F = #TODO


    # # Check if F is correct
    # logging.info("(F - F_gt) = %s", F - F_gt)
    # # Save new F in order to test using lab2_part1.py
    # np.savetxt(args.F_new, F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--img_1', help='', default="img/chapel_0.png")
    parser.add_argument('--img_2', help='', default="img/chapel_1.png")
    parser.add_argument('--F_new', help='', default="estimated_F.txt")
    args = parser.parse_args()

    main(args)


