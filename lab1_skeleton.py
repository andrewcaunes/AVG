"""
    Skeleton code for lab 1
    Usage : python path/to/lab1_skeleton.py --filename path/to/fig1_6c__.jpg

"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
# The following import is to enable logging with line number (instead of printing)
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

# args come from the script at the end of the file
def main(args):
    """
    Goal : Perform affine rectification then metric rectification on the given image
    """
    # Read image
    img = cv2.imread(args.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    logging.info('\n-------- Task 1: Affine rectification --------')
    # Get 2 pairs of parallel lines in the affine image
    # (You can use the following incomplete function to get points)
    points = acquire_points(img, n=4)
    logging.info(f'chosen points: {points}')  # each row is a point
    
    # [Tip : uncomment the code step by step once the previous step is working]
    # convert chosen pts to homogeneous coordinate
    # pts_homo = #TODO
    
    # # Identify image of the line at infinity on the projective plane
    # line_0_0 = #TODO # the first horizontal line
    # line_0_1 = #TODO # the 2nd horizontal line
    # pt_vanishing_0 = #TODO # first vanishing point
    # pt_vanishing_0 /= pt_vanishing_0[-1]  # normalize
    # logging.info('First vanishing point: %s', pt_vanishing_0)
    
    # # Debug : Plot first horizontal lines and vanishing point
    # plot_lines_and_vanishing_point(img, line_0_0, line_0_1, pt_vanishing_0)

    # line_1_0 = #TODO # the 1st vertical line
    # line_1_1 = #TODO # the 2nd vertical line
    # pt_vanishing_1 = #TODO # 2nd vanishing point

    # pt_vanishing_1 /= pt_vanishing_1[-1]
    # logging.info('Second vanishing point: %s', pt_vanishing_1)

    # # Debug : Plot second vertical lines and vanishing point
    # plot_lines_and_vanishing_point(img, line_1_0, line_1_1, pt_vanishing_1)

    # # l_inf = #TODO # image of line at inf
    # # l_inf /= l_inf[-1]
    # logging.info('Line at infinity: %s', l_inf)

    # # Construct the projectivity that affinely rectify image
    # H = #TODO 
    
    # # Check your results : what should be the image of line at inf on affinely rectified image?
    # logging.info('Image of line at inf on affinely rectified image: %s', 
    #              (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

    # # H_E is a Euclidean transformation to center the image for visualization
    # H_E = euclidean_trans(np.deg2rad(0), 50, 250)
    # view_H = H_E @ H
    # # affine_img = #TODO 
    # # affine_pts = #TODO 
    # for i in range(affine_pts_for_visualization.shape[0]):
    #     affine_pts_for_visualization[i] /= affine_pts_for_visualization[i, -1]

    # plt.plot(*zip(*affine_pts_for_visualization[:, :-1]), marker='o', color='r', ls='')
    # plt.imshow(affine_img_for_visualization)
    # plt.show()

    # logging.info('\n-------- Task 2: Metric rectification --------')
    # # Get 2 pairs of orthogonal lines in the affine image
    # # (you can re-use the points from the parallel lines))
    # aff_line_0_0 = #TODO
    # aff_line_0_1 = #TODO

    # aff_line_1_0 = #TODO
    # aff_line_1_1 = #TODO

    # aff_line_0_0 /= aff_line_0_0[-1]
    # aff_line_0_1 /= aff_line_0_1[-1]
    # aff_line_1_0 /= aff_line_1_0[-1]
    # aff_line_1_1 /= aff_line_1_1[-1]

    # # Construct constraint matrix C to find vector s
    # C0 = #TODO 
    # C1 = #TODO 
    # C = np.vstack([C0, C1])
    # logging.info('Constraint matrix C:\n%s', C)


    # # Find s by looking for the kernel of C (hint: SVD)
    
    # s = #TODO 

    # mat_S = np.array([
    #     [s[0], s[1]],
    #     [s[1], s[2]],
    # ])
    # logging.info('Matrix S:\n%s', mat_S)

    # # Find the projectivity that do metric rectification
    # eigval, eigvec = #TODO 
    # if #TODO:
    #     raise ValueError("Error: Found non-positive eigenvalues. The matrix S should be positive definite.")
    # sigma = #TODO 
    # K = #TODO 
    # H = #TODO 
    
    # # Check results by computing dual conic, what should be the image 
    # # of dual conic on the metric rectified image?
    # aff_dual_conic = np.array([
    #     [s[0], s[1], 0],
    #     [s[1], s[2], 0],
    #     [0, 0, 0]
    # ])
    # logging.info('Image of dual conic on metric rectified image: %s', H @ aff_dual_conic @ H.T)

    # # Automatically find H_E for centering image
    # temp_pts = (H @ affine_pts.T).T
    # for i in range(temp_pts.shape[0]):
    #     temp_pts[i] /= temp_pts[i, -1]
    # H_E = auto_euclidean_trans(temp_pts, img.shape)
    # view_H = H_E @ H

    # # Warp image and points
    # eucl_img = cv2.warpPerspective(#TODO, (img.shape[1]*2, img.shape[0]*2)) # *2 is used to ensure enough space
    # eucl_pts = #TODO
    # for i in range(eucl_pts.shape[0]):
    #     eucl_pts[i] /= eucl_pts[i, -1]
    # plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
    # plt.imshow(eucl_img)
    # plt.show()

def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])


def points_that_should_work_if_pointing_does_not_work():
    return np.array([
        [ 745.65367965,  646.25324675],
        [1057.34199134,  464.43506494],
        [1382.01731602,  641.92424242],
        [1061.67099567,  867.03246753]
    ])

def get_bounds(points):
    """Calculate the bounds of transformed points"""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, max_x, min_y, max_y

def auto_euclidean_trans(points, img_shape):
    """Automatically compute translation to center the transformed points"""
    min_x, max_x, min_y, max_y = get_bounds(points)
    
    # Calculate center of transformed points
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    
    # Calculate translation to center of image
    tx = img_shape[1]/2 - center_x
    ty = img_shape[0]/2 - center_y
    
    return euclidean_trans(np.deg2rad(0), tx, ty)

def plot_lines_and_vanishing_point(img, line1, line2, vanishing_point, 
                                   title="lines and vanishing point", 
                                   line1_label="Line 1", 
                                   line2_label="Line 2"):
    """
    Plot two lines and their vanishing point on the image
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    x_vals = np.array(ax.get_xlim())
    # Plot lines
    y_vals_1 = -(line1[0] * x_vals + line1[2]) / line1[1]
    y_vals_2 = -(line2[0] * x_vals + line2[2]) / line2[1]
    ax.plot(x_vals, y_vals_1, '--r', label=line1_label)
    ax.plot(x_vals, y_vals_2, '--b', label=line2_label)
    # Plot vanishing point
    ax.plot(vanishing_point[0], vanishing_point[1], 'ro', label='Vanishing point')
    ax.legend()
    plt.title(title)
    plt.show()

def acquire_points(image, n=4):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = # TODO
    plt.close(fig)  # Close the figure after getting points
    return np.array(points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()
    main(args)
