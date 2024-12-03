import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


def main(args):
    img = cv2.imread(args.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    points = acquire_points(img, n=4)

    print('chosen coord: ', points)  # each row is a point

    '''
    Affine rectification
    '''
    print('\n-------- Task 1: Affine rectification --------')
    pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)  # convert chosen pts to homogeneous coordinate
    print('Task 1.1: Identify image of the line at inf on projective plane')
    hor_0 = #TODO # the first horizontal line
    hor_1 = #TODO # the 2nd horizontal line
    pt_ideal_0 = #TODO # first ideal point
    pt_ideal_0 /= pt_ideal_0[-1]  # normalize
    print('@Task 1.1: first ideal point: ', pt_ideal_0)
    
    # # Debug : Plot first horizontal lines and ideal point
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # x_vals = np.array(ax.get_xlim())
    # # Plot horizontal lines
    # y_vals_h0 = -(hor_0[0] * x_vals + hor_0[2]) / hor_0[1]
    # y_vals_h1 = -(hor_1[0] * x_vals + hor_1[2]) / hor_1[1]
    # ax.plot(x_vals, y_vals_h0, '--r', label='Horizontal line 1')
    # ax.plot(x_vals, y_vals_h1, '--b', label='Horizontal line 2')
    # # Plot first ideal point
    # ax.plot(pt_ideal_0[0], pt_ideal_0[1], 'ro', label='First ideal point')
    # ax.legend()
    # plt.title("First Pair of Lines and Ideal Point")
    

    # ver_0 = #TODO # the 1st vertical line
    # ver_1 = #TODO # the 2nd vertical line
    # pt_ideal_1 = #TODO # 2nd ideal point
    # pt_ideal_1 /= pt_ideal_1[-1]
    # print('@Task 1.1: second ideal point: ', pt_ideal_1)

    # Debug : Plot second vertical lines and ideal point
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # x_vals = np.array(ax.get_xlim())
    # # Plot vertical lines
    # y_vals_v0 = -(ver_0[0] * x_vals + ver_0[2]) / ver_0[1]
    # y_vals_v1 = -(ver_1[0] * x_vals + ver_1[2]) / ver_1[1]
    # ax.plot(x_vals, y_vals_v0, '--r', label='Vertical line 1')
    # ax.plot(x_vals, y_vals_v1, '--b', label='Vertical line 2')
    # # Plot second ideal point
    # ax.plot(pt_ideal_1[0], pt_ideal_1[1], 'ro', label='Second ideal point')
    # ax.legend()
    # plt.title("Second Pair of Lines and Ideal Point")
    # plt.show()

    # l_inf = #TODO # image of line at inf
    # l_inf /= l_inf[-1]
    # print('@Task1.1: line at infinity: ', l_inf)



    # print('Task 1.2: Construct the projectivity that affinely rectify image')
    # H = #TODO 
    # print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

    # # H_E is a Euclidean transformation to center the image for visualization
    # H_E = euclidean_trans(np.deg2rad(0), 50, 250)
    # view_H = H_E @ H
    # affine_img = #TODO 
    # affine_pts = #TODO 
    # for i in range(affine_pts.shape[0]):
    #     affine_pts[i] /= affine_pts[i, -1]

    # plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
    # plt.imshow(affine_img)
    # plt.show()
    # print('-------- End of Task 1 --------\n')

    # '''
    # Task 2: Metric rectification
    # '''
    # print('\n-------- Task 2: Metric rectification --------')
    # print('Task 2.1: transform 4 chosen points from projective image to affine image')
    # aff_hor_0 = #TODO
    # aff_hor_1 = #TODO

    # aff_ver_0 = #TODO
    # aff_ver_1 = #TODO

    # aff_hor_0 /= aff_hor_0[-1]
    # aff_hor_1 /= aff_hor_1[-1]
    # aff_ver_0 /= aff_ver_0[-1]
    # aff_ver_1 /= aff_ver_1[-1]
    # print('@Task 2.1: first chosen point coordinate')
    # print('\t\t on projective image: ', pts_homo[0])
    # print('\t\t on affine image: ', affine_pts[0])

    # print('Task 2.2: construct constraint matrix C to find vector s')
    # C0 = #TODO 

    # C1 = #TODO 

    # C = np.vstack([C0, C1])

    # print('@Task 2.2: constraint matrix C:\n', C)

    # print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
    
    # s = #TODO 

    # print('@Task 2.3: s = ', s)
    # print('@Task 2.3: C @ s = \n', C @ s.reshape(-1, 1))
    # mat_S = np.array([
    #     [s[0], s[1]],
    #     [s[1], s[2]],
    # ])
    # print('@Task 2.3: matrix S:\n', mat_S)

    # print('Task 2.4: Find the projectivity that do metric rectificaiton')
    # eigval, eigvec = #TODO 
    # sigma = #TODO 
    # K = #TODO 
    # H = #TODO 

    # # Check results by computing dual conic
    # aff_dual_conic = np.array([
    #     [s[0], s[1], 0],
    #     [s[1], s[2], 0],
    #     [0, 0, 0]
    # ])
    # print('@Task 2.3: image of dual conic on metric rectified image: ', H @ aff_dual_conic @ H.T)

    # H_E = euclidean_trans(np.deg2rad(0), 50, 250)
    # view_H = H_E @ H
    # eucl_img = cv2.warpPerspective(#TODO)

    # eucl_pts = (view_H @ affine_pts.T).T
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

def acquire_points(image, n=4):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = #TODO
    return np.array(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()
    main(args)
