#!/usr/bin/env python

'''
Structure tensor analysis tools

Author: Bryson Gray
2022

'''

import os

import cv2
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

def load_img(impath, img_down=0, reverse_intensity=False):
    imname = os.path.split(impath)[1]
    print(f'loading image {imname}...')
    I = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    if img_down:
        print('downsampling image...')
        I = resize(I, (I.shape[0]//img_down, I.shape[1]//img_down), anti_aliasing=True)
    # fit I to range (0,1)
    if np.max(I[0]) > 1:
        I = I * 1/255
    if reverse_intensity == True:
        I = 1 - I

    return I


def structure_tensor(I, derivative_sigma=1.0, tensor_sigma=1.0, normalize=True, masked=False, id_minus_S=False):
    '''
    Construct structure tensors from a grayscale image. Accepts 2D or 3D arrays

    Parameters
    ----------
    I : array
        2D or 3D scalar image
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in which case it is equal
        for all axes.
    Returns
    -------
    S : array
        Array of structure tensors with image dimensions along the first axes and tensors in the last two dimensions. Tensors are arranged in x-y-z (i.e. col-row-slice) order.

    '''
    if I.dtype == np.uint8:
        I = I.astype(float) / 255

    if I.ndim == 2:
        # note the kernel size is 2*radius + 1 and the radius of the gaussian filter is round(truncate * sigma) where truncate defaults to 4.0.
        # gaussian_filter has default border mode 'reflect'.
        Ix =  gaussian_filter(I, sigma=[derivative_sigma, derivative_sigma], order=(0,1))
        Iy =  gaussian_filter(I, sigma=[derivative_sigma, derivative_sigma], order=(1,0))
        norm = np.sqrt(Ix**2 + Iy**2) + np.finfo(float).eps
        if normalize:
            Ix = Ix / norm
            Iy = Iy / norm

        # construct the structure tensor, s
        Ixx = gaussian_filter(Ix*Ix, sigma=[tensor_sigma, tensor_sigma])
        Ixy = gaussian_filter(Ix*Iy, sigma=[tensor_sigma, tensor_sigma])
        Iyy = gaussian_filter(Iy*Iy, sigma=[tensor_sigma, tensor_sigma])

        # S = np.stack((Iyy, Ixy, Ixy, Ixx), axis=-1)
        S = np.stack((1-Ixx,-Ixy,-Ixy,1-Iyy), axis=-1) # identity minus the structure tensor
        # # null out the structure tensor where the norm is too small
        if masked:
            S[norm < 1e-9] = None
        S = S.reshape((S.shape[:-1]+(2,2)))

    elif I.ndim == 3:

        Ix =  gaussian_filter(I, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(0,0,1))
        Iy =  gaussian_filter(I, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(0,1,0))
        Iz =  gaussian_filter(I, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(1,0,0))

        norm = np.sqrt(Ix**2 + Iy**2 + Iz**2) + np.finfo(float).eps
        if normalize:
            Ix = Ix / norm
            Iy = Iy / norm
            Iz = Iz / norm

        Ixx = gaussian_filter(Ix*Ix, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        Iyy = gaussian_filter(Iy*Iy, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        Izz = gaussian_filter(Iz*Iz, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        Ixy = gaussian_filter(Ix*Iy, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        Ixz = gaussian_filter(Ix*Iz, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        Iyz = gaussian_filter(Iy*Iz, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])

        S = np.stack((Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz), axis=-1)
        S = S.reshape((S.shape[:-1]+(3,3)))
        # 
        if not id_minus_S:
            S = -S # identity minus the structure tensor
        else:
            S = np.eye(3) - S
    else:
        raise Exception(f'Input must be a 2 or 3 dimensional array but found: {I.ndim}')

    return S

def anisotropy(w):
    """
    Calculate anisotropy from eigenvalues. Accepts 2 or 3 eigenvalues

    Parameters
    ----------
    w : array
        Array with eigenvalues along the last dimension.
    
    Returns
    --------
    A : array
        Array of anisotropy values.
    """

    if w.shape[-1] == 3:
        w = w.transpose(3,0,1,2)
        trace = np.sum(w, axis=0)
        A = np.sqrt((3/2) * (np.sum((w - (1/3)*trace)**2,axis=0) / np.sum(w**2, axis=0)))
        A = np.nan_to_num(A)
        A = A/np.max(A)
    elif w.shape[-1] == 2:
        A = abs(w[...,0] - w[...,1]) / abs(w[...,0] + w[...,1])
    else:
        raise Exception(f'Accepts 2 or 3 eigenvalues but found {w.shape[-1]}')
    
    return A


def angles(S, cartesian=False):
    """
    Compute angles from structure tensors.

    Parameters
    ----------
    S : ndarray
        Structure tensor valued image array.
    
    Returns
    -------
    angles : ndarray
        For S of shape (...,2,2), returns an array of values between -pi/2 and pi/2.
        For S of shape (...,3,3), returns an array of shape (...,2) where the first
        element of the last dimension is the angle from the +z axis with range [0,pi],
        and the second element is the counterclockwise angle from the +y axis with
        range [-pi/2, pi/2]. If cartesian == True, the output is an array of the principal
        eigenvectors of S in x-y-z (col-row-slice) order.

    """
    w,v = np.linalg.eigh(S)
    v = v[...,-1] # the principal eigenvector is always the last one since they are ordered by least to greatest eigenvalue.
    # Remember that structure tensors are in x-y-z order (i.e. col-row-slice instead of slice-row-col).
    if cartesian:
        return v
    
    if w.shape[-1] == 2:
        theta = np.arctan(v[...,0] / (v[...,1] + np.finfo(float).eps)) # x/y gives the counterclockwise angle from the vertical direction (y axis). Range [-pi/2, pi/2]
        return theta
    
    else:
        x = v[...,0]
        y = v[...,1]
        z = v[...,2]
        theta = np.arctan(np.sqrt(x**2 + y**2) / (z + np.finfo(float).eps)) # range is (-pi/2,pi/2)
        theta = np.where(theta < 0, theta + np.pi, theta) # range is (0,pi)
        phi = np.arctan(x / (y + np.finfo(float).eps)) # range (-pi/2, pi/2)
        return np.stack((theta,phi), axis=-1)


def hsv(S, I):
    """
    Compute angles, anisotropy index, and hsv image from 2x2 structure tensors.

    Parameters
    ----------
    S : array
        Array of structure tensors with shape MxNx2x2
    I : array
        Image with shape MxN

    Returns
    -------
    theta : array
        Array of angles (counterclockwise from left/right) with shape MxN. Angles were mapped from [-pi/2,pi/2] -> [0,1] for easier visualization.
    AI : array
        Array of anisotropy index with shape MxN
    hsv : array
        Image with theta -> hue, AI -> saturation, and I -> value (brightness).
    """
    # check if I is 2D
    if I.ndim != 2:
        raise Exception(f'Only accepts two dimensional images but found {I.ndim} dimensions')

    # print('calculating orientations and anisotropy...')
    w,v = np.linalg.eigh(S)
    v = v[...,-1] # the principal eigenvector is always the last one since they are ordered by least to greatest eigenvalue with all being > 0
    theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi # TODO: verify this is correct since changing S component order. 
    # row/col gives the counterclockwise angle from left/right direction. Rescaled [-pi/2,pi/2] -> [0,1]
    AI = anisotropy(w) # anisotropy index (AI)

    # make hsv image where hue= primary orientation, saturation= anisotropy, value= original image
    # print('constructing hsv image...')
    
    if S.shape[:-2] != I.shape:
        down = [x//y for x,y in zip(I.shape, S.shape[:-2])]
        I = resize(I, (I.shape[0]//down[0], I.shape[1]//down[1]), anti_aliasing=True)
    stack = np.stack([theta,AI,I], -1)
    hsv = matplotlib.colors.hsv_to_rgb(stack)

    return theta, AI, hsv

def project_to_plane(vectors, normal, L=None):
    """
    Poject a sequence of three-dimensional vectors onto a plane through the origin defined by its normal vector.

    Parameters
    ----------
    vectors : array_like
        The sequence of angles as three-dimensional vectors with components along the last axis.
    normal : array_like
        A sequence of three scalars defining the normal vector to the plane on which to project the angles.
    L : array_like
        Linear transform from 3D basis to the new basis in the 2D plane

    Returns
    -------
    vectors_p : array
        vectors projected onto the plane.
    
    """

    vectors = np.asarray(vectors)
    normal = np.asarray(normal)
    L = np.asarray(L)

    normal = normal / np.sum(normal**2)**0.5 # ensure normal has unit length
    
    u = np.einsum('...i,i->...', vectors, normal) 
    u = u[...,None] * normal[None]
    vectors_p = vectors - u
    if L is not None:
        vectors_p = np.einsum('ij,...j->...i', L, vectors_p)

    return vectors_p


#TODO: Remove
# def periodic_mean(points, period=180):
#     period_2 = period/2
#     if max(points) - min(points) > period_2:
#         _points = np.array([0 if x > period_2 else 1 for x in points]).reshape(-1,1)
#         n_left =_points.sum()
#         n_right = len(points) - n_left
#         if n_left >0:
#             mean_left = (points * _points).sum()/n_left
#         else:
#             mean_left =0
#         if n_right >0:
#             mean_right = (points * (1-_points)).sum() / n_right
#         else:
#             mean_right = 0
#         _mean = (mean_left*n_left+mean_right*n_right+n_left*period)/(n_left+n_right)
#         return np.array([_mean % period])
#     else:
#         return points.mean(axis=0)

#TODO: Remove
# def periodic_mean(points, period=180):

#     half_period = period/2
#     is_left = np.array([0 if x > half_period else 1 for x in points])
    
#     n_left = is_left.sum()
#     n_right = len(points) - n_left

#     if n_left > 0 and n_right > 0:

#         mean_left = (points * is_left).sum() / n_left
#         mean_right = (points * (1-is_left)).sum() / n_right

#         if mean_right - mean_left <= period/2:
#             mean = (n_left*mean_left + n_right*mean_right)/len(points)
#         else:
#             mean = (n_left*(mean_left + period) + n_right*mean_right)/len(points) % period
    
#     else:
#         mean = points.sum()/len(points)
    
#     return mean

#TODO: Remove
# def spherical_kmeans(vectors, n_clusters, cartesian=True):
#     """ Compute antipodally symetric spherical k-means.

#     Parameters
#     ----------
#     angles : array_like
#         An array of shape (N,3), where N is the number of sample directions and the last dimension is vector components in cartesian coordinates.
#     n_clusters : int
#         Number of means (k) for k-means.

#     Returns
#     -------
#     means : ndarray of shape (n_clusters, 3)
    
#     """
#     skm = apsym_kmeans.APSymKMeans(n_clusters=n_clusters)
#     skm.fit(vectors)
#     means = skm.cluster_centers_

#     if not cartesian: # then return means in spherical coordinates (theta (polar angle), phi (azimuthal angle))
#         x = means[...,0]
#         y = means[...,1]
#         z = means[...,2]
#         theta = np.arctan(np.sqrt(x**2 + y**2) / (z + np.finfo(float).eps)) # range is (-pi/2,pi/2)
#         theta = np.where(theta < 0, theta + np.pi, theta) # range is (0,pi)
#         phi = np.arctan(x / (y + np.finfo(float).eps)) # range (-pi/2, pi/2)
#         return np.stack((theta,phi), axis=-1)

#     return means