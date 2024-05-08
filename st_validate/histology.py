#!/usr/bin/env python

'''
Histology fiber orienataion analysis tools

Author: Bryson Gray
2022

'''

from scipy.ndimage import gaussian_filter
import os
import cv2
from skimage.transform import resize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import apsym_kmeans
from periodic_kmeans.periodic_kmeans import PeriodicKMeans


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
        Array of values between -pi/2 and pi/2.
    """
    w,v = np.linalg.eigh(S)
    v = v[...,-1] # the principal eigenvector is always the last one since they are ordered by least to greatest eigenvalue.
    # Remember that structure tensors are in x-y-z order (i.e. col-row-slice instead of slice-row-col).
    if w.shape[-1] == 2:
        if cartesian:
            return v
        theta = np.arctan(v[...,0] / (v[...,1] + np.finfo(float).eps)) # x/y gives the counterclockwise angle from the vertical direction (y axis).
        return theta
    else:
        if cartesian:
            return v
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


def angle_to_rgb(angle, brightness=1):
    r = np.abs(brightness * np.sin(angle))
    g = np.abs(brightness * np.sin(angle + 2*np.pi / 3.))
    b = np.abs(brightness * np.sin(angle + 4*np.pi/ 3.))

    return [r,g,b]


def vec_to_theta(vec):
    return np.arctan(vec[...,0] / (vec[...,1] + np.finfo(float).eps))


def periodic_mean(points, period=180):
    period_2 = period/2
    if max(points) - min(points) > period_2:
        _points = np.array([0 if x > period_2 else 1 for x in points]).reshape(-1,1)
        n_left =_points.sum()
        n_right = len(points) - n_left
        if n_left >0:
            mean_left = (points * _points).sum()/n_left
        else:
            mean_left =0
        if n_right >0:
            mean_right = (points * (1-_points)).sum() / n_right
        else:
            mean_right = 0
        _mean = (mean_left*n_left+mean_right*n_right+n_left*period)/(n_left+n_right)
        return np.array([_mean % period])
    else:
        return points.mean(axis=0)


def spherical_kmeans(vectors, n_clusters, cartesian=True):
    """ Compute antipodally symetric spherical k-means.

    Parameters
    ----------
    angles : array_like
        An array of shape (N,3), where N is the number of sample directions and the last dimension is vector components in cartesian coordinates.
    n_clusters : int
        Number of means (k) for k-means.

    Returns
    -------
    means : ndarray of shape (n_clusters, 3)
    
    """
    skm = apsym_kmeans.APSymKMeans(n_clusters=n_clusters)
    skm.fit(vectors)
    means = skm.cluster_centers_

    if not cartesian: # then return means in spherical coordinates (theta (polar angle), phi (azimuthal angle))
        x = means[...,0]
        y = means[...,1]
        z = means[...,2]
        theta = np.arctan(np.sqrt(x**2 + y**2) / (z + np.finfo(float).eps)) # range is (-pi/2,pi/2)
        theta = np.where(theta < 0, theta + np.pi, theta) # range is (0,pi)
        phi = np.arctan(x / (y + np.finfo(float).eps)) # range (-pi/2, pi/2)
        return np.stack((theta,phi), axis=-1)

    return means

def circular_kmeans(angles, n_clusters):

    angles = angles.flatten()
    angles = angles[~np.isnan(angles)]
    angles = np.where(angles < 0, angles + np.pi, angles) # flip angles to be in the range [0,pi] for periodic kmeans
    if n_clusters==1:
        means = periodic_mean(angles[...,None], period=np.pi)
    elif n_clusters > 1:
        periodic_kmeans = PeriodicKMeans(angles[...,None], period=np.pi, no_of_clusters=n_clusters)
        _, _, centers = periodic_kmeans.clustering()
        means = np.array(centers).squeeze()
    else:
        raise ValueError(f'n_clusters must be greater than or equal to 1, but got {n_clusters}')
    
    return means


def plot_angles(image, angles=None, means=None, ntheta=500, axis=False, axes_coords=[0.1, 0.1, 0.8, 0.8], fig=None, show=True, title=None, xlabel=None, ylabel=None, colors=None):
    """Plot angles as a polar histogram. This function currently only supports polar (1D) angles.

    Parameters
    ----------
        angles : array_like
            Array of angles. This array will be flattened to create a histogram.
        image : array_like, optional
            2D grayscale image or rgb image with channels in the last dimension.
        means : sequence of floats, optional
            The mean or means of angles outputted from k-means clustering.
        ntheta : int
            Number of bins in to use for the histogram.
        axes : bool
            Show polar axis. Default is False
    """

    # plot
    if fig is None:
        fig = plt.figure()

    ax_image = fig.add_axes(axes_coords)
    if len(image.shape) == 2:
        ax_image.imshow(image, cmap='gray', alpha=1)
    else:
        ax_image.imshow(image, alpha=1) # image can have rgb channels
    # ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if title is not None:
        ax_image.set_title(title)
    if xlabel is not None:
        ax_image.set_xlabel(xlabel, fontsize=24)
    if ylabel is not None:
        ax_image.set_ylabel(ylabel, fontsize=24)

    
    if angles is not None:
        angles_flat = angles.flatten()
        angles_ = np.where(angles_flat<0, angles_flat+np.pi, angles_flat-np.pi)
        angles_sym = np.concatenate((angles_flat, angles_), axis=0)

        t = np.arange(ntheta+1)*(2*np.pi/ntheta) - np.pi
        x, _ = np.histogram(angles_sym, t)

        xf = np.fft.rfft(x)
        n = np.arange(len(xf))
        xf = xf * np.exp(-0.1*n)
        xfinv = np.fft.irfft(xf,ntheta+1)
        xfinv = xfinv / np.sum(xfinv)

        polar_coords = np.array(axes_coords) + np.array([0.05, 0.05, -0.1, -0.1])
        ax_polar = fig.add_axes(polar_coords, projection = 'polar')
        ax_polar.patch.set_alpha(0)
        ax_polar.plot(t, xfinv, 'royalblue', linewidth=8)
        ax_polar.set_theta_offset(-np.pi/2)
        ax_polar.set_yticklabels([])
        ax_polar.grid(False)
        if not axis:
            ax_polar.axis('off')


    if means is not None:
        if means.shape==():
            means = [means]
        ax_means = fig.add_axes(axes_coords)

        for i,m in enumerate(means):
            if colors is None:
                color='lime'
            elif isinstance(colors[i], (int, float)): # if color is a scalar, treat it as an angle and convert to rgb
                color = angle_to_rgb(colors[i])
            elif isinstance(colors[i], (np.ndarray, list, tuple)): # if color is a sequence, treat it as an rgb value.
                color = np.abs(colors[i])
                if len(color) != 3:
                    raise ValueError("The colors must be either scalars or sequences of length 3.")
                
            if isinstance(m, (int, float)):
                m = [np.sin(m), np.cos(m)]
            ax_means.quiver(0, 0, m[0], -m[1], scale_units='width', scale=3, width=0.02, color=color)
            ax_means.quiver(0, 0, -m[0], m[1], scale_units='width', scale=3, width=0.02, color=color)
        ax_means.axis('off')


    if show:
        plt.show()

    return


def plot_angles_3d(image, vectors=None, means=None, mip=False):
    """Plot 3D vector orientations as a histogram in orthogonal views with the optional original image and vector means.

    Parameters
    ----------
    vectors : array_like
        The sequence of angles as three-dimensional vectors with components along the last axis.
    image : array_like
        3D grayscale image volume array.
    plot_means : bool, defaults to False.

    """
    if mip:
        I_ortho = [image.max(axis=0),
                   image.max(axis=1),
                   image.max(axis=2)]
    else:
        I_ortho = [image[image.shape[0]//2],
                   image[:, image.shape[1]//2],
                   image[:, :, image.shape[2]//2]]
        
    if vectors is not None:
        vectors = vectors.reshape(-1,3)
        # vectors are in x-y-z (col-row-slice) order so they must be reordered
        x = vectors[...,0]
        y = vectors[...,1]
        z = vectors[...,2]
        vec_2d = [
                np.stack((x, y), axis=-1), # xy
                np.stack((x, z), axis=-1), # xz
                np.stack((y, z), axis=-1), # yz
                ]

        angles_2d = []
        for i in range(3):
            angles_2d.append(vec_to_theta(vec_2d[i]))
    
    mu_2d = []
    colors = []
    if means is not None:
        # project into orthogonal planes. Resulting shape is (3, n_clusters, 2)
        for m in means: # for each mean append the (3,2) array representing one 2d vector per orthogonal plane
            mu_2d.append([[m[0], m[1]],
                          [m[0], m[2]],
                          [m[1], m[2]]])
        mu_2d = np.array(mu_2d) # shape (n_clusters, 3, 2)
        mu_2d = np.transpose(mu_2d, (1,0,2)) # shape (3, n_clusters, 2)
        colors = np.abs(means)

    fig = plt.figure()

    axes_coords_list = [[0.1, 0.1, 0.8, 0.8], # x0, y0, ∆x, ∆y
                        [1.0, 0.1, 0.8, 0.8],
                        [2.0, 0.1, 0.8, 0.8]]
    # titles = ["x-y", "x-z", "y-z"]
    xlabels = ["X", "X", "Y"]
    ylabels = ["Y", "Z", "Z"]
    for i in range(3):
        if vectors is not None:
            plot_angles(image=I_ortho[i], angles=angles_2d[i], means=mu_2d[i], axes_coords=axes_coords_list[i], fig=fig, show=False, title=None, xlabel=xlabels[i], ylabel=ylabels[i], colors=colors)
        else:
            plot_angles(image=I_ortho[i], means=mu_2d[i], axes_coords=axes_coords_list[i], fig=fig, show=False, title=None, xlabel=xlabels[i], ylabel=ylabels[i], colors=colors)

    plt.show()

    return