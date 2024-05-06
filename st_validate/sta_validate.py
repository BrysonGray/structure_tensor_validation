#!/usr/bin/env python
# ruff: noqa: E501
# ruff: noqa: E741
'''
Structure tensor analysis validation functions.

Author: Bryson Gray
2023

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.linalg import expm
from tqdm.contrib import itertools as tqdm_itertools

from periodic_kmeans.periodic_kmeans import PeriodicKMeans
import histology
import apsym_kmeans


def anisotropy_correction(image, dI, direction='up', blur=False):
    isotropic = np.all(np.array(dI) == dI[0])
    if not isotropic:
    # downsample all dimensions to largest dimension or upsample to the smallest dimension.
        x_in = [np.arange(n)*d for n,d in zip(image.shape, dI)]

        if direction == 'down':
            dx = np.max(dI)
        elif direction == 'up':
            dx = np.min(dI)

        x_out = [np.arange(0,n*d, step=dx) for n, d in zip(image.shape, dI)]
        Xout = np.stack(np.meshgrid(*x_out, indexing='ij'), axis=-1)
        image = scipy.interpolate.interpn(points=x_in, values=image, xi=Xout, method='linear', bounds_error=False, fill_value=None)
        
    if blur is not False:
        image = gaussian_filter(image, sigma=blur)
    
    return image
    

def gather(I, patch_size=None):
    """ Gather I into patches.

        Parameters
        ----------
        I : three or four-dimensional array with last dimension of size n_features

        patch_size : int, or {list, tuple} of length I.ndim
            The side length of each patch

        Returns
        -------
        I_patches : four or five-dimensional array with samples aggregated in the second
            to last dimension and the last dimension has size n_features.
    """
    if patch_size is None:
        patch_size = [I.shape[1] // 10] * (I.ndim-1) # default to ~100 tiles in an isostropic image
    elif isinstance(patch_size, int):
        patch_size = [patch_size] * (I.ndim-1)
    n_features = I.shape[-1]
    if I.ndim == 3:
        i, j = [x//patch_size[i] for i,x in enumerate(I.shape[:2])]
        I_patches = I[:i*patch_size[0],:j*patch_size[1]].copy() # crop so 'I' divides evenly into patch_size (must create a new array to change stride lengths)
        # reshape into patches by manipulating strides. (np.reshape preserves contiguity of elements, which we don't want in this case)
        nbits = I_patches.strides[-1]
        I_patches = np.lib.stride_tricks.as_strided(I_patches, shape=(i,j,patch_size[0],patch_size[1],n_features),
                                                    strides=(patch_size[0]*I_patches.shape[1]*n_features*nbits,
                                                             patch_size[1]*n_features*nbits,
                                                             I_patches.shape[1]*n_features*nbits,
                                                             n_features*nbits,
                                                             nbits))
        I_patches = I_patches.reshape(i,j,np.prod(patch_size),n_features)
    elif I.ndim == 4:
        i, j, k = [x//patch_size[i] for i,x in enumerate(I.shape[:3])]
        I_patches = np.array(I[:i*patch_size[0], :j*patch_size[1], :k*patch_size[2]])
        nbits = I_patches.strides[-1]
        I_patches = np.lib.stride_tricks.as_strided(I_patches, shape=(i, j, k, patch_size[0], patch_size[1], patch_size[2], n_features),
                                                strides=(patch_size[0]*I_patches.shape[1]*I_patches.shape[2]*n_features*nbits,
                                                        patch_size[1]*I_patches.shape[2]*n_features*nbits,
                                                        patch_size[2]*n_features*nbits,
                                                        I_patches.shape[1]*I_patches.shape[2]*n_features*nbits,
                                                        I_patches.shape[2]*n_features*nbits,
                                                        n_features*nbits,
                                                        nbits))
        I_patches = I_patches.reshape(i,j,k,np.prod(patch_size),n_features)
    return I_patches


def make_phantom(x, angles, period=10, width=1.0, noise=1e-6, crop=None,\
                 blur_correction=False, display=False, interp=True, inverse=False):
    """
    Parameters
    ----------
    x : list of arrays
        x[i] stores the location of voxels on the i-th axis of the image

    angles : list or ndarray 
        Angles of the lines in radians. For 2D phantoms, this must have length n where n
        is the number of angles and the values must be in the range [-pi/2, pi/2].
        In 2D the angle is relative to the first image axis which points toward the bottom of the image. 
        For 3D phantoms, this must have shape (n,2) where the first value (the polar angle) is relative
        to the first image axis and in the range [0, pi] and the second value (the azimuthal angle) is
        relative to the second image axis and in the range [-pi/2, pi/2].

    period : int
        Space between lines.

    width : int
        Width of the lines.

    noise : float
        Noise level.

    blur_correction : bool
        If True, upsample by interpolating and apply a Gaussian filter to the image to create isotropic blur.
    
    display : bool
    
    interp : bool
        If True, interpolate the image to the largest dimension.

    Returns
    -------
    phantom : ndarray of shape nI

    labels : ndarray, optional

    """
    d = np.array([xi[1] - xi[0] for xi in x])
    b = np.array([len(xi)//2 for xi in x])
    X = np.stack(np.meshgrid(*x, indexing='ij'), axis=-1)
    blur_factor = np.sqrt(d[0]**2 - d[1]**2)

    I = np.random.randn(*X.shape[:-1])*noise
    labels = None

    if len(x) == 3:
        sigma = (np.diag(d)*width)**2 # sigma is the covariance matrix
        blur = (0., blur_factor, blur_factor)
        for angle in angles:
            direction = np.array([np.cos(angle[0]),
                                np.sin(angle[0])*np.cos(angle[1]),
                                np.sin(angle[0])*np.sin(angle[1])
                                ]).T

            # rotation matrix using Rodrigues' formula
            if np.all(direction == [1.0,0.0,0.0]):
                sigma_ = sigma
                x_ = (X - b)[...,None]
            else:
                axis = np.cross(direction,np.array([1.0,0.0,0.0]))
                axis = axis / np.sum(axis**2)**0.5
                alpha = np.arccos(np.dot(direction,np.array([1.0,0.0,0.0])))    
                K = np.array([[0.0,-axis[2],axis[1]],
                        [axis[2],0.0,-axis[0]],
                        [-axis[1],axis[0],0.0]])
                R = expm(alpha*K)
                # covariance
                sigma_ = R@sigma@R.T
                x_ = (R@(X-b)[...,None])
            sigma__ = sigma_[1:,1:]
            Z = 1.0/np.sqrt(2.0*np.pi**2)/np.linalg.det(sigma__)**0.5
            # note that the 0the component will not go into the gaussian
            x__ = x_[...,1:,:]

            # draw parallel lines using mod
            if period is not None:
                x__ = ((x__+period/2)%period) - period/2
                
            tmp = np.linalg.inv(sigma__)@x__
            tmp = x__.swapaxes(-1,-2)@tmp
            I_ = Z*np.exp(-0.5*tmp[...,0,0])
            I += I_
        if inverse:
            alpha = 10
            I = np.exp(-alpha*I)

        if blur_correction:
            I = anisotropy_correction(I, d, labels, blur=blur)
        elif interp:
            I = anisotropy_correction(I, d, labels)

        if crop is not None:
            if crop > 0:
                I[crop:-crop, crop:-crop, crop:-crop]

        if display:
            fig, ax = plt.subplots(3, figsize=(6,4))
            ax[0].imshow(I[I.shape[0]//2])
            ax[0].set_title('Image xy')
            ax[1].imshow(I[:,I.shape[1]//2])
            ax[1].set_title('Image zx')
            ax[2].imshow(I[:,:,I.shape[2]//2])
            ax[2].set_title('Image zy')
            plt.show()

    elif len(x) == 2:
        blur = (0., blur_factor)
        for angle in angles:
            sigma = (np.sin(angle)*d[0]*width)**2 + (np.cos(angle)*d[1]*width)**2 # variance (not standard deviation)
            x__ = (X - b)@np.array([-np.sin(angle), np.cos(angle)])
            if period is not None:
                x__ = ((x__+period/2)%period) - period/2
            Z = 1.0 / (2.0*np.pi*sigma)
            I_ = Z*np.exp(-0.5 * x__**2 / sigma)

            I += I_

        if inverse:
            alpha = 10
            I = np.exp(-alpha*I)

        if blur_correction:
            I = anisotropy_correction(I, d, blur=True)
        elif interp:
            I = anisotropy_correction(I, d)

        if crop is not None:
            if crop > 0:
                I[crop:-crop, crop:-crop]
        if display:
            plt.imshow(I)
            plt.title('Image')
    
    return I


def sta_test(I, derivative_sigma, tensor_sigma, true_thetas=None, patch_size=None, crop=None, crop_end=0, display=False, return_all=False):
    """Test structure tensor analysis on a phantom.

    Parameters
    ----------
    I : two or three-dimensional image array

    derivative_sigma : list, float
        Sigma for the derivative filter.

    tensor_sigma : float
        Sigma for the structure tensor filter.

    err_type : {'pixelwise', 'piecewise'}
        Pixelwise returns average angular difference per pixel. 
        Piecewise computes k-means for multiple line angles for comparison with ground truth. This operation
        is optionally divided into image patches with size specified by argument patch_size.
    
    true_thetas : list or ndarray 
        True angles of the lines in radians. For 2D phantoms, this must have length n where n
        is the number of angles and the values must be in the range [-pi/2, pi/2].
        For 3D phantoms, this must have shape (n,2) where the first value (the polar angle) is
        in the range [0, pi], and the second value (the azimuthal angle) is in the range [-pi/2, pi/2].

    patch_size : int, or {list, tuple} of length I.ndim, optional
        If not None, the image is divided into patches with patch size length given by patch_size. The error is computed per patch.
    
    crop : int, optional
        Number of pixels to crop from the edges before computing angle averages.

    display : bool, default=False

    return_all : bool, default=False
        If True, return error, mean angle values, angles,
        and diff (array of differences between mean and ground truth per patch).
    
    Returns
    -------
    error : float
        Average angular difference between ground truth and estimated angles for
        pixelwise error, or jensen-shannon divergence for piecewise error.

    """

    nI = I.shape
    dim = len(nI)

    if dim == 2:
        # compute structure tensor and angles
        S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma, masked=False)
        angles = histology.angles(S)

        # first crop boundaries to remove artifacts related to averaging tensors near the edges.
        if crop is not None:
            if crop > 0:
                angles = angles[crop:-(crop+crop_end), crop:-crop]
        if patch_size is not None:
            # gather angles into non-overlapping patches
            angles_ = angles[...,None]
            angles_ = gather(angles[...,None], patch_size=patch_size)
            angles_ = angles_.squeeze(axis=-1)
        else:
            angles_ = angles.reshape(-1,dim)[None,None]

        # Estimate kmeans centers and errors for each tile.
        diff = np.zeros(angles_.shape[:2])
        for i in range(angles_.shape[0]):
            for j in range(angles_.shape[1]):
                angles_tile = angles_[i,j][~np.isnan(angles_[i,j])]
                angles_tile = np.where(angles_tile < 0, angles_tile + np.pi, angles_tile) # flip angles to be in the range [0,pi] for periodic kmeans
                if len(true_thetas) == 1:
                    mu_ = histology.periodic_mean(angles_tile.flatten()[...,None], period=np.pi)
                elif len(true_thetas) == 2:
                    periodic_kmeans = PeriodicKMeans(angles_tile[...,None], period=np.pi, no_of_clusters=2)
                    _, _, centers = periodic_kmeans.clustering()
                    mu_ = np.array(centers).squeeze()
                else:
                    raise Exception(f"argument \"true_thetas\" must be float or sequence of length 2 for 2D images.")
                
                mu_flipped = np.where(mu_ < 0, mu_ + np.pi, mu_ - np.pi)
                mu = np.stack((mu_,mu_flipped), axis=-1)
                diff_ = np.abs(mu[...,None] - true_thetas) # this has shape (2,2,2) for 2 mu values each with 2 possible orientations, and each compared to both ground truth angles
                if len(true_thetas) == 1:
                    diff[i,j] = np.min(diff_) * 180/np.pi
                else:
                    argmin = np.array(np.unravel_index(np.argmin(diff_), (2,2,2))) # the closest mu value and orientation is the first error
                    remaining_idx = 1 - argmin # the second error is the best error from the other mu value compared to the other ground truth angle
                    diff[i,j] = np.mean([diff_[tuple(argmin)], np.min(diff_,1)[remaining_idx[0],remaining_idx[2]]]) * 180/np.pi

        error = np.mean(diff)

        if display:
            fig, ax = plt.subplots(1,2, figsize=(6,4))
            ax[0].imshow(angles)
            ax[0].set_title('Angles')
            ax[1].imshow(diff)
            ax[1].set_title('Difference')
            plt.show()

    elif dim == 3:
        # compute structure tensor and angles
        S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma, masked=False)
        angles = histology.angles(S, cartesian=True) # shape is (i,j,k,3) where the last dimension is in x,y,z order
        # crop boundaries to remove artifacts related to averaging tensors near the edges.
        if crop is not None:
            if crop > 0:
                angles = angles[crop:-(crop+crop_end), crop:-crop, crop:-crop]
        if patch_size is not None:
            # gather angles into non-overlapping patches
            angles_ = gather(angles, patch_size=patch_size)
        else:
            angles_ = angles.reshape(-1,dim)[None,None,None]

        # convert true_thetas to cartesian coordinates for easier error calculation
        true_thetas = np.array([np.sin(true_thetas[:,0])*np.sin(true_thetas[:,1]),
                                np.sin(true_thetas[:,0])*np.cos(true_thetas[:,1]),
                                np.cos(true_thetas[:,0])
                                ]).T

        if len(true_thetas) == 1:
            skm = apsym_kmeans.APSymKMeans(n_clusters=1)
        elif len(true_thetas) == 2:
            skm = apsym_kmeans.APSymKMeans(n_clusters=2)
        else:
            raise Exception(f"argument \"true_thetas\" must be have 1 or 2 dimensions but got {true_thetas.ndim}.")
        
        # Estimate kmeans centers for each tile.
        diff = np.empty(angles_.shape[:3])
        for i in range(angles_.shape[0]):
            for j in range(angles_.shape[1]):
                for k in range(angles_.shape[2]):
                    if len(true_thetas) == 1:
                        skm.fit(angles_[i,j,k])
                        mu_ = skm.cluster_centers_
                        diff[i,j,k] = np.arccos(np.abs(mu_.dot(true_thetas.T))) * 180/np.pi
                    else:
                        skm.fit(angles_[i,j,k])
                        mu_ = skm.cluster_centers_ # shape (n_clusters, n_features)
                        diff_ = np.empty((len(mu_),len(true_thetas))) # shape (2,2) for two permutations of the difference between two means and two true_thetas
                        for m in range(len(mu_)):
                            for n in range(len(true_thetas)):
                                diff_[m,n] = np.arccos(np.abs(mu_[m].dot(true_thetas[n])))
                        argmax = np.unravel_index(np.argmin(diff_), (2,2))
                        corrolary = tuple([1 - x for x in argmax]) # the corresponding cos_dif of the other mu to the other grid_theta
                        diff[i,j,k] = np.mean([diff_[argmax], diff_[corrolary]]) * 180/np.pi

        error = np.mean(diff)

        if display:

            fig, ax = plt.subplots(1,3, figsize=(6,3))
            ax[0].imshow(np.abs(angles[nI[1]//2]))
            ax[0].set_title('angles xy')
            ax[1].imshow(np.abs(angles[:,nI[1]//2]))
            ax[1].set_title('angles zx')
            ax[2].imshow(np.abs(angles[:,:,nI[1]//2]))
            ax[2].set_title('angles zy')
            plt.show()
            if angles_.shape[0] > 1:
                fig, ax = plt.subplots(1,3, figsize=(6,3))
                ax[0].imshow(diff[diff.shape[0]//2])
                ax[0].set_title('diff xy')
                ax[1].imshow(diff[:,diff.shape[1]//2])
                ax[1].set_title('diff zx')
                ax[2].imshow(diff[:,:,diff.shape[1]//2])
                ax[2].set_title('diff zy')
                plt.show()
            else:
                print(f'error = {diff[0,0,0]} degrees')

    if return_all:
        return error, mu_, angles, diff
    else:
        return error


def run_tests(derivative_sigmas, tensor_sigmas, nIs, angles, periods=[10], blur_correction=False):
    """ Run a series of ST tests

    Parameters
    ----------
    derivative_sigmas : list
        List of derivative standard deviations. 
    tensor_sigmas : list
        List of tensor (window) standard deviations.
    nIs : list of tuples
        List of image sizes. Each tuple contains the number of pixels along each dimension.
        For 3D images the last two dimensions must be the same size
    angles : list, optional
        A list of angles or pairs of angles of the phantom lines in radians.
    periods : list, optional
        The period (distance between lines) for each phantom, by default [10]
    blur_correction : bool, optional
        If True, apply Gaussian blur to high resolution dimension to create equal blur in each dimension, by default False

    Returns
    -------
    pandas Dataframe
        Dataframe storing input parameters and resultant errors.
    """

    error_df = pd.DataFrame({'derivative_sigma':[], 'tensor_sigma':[], 'anisotropy_ratio':[], 'period':[], 'angles':[], 'error':[]})
    # ensure all arguments are lists
    if not isinstance(derivative_sigmas, (list, tuple, np.ndarray)):
        derivative_sigmas = [derivative_sigmas]
    if not isinstance(tensor_sigmas, (list, tuple, np.ndarray)):
        tensor_sigmas = [tensor_sigmas]
    if not isinstance(nIs[0], (list, tuple, np.ndarray)):
        nIs = [nIs]
    if not isinstance(periods, (list, tuple, np.ndarray)):
        periods = [periods]
    if not isinstance(angles[0], (list, tuple, np.ndarray)):
        angles = [angles]
    
    for i1,i2,i3 in tqdm_itertools.product(range(len(nIs)), range(len(periods)), range(len(angles))):
        nI = nIs[i1]
        anisotropy_ratio = float(nI[1]/nI[0])
        if len(nI) == 2:
            dI = (anisotropy_ratio, 1.0)
        elif len(nI) == 3:
            dI = (anisotropy_ratio, 1.0, 1.0)
        else:
            raise Exception(f"nI must have length of either two or three but got {len(nI)}")
        
        x = [np.arange(ni)*di for ni,di in zip(nI,dI)]
        period = periods[i2]
        angle = angles[i3]
        I = make_phantom(x, angle, period, blur_correction=blur_correction)

        for s1 in range(len(derivative_sigmas)):
            for s2 in range(len(tensor_sigmas)):
                derivative_sigma = derivative_sigmas[s1]
                tensor_sigma = tensor_sigmas[s2]
                crop_all = round(max(derivative_sigma,tensor_sigma)*8/3) # two-thirds the radius of the largest kernel
                crop_end = round(anisotropy_ratio) - 1
                error = sta_test(I, derivative_sigma, tensor_sigma, true_thetas=angle, crop=crop_all, crop_end=crop_end)

                new_row = {'derivative_sigma': derivative_sigma, 'tensor_sigma': tensor_sigma,
                            'anisotropy_ratio': anisotropy_ratio, 'period': period, 'width': 1,
                            'angles': [angle], 'error': error
                        }
                error_df = pd.concat((error_df, pd.DataFrame(new_row)), ignore_index=True)

    return error_df