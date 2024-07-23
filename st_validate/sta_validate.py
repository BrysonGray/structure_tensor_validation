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
import pandas as pd
from scipy.linalg import expm
from tqdm.contrib import itertools as tqdm_itertools
import periodic_kmeans
import histology
import utils


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

            direction = utils.sph_to_cart(angle, order='ij')

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
            I = utils.anisotropy_correction(I, d, blur=blur)
        elif interp:
            I = utils.anisotropy_correction(I, d)

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
            I = utils.anisotropy_correction(I, d, blur=True)
        elif interp:
            I = utils.anisotropy_correction(I, d)

        if crop is not None:
            if crop > 0:
                I[crop:-crop, crop:-crop]
        if display:
            plt.imshow(I)
            plt.title('Image')
    
    return I


def sta_test(I, derivative_sigma, tensor_sigma, true_thetas=None, crop=None, crop_end=None):
    """Test structure tensor analysis on a phantom.

    Parameters
    ----------
    I : two or three-dimensional image array

    derivative_sigma : list, float
        Sigma for the derivative filter.

    tensor_sigma : float
        Sigma for the structure tensor filter.

    true_thetas : list or ndarray 
        True angles of the lines in radians. For 2D phantoms, this must have length n where n
        is the number of angles and the values must be in the range [-pi/2, pi/2].
        For 3D phantoms, this must have shape (n,2) where the first value (the polar angle) is
        in the range [0, pi], and the second value (the azimuthal angle) is in the range [-pi/2, pi/2].
    
    crop : int, optional
        Number of pixels to crop from the edges before computing angle averages.
    
    crop_end : int, optional
        Number of pixels to crop along the first dimension to remove the upsampling artifact due to anisotropy correction.

    Returns
    -------
    error : float
        Average angular difference between ground truth and estimated angles in degrees
        
    """

    nI = I.shape
    dim = len(nI)
    if dim == 0 or dim > 3:
        raise TypeError(f"Input image should have two or three dimensions but got {dim}")
    if len(true_thetas) == 0 or len(true_thetas) > 3:
        raise Exception(f"Argument \"true_thetas\" must be have length 1 or 2 but got {len(true_thetas)}.")
        
    # Compute angles from image
    S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma)
    if dim == 2:
        angles = histology.angles(S) # range [-pi/2, pi/2]
    elif dim == 3:
        angles = histology.angles(S, cartesian=True)        

    if crop == 0.0:
        crop = None
    if crop_end == 0.0:
        crop_end = None
    # first crop boundaries to remove artifacts related to averaging tensors near the edges.
    if crop is not None:
        if crop > 0:
            angles = angles[crop:-crop, crop:-crop]
    if crop_end is not None:
        angles = angles[:-crop_end]
    

    # Compute mean or means
    if dim == 2:
        angles = angles.flatten()
        angles = np.where(angles < 0, angles + np.pi, angles) # range [0,pi]
        if len(true_thetas) == 1:
            x = np.arange(180) * np.pi/180
            mu = periodic_kmeans.periodic_mean(angles, x, period=np.pi)[None]
        else:
            mu = periodic_kmeans.periodic_kmeans(angles, period=np.pi, k=2)
        
        # Get difference between mean(s) and the true angle(s)
        diff = periodic_kmeans.distance(mu, np.array(true_thetas), period=np.pi) # shape (k,k) for k means
        diff = periodic_kmeans.multiple_exclusive_distances(diff) # shape (k,)

        error = np.mean(diff)

    if dim == 3:
        angles = angles.reshape(-1,dim)
        # convert true_thetas to cartesian coordinates for easier error calculation
        true_thetas = utils.sph_to_cart(true_thetas)

        if len(true_thetas) == 1:
            mu = periodic_kmeans.apsym_kmeans(angles, k=1)
            diff = np.arccos(np.abs(mu.dot(true_thetas.T)))
        else:
            mu = periodic_kmeans.apsym_kmeans(angles, k=2)
            diff = periodic_kmeans.distance_3d(mu, true_thetas)
            diff = periodic_kmeans.multiple_exclusive_distances(diff)
            diff = np.mean(diff)

        error = np.mean(diff)
    
    return error.astype(np.float64) * 180/np.pi
        

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

                new_row = {'derivative_sigma': derivative_sigma, 'tensor_sigma': tensor_sigma, 'anisotropy_ratio': anisotropy_ratio,
                           'period': period, 'angles': [angle], 'error': error
                          }
                error_df = pd.concat((error_df, pd.DataFrame(new_row)), ignore_index=True)

    return error_df