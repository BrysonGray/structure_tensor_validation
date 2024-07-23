#!/usr/bin/env python

"""Utils.py : Helper functions."""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Literal
import scipy


def sph_to_cart(x: np.ndarray, order: Literal['ij', 'xy'] = 'xy') -> np.ndarray:
    if x.ndim == 1:
        x = x[None]
    if order=='xy':
        xout = np.array([np.sin(x[:,0])*np.sin(x[:,1]),
                                np.sin(x[:,0])*np.cos(x[:,1]),
                                np.cos(x[:,0])
                                ]).T
    elif order=='ij':
        xout = np.array([np.cos(x[:,0]),
                        np.sin(x[:,0])*np.cos(x[:,1]),
                        np.sin(x[:,0])*np.sin(x[:,1])
                        ]).T
    
    return xout.squeeze()


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