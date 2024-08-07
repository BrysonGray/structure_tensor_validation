#!/usr/bin/env python

'''
Tools for visualizing image orientations

Author: Bryson Gray
2024

'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils import vec_to_theta
from typing import Literal


def angle_to_rgb(angle: np.ndarray, brightness: float = 1.0, cmap: Literal['rb', 'rgb'] = 'rb') -> list:

    if cmap == 'rb':
        r = np.sin(angle)
        g = np.zeros_like(angle)
        b = np.cos(angle)
        rgb = np.abs(np.stack([r,g,b], axis=-1))
        rgb /= np.max(rgb, axis=-1)[...,None]
    elif cmap=='rgb':
        r = np.abs(brightness * np.sin(angle))
        g = np.abs(brightness * np.sin(angle + 2*np.pi / 3.))
        b = np.abs(brightness * np.sin(angle + 4*np.pi/ 3.))
        rgb = np.abs(np.stack([r,g,b], axis=-1))

    return rgb


def plot_angles(image,
                angles=None,
                means=None,
                mean_colors=None,
                hist_color=None,
                border_color=None,
                axes_coords=[0.1, 0.1, 0.8, 0.8],
                fig=None,
                show=True,
                title=None,
                xlabel=None,
                ylabel=None):
    
    """Plot angles as a polar histogram. This function currently only supports polar (1D) angles.

    Parameters
    ----------
        angles : array_like
            Array of angles. This array will be flattened to create a histogram.
        image : array_like, optional
            2D grayscale image or rgb image with channels in the last dimension.
        means : sequence of floats, optional
            The mean or means of angles outputted from k-means clustering.

    """

    # plot
    if fig is None:
        fig = plt.figure()

    ax_image = fig.add_axes(axes_coords)
    if len(image.shape) == 2:
        ax_image.imshow(image, cmap='gray', alpha=1)
    else:
        ax_image.imshow(image, alpha=1, ) # image can have rgb channels

    if border_color is not None:
        rect = patches.Rectangle((-0.5,-0.5), image.shape[1], image.shape[0], color=border_color, fill=False, linewidth=10)
        ax_image.add_patch(rect)
    ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image

    if title is not None:
        ax_image.set_title(title)
    if xlabel is not None:
        ax_image.set_xlabel(xlabel, fontsize=24)
    if ylabel is not None:
        ax_image.set_ylabel(ylabel, fontsize=24)

    if angles is not None:

        nbins = 500
        angles_flat = angles.flatten()
        angles_ = np.where(angles_flat<0, angles_flat+np.pi, angles_flat-np.pi)
        angles_sym = np.concatenate((angles_flat, angles_), axis=0)

        t = np.arange(nbins+1)*(2*np.pi/nbins) - np.pi
        x, _ = np.histogram(angles_sym, t)

        xf = np.fft.rfft(x)
        n = np.arange(len(xf))
        xf = xf * np.exp(-0.1*n)
        xfinv = np.fft.irfft(xf,nbins+1)
        xfinv = xfinv / np.sum(xfinv)

        polar_coords = np.array(axes_coords) + np.array([0.05, 0.05, -0.1, -0.1])
        ax_polar = fig.add_axes(polar_coords, projection = 'polar')
        ax_polar.patch.set_alpha(0)
        if hist_color is None:
            hist_color = 'royalblue'
        ax_polar.plot(t, xfinv, color=hist_color, linewidth=8)
        ax_polar.set_theta_offset(-np.pi/2)
        ax_polar.set_yticklabels([])
        ax_polar.grid(False)
        ax_polar.axis('off')

    if means is not None:
        if means.shape==():
            means = [means]
        ax_means = fig.add_axes(axes_coords)

        for i,m in enumerate(means):
            
            if mean_colors is not None:
                color = mean_colors[i]
            else:
                color='k'

            if isinstance(m, (int, float)):
                m = [np.sin(m), np.cos(m)]

            ax_means.quiver(0, 0, m[0], -m[1], scale_units='width', scale=3, width=0.02, color=color)
            ax_means.quiver(0, 0, -m[0], m[1], scale_units='width', scale=3, width=0.02, color=color)
        ax_means.axis('off')


    if show:
        plt.show()

    return fig


def plot_angles_3d(image, vectors=None, means=None, mip=False, border_color=None, hist_color=None):
    """Plot 3D vector orientations as a histogram in orthogonal views with the optional original image and vector means.

    Parameters
    ----------
    image : array_like
        3D grayscale image volume array.
    vectors : array_like
        The sequence of angles as three-dimensional vectors with components along the last axis.
    means : array_like

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
    
    if means is not None:
        mu_2d = []
        # project into orthogonal planes. Resulting shape is (3, n_clusters, 2)
        for m in means: # for each mean append the (3,2) array representing one 2d vector per orthogonal plane
            mu_2d.append([[m[0], m[1]],
                          [m[0], m[2]],
                          [m[1], m[2]]])
        mu_2d = np.array(mu_2d) # shape (n_clusters, 3, 2)
        mu_2d = np.transpose(mu_2d, (1,0,2)) # shape (3, n_clusters, 2)
        # mean_colors = np.abs(means) / np.max(np.abs(means))
        mean_colors = np.abs(np.stack((means[:,2], means[:,0], means[:,1]), axis=-1)) / np.max(np.abs(means))
    else:
        mu_2d = [None,None,None]
        mean_colors = None
        
    figsize = [p*c for p,c in zip(plt.rcParams["figure.figsize"], [3.0, 1.0])]
    # figsize = [p*c for p,c in zip(plt.rcParams["figure.figsize"], [1.0, 3.0])]

    fig = plt.figure(figsize=figsize)

    axes_coords_list = [[0.005, 0.05, 0.325, 0.85], # x0, y0, ∆x, ∆y
                        [0.33, 0.05, 0.325, 0.85],
                        [0.66, 0.05, 0.325, 0.85]]
    # axes_coords_list = [[0.05, 0.66, 0.85, 0.325], # x0, y0, ∆x, ∆y
    #                 [0.05, 0.33, 0.85, 0.325],
    #                 [0.05, 0.005, 0.85, 0.325]]
    # titles = ["x-y", "x-z", "y-z"]
    xlabels = ["X", "X", "Y"]
    ylabels = ["Y", "Z", "Z"]

    for i in range(3):
        if vectors is not None:
            plot_angles(image=I_ortho[i], angles=angles_2d[i], means=mu_2d[i], mean_colors=mean_colors,\
                        axes_coords=axes_coords_list[i], fig=fig, show=False, title=None, xlabel=xlabels[i],\
                        ylabel=ylabels[i], border_color=border_color, hist_color=hist_color)
        else:
            plot_angles(image=I_ortho[i], means=mu_2d[i], mean_colors=mean_colors, axes_coords=axes_coords_list[i],\
                        fig=fig, show=False, title=None, xlabel=xlabels[i], ylabel=ylabels[i], border_color=border_color)

    plt.show()

    return fig