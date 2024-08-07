#!/usr/bin/env python

"""
k-means on one dimensional periodic data

Author: Bryson Gray
2024
"""

import numpy as np


def multiple_exclusive_distances(diff):
    """ Get the mean distance between estimated means and true means.

    Parameters
    ----------
    diff: numpy.ndarray
        
    """
    if len(diff) > 1:
        ids = []
        diff_ = diff.copy()
        for i in range(len(diff_)):
            x = np.unravel_index(np.nanargmin(diff_), diff.shape) # get the index of the smallest difference
            ids.append(x)
            # remove the mean (row) and true angle (col) it's closest to from the diff matrix
            diff_[x[0],:] = np.nan # remove row
            diff_[:,x[1]] = np.nan # remove col
        ids = np.array(ids)
        error = diff[tuple([x for x in ids.T])]
    else:
        error = np.array(diff[0,0])
    
    return error

def distance(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:

    d = np.abs(a[:,None] - b[None])
    d = np.minimum(d, np.array([period]) - d)
    return d


def periodic_mean(data: np.ndarray, x: np.ndarray, period: float) -> float:
    period = np.pi
    d2 = distance(x, data, period)**2
    id = np.argmin(d2.sum(axis=1))

    return x[id]

#TODO: Remove
def periodic_mean_(points, period=180):

    half_period = period/2
    is_left = np.array([0 if x > half_period else 1 for x in points])
    
    n_left = is_left.sum()
    n_right = len(points) - n_left

    if n_left > 0 and n_right > 0:

        mean_left = (points * is_left).sum() / n_left
        mean_right = (points * (1-is_left)).sum() / n_right

        if mean_right - mean_left <= period/2:
            mean = (n_left*mean_left + n_right*mean_right)/len(points)
        else:
            mean = (n_left*(mean_left + period) + n_right*mean_right)/len(points) % period
    
    else:
        mean = points.sum()/len(points)
    
    return mean


def periodic_kmeans(data: np.ndarray, k: int, period: float, nstarts=1, max_iter=100) -> np.ndarray:
    metrics = []
    mus = []
    for i in range(nstarts):
        # initialize k random starting points
        x = np.arange(180) * period/180
        mu = np.random.choice(data,k,replace=False)
        for j in range(max_iter):
            mu_old = mu.copy()
            d = distance(mu, data, period)
            labels = np.argmin(d, axis=0)
            for j in range(len(mu)):
                mu[j] = periodic_mean(data[labels==j], x, period)
            if np.all(mu == mu_old):
                break
        mus.append(mu)
        metrics.append(np.sum(np.min(d, axis=0)))

    id = np.argmin(metrics)

    return mus[id]


def distance_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Get the antipodally symmetric angular distance between 3D vectors.

    Parameters
    -----------
    a: ndarray with shape (N,3)
    b: ndarray with shape (M,3)

    Returns
    -------
    d: ndarray with shape (N,M)
        Array with distance  
        
    """
    d = a.dot(b.T)
    d = np.arccos(np.abs(d))

    return d


def apsym_kmeans(data, k, nstarts=1):
    metrics = []
    mus = []
    for i in range(nstarts):
        mu_id = np.random.choice(np.arange(len(data)), k, replace=False)
        mu = data[mu_id]
        # while True:
        count = 0
        for t in range(100):
            count += 1
            mu_old = mu.copy()
            # d = distance_3d(mu, data)
            d = mu.dot(data.T)
            labels = np.argmax(np.abs(d), axis=0)
            flip_ids = np.array(d[labels, range(len(labels))] < 0).nonzero()
            data[flip_ids] *= -1
            for j in range(len(mu)):
                data_j = data[labels==j]
                mu[j] = np.sum(data_j, axis=0) / len(data_j)
                mu[j] /= np.sum(mu[j]**2)**0.5
            if np.all(mu == mu_old):
                break
        mus.append(mu)
        metrics.append(np.sum(np.min(d, axis=0)))
    # print(count)
    id = np.argmin(metrics)
    
    return mus[id]