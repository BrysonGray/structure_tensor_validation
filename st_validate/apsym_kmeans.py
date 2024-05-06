#!/usr/bin/env python

""" 
Compute k-means on antipodally symmetric (apsym) directions. Each sample gets equal weight.

Author: Bryson Gray
2023
"""
import warnings
import numpy as np
from numbers import Integral, Real

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
)
from sklearn.cluster import _kmeans
from sklearn.cluster._k_means_common import _is_same_clustering

def _apsym_lloyd_iter(X, centers, update_centers=True):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, 3).
        Observations to cluster. Assumes sample directions are already unit normalized.

    centers : ndarray of shape (n_clusters, 3)
        Cluster centers
    
    update_centers : bool
        If True, perform M step and return centers_new and center_shift. (default True)
    
    Returns
    -------
    Xnew : ndarray of shape (n_samples, 3)
        Input directions multiplied by -1 as needed to cluster them with their nearest center.

    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    centers_new : ndarray of shape (n_clusters, n_features)

    center_shift : ndarray of shape (n_clusters)
        Distance between old and new centers.

    dist : ndarray of shape (n_samples, n_clusters)

    inertia : float
        The sum of squared distances of samples to their closest cluster center.
    """
    # cosine distance of each input direction and its antipode to each center
    cosine_dist = X[:,None]@centers.T
    cosine_dist = np.squeeze(cosine_dist)
    dist = 1 - np.abs(cosine_dist)
    if centers.shape[0] == 1:
        labels = np.zeros(dist.shape[0], dtype=np.int32)
        cosine_dist = cosine_dist[:,None]
    else:
        labels = np.argmin(dist, axis=-1).astype(np.int32)
    flip_ids = np.asarray(cosine_dist[range(len(labels)),labels] < 0).nonzero() # the indices of the directions whose cosine distance to their assigned center is negative
    Xnew = X.copy()
    Xnew[flip_ids] *= -1
    if update_centers:
        centers_new = np.zeros_like(centers, dtype=float)
        for k in range(len(centers)):
            cluster = Xnew[labels==k]
            if len(cluster)==0:
                # if the cluster is empty, set the center to k-1
                centers_new[k] = centers_new[k-1]
            else:
                centers_new[k] = np.sum(cluster, axis=0)/cluster.shape[0]
                centers_new[k] = centers_new[k] / row_norms(centers_new[None,k])[:,None]
                center_shift = row_norms((centers_new - centers))

        return Xnew, labels, dist, centers_new, center_shift

    return Xnew, labels, dist

def _inertia(dist):
    return np.sum(dist)

def _kmeans_single_lloyd(
        X,
        centers_init,
        max_iter=300,
        verbose=False,
        tol=1e-4,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features) # TODO: never sparse
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,) # TODO: Weights not yet implemented.
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1 # TODO: Not using multithreading for now
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    # weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    # if sp.issparse(X):
    #     lloyd_iter = lloyd_iter_chunked_sparse
    #     _inertia = _inertia_sparse
    # else:
        # lloyd_iter = lloyd_iter_chunked_dense
        # _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubscription.
    # with threadpool_limits(limits=1, user_api="blas"):
    # TODO: Not using multithreading. Possibly add this in later
    for i in range(max_iter):
        # lloyd_iter(
        #     X,
        #     sample_weight,
        #     centers,
        #     centers_new,
        #     weight_in_clusters,
        #     labels,
        #     center_shift,
        #     n_threads,
        # )
        X, labels, dist, centers_new, center_shift = _apsym_lloyd_iter(X, centers)

        if verbose:
            inertia = _inertia(dist)
            print(f"Iteration {i}, inertia {inertia}.")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        # lloyd_iter(
        #     X,
        #     sample_weight,
        #     centers,
        #     centers,
        #     weight_in_clusters,
        #     labels,
        #     center_shift,
        #     n_threads,
        #     update_centers=False,
        # )
        X, labels, dist = _apsym_lloyd_iter(X, centers, update_centers=False)

    inertia = _inertia(dist)

    return labels, inertia, centers, i + 1

class APSymKMeans():
    """K-Means clustering.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : 'auto' or int, default=1
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    copy_x : bool, default=True # TODO: remove this
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd" # TODO: remove. We will only use lloyd
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    """
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        # copy_x=True, # TODO: remove
        # algorithm="lloyd", # TODO: remove
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _tolerance(self, X, tol):
        """Return a tolerance which is dependent on the dataset."""
        if tol == 0:
            return 0
        variances = np.var(X, axis=0)
        return np.mean(variances) * tol
    
    def _check_params_vs_input(self, X, default_n_init=10):
        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = self._tolerance(X, self.tol)

        # n-init
        # TODO(1.4): Remove
        if self.n_init == "auto":
            if isinstance(self.init, str) and self.init == "k-means++":
                self.n_init = 1
            elif isinstance(self.init, str) and self.init == "random":
                self.n_init = default_n_init
            elif callable(self.init):
                self.n_init = default_n_init
            else:  # array-like
                self.n_init = 1

        if _is_arraylike_not_scalar(self.init) and self.n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.__class__.__name__} instead of "
                    f"n_init={self.n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self.n_init = 1

    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        init_size=None,
        n_centroids=None,
        # sample_weight=None,
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters)

        sample_weight : ndarray of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            # sample_weight = sample_weight[init_indices]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans._kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                # sample_weight=sample_weight,
            )
            centers = centers / row_norms(centers)[:,None]
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                # p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        # if sp.issparse(centers):
        #     centers = centers.toarray()

        return centers

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) # TODO: this will never be sparse.
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # TODO: probably not necessary. check to make sure
        # X = self._validate_data(
        #     X,
        #     accept_sparse="csr",
        #     dtype=[np.float64, np.float32],
        #     order="C",
        #     copy=self.copy_x,
        #     accept_large_sparse=False,
        # )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        # self._n_threads = _openmp_effective_n_threads() # TODO: Remove. Use single thread.

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        # if not sp.issparse(X):
        #     X_mean = X.mean(axis=0)
        #     # The copy was already done above
        #     X -= X_mean

        #     if init_is_array_like:
        #         init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # if self._algorithm == "elkan":
        #     kmeans_single = _kmeans_single_elkan
        # else:
        #     kmeans_single = _kmeans_single_lloyd
        #     self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self.n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                # sample_weight=sample_weight,
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            # labels, inertia, centers, n_iter_ = kmeans_single(
            #     X,
            #     sample_weight,
            #     centers_init,
            #     max_iter=self.max_iter,
            #     verbose=self.verbose,
            #     tol=self._tol,
            #     n_threads=self._n_threads,
            # )
            labels, inertia, centers, n_iter_ = _kmeans_single_lloyd(
                X,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        # if not sp.issparse(X):
        #     if not self.copy_x:
        #         X += X_mean
        #     best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
