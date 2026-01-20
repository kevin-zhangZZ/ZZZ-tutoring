"""
Pure numpy functions for Monte Carlo area estimation.
No Streamlit imports - this module can be tested independently.
"""

import numpy as np


def curve(x):
    """
    Compute y = sin(πx) for given x values.
    
    Parameters:
    -----------
    x : array-like
        Input x values
    
    Returns:
    --------
    y : ndarray
        Output y = sin(πx)
    """
    return np.sin(np.pi * x)


def true_area():
    """
    Compute the true area under y = sin(πx) on [0,1].
    
    Returns:
    --------
    float
        True area = 2/π
    """
    return 2.0 / np.pi


def generate_points(rng, n_new):
    """
    Generate n_new random points uniformly in [0,1]×[0,1]
    and determine which are under the curve.
    
    Parameters:
    -----------
    rng : numpy.random.Generator
        Random number generator
    n_new : int
        Number of points to generate
    
    Returns:
    --------
    x : ndarray
        x coordinates of points (shape: (n_new,))
    y : ndarray
        y coordinates of points (shape: (n_new,))
    under : ndarray (bool)
        True if point is under the curve (y < sin(πx))
    """
    x = rng.random(n_new)
    y = rng.random(n_new)
    y_curve = curve(x)
    under = y < y_curve
    return x, y, under


def running_estimate(under_all):
    """
    Compute running estimate of area: cumulative mean of under_all.
    
    Parameters:
    -----------
    under_all : ndarray (bool)
        Boolean array indicating which points are under the curve
    
    Returns:
    --------
    estimates : ndarray
        Running estimates (cumulative mean) for each n from 1 to len(under_all)
    """
    n_total = len(under_all)
    if n_total == 0:
        return np.array([])
    # Cumulative sum, then divide by array [1, 2, 3, ..., n_total]
    cumsum = np.cumsum(under_all.astype(float))
    n_vals = np.arange(1, n_total + 1, dtype=float)
    return cumsum / n_vals


def choose_n_values(n_max, steps):
    """
    Choose a sorted list of unique integer n values for convergence plotting.
    Ensures we get approximately 'steps' points, spaced roughly logarithmically.
    
    Parameters:
    -----------
    n_max : int
        Maximum n value
    steps : int
        Approximate number of points to include
    
    Returns:
    --------
    n_vals : ndarray
        Sorted unique integers from 1 to n_max
    """
    if n_max <= steps:
        return np.arange(1, n_max + 1, dtype=int)
    
    # Generate log-spaced indices
    log_indices = np.logspace(0, np.log10(n_max), steps)
    n_vals = np.unique(np.round(log_indices).astype(int))
    # Ensure we include 1 and n_max
    n_vals = np.union1d([1], n_vals)
    n_vals = np.union1d(n_vals, [n_max])
    # Clamp to valid range
    n_vals = n_vals[(n_vals >= 1) & (n_vals <= n_max)]
    return n_vals


def estimate_at_n(under_all, n):
    """
    Compute Monte Carlo estimate using first n points.
    
    Parameters:
    -----------
    under_all : ndarray (bool)
        Boolean array of points under the curve
    n : int
        Number of points to use (first n from under_all)
    
    Returns:
    --------
    float
        Estimate = (# points under curve) / n
    """
    if n == 0 or len(under_all) == 0:
        return 0.0
    n_use = min(n, len(under_all))
    return np.mean(under_all[:n_use].astype(float))


def downsample_indices(n_total, max_display):
    """
    Randomly select indices for downsampling display.
    
    Parameters:
    -----------
    n_total : int
        Total number of points available
    max_display : int
        Maximum number of points to display
    
    Returns:
    --------
    indices : ndarray
        Sorted random indices (length min(n_total, max_display))
    """
    if n_total <= max_display:
        return np.arange(n_total, dtype=int)
    return np.sort(np.random.choice(n_total, size=max_display, replace=False))
