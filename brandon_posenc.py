# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:50:36 2024

@author: vkarmarkar
"""

import numpy as np

safe_sin = lambda x: np.sin(x % (100 * np.pi))

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: np.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: np.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = np.array([2**i for i in range(deg)])
    xb = np.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(np.concatenate([xb, xb + 0.5 * np.pi], axis=-1))
    return np.concatenate([x] + [four_feat], axis=-1)

# Define the range for each dimension
x_range = np.linspace(-np.pi, np.pi, 10)
y_range = np.linspace(-np.pi, np.pi, 10)
z_range = np.linspace(-np.pi, np.pi, 10)

# Generate the 3D meshgrid
x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

# Create the coordinates array
coords = np.stack([x, y, z], axis=-1)
coords = coords.reshape((-1, coords.shape[-1]))
print(coords.shape)

coords_encoded = posenc(coords, 4)
print(coords_encoded.shape)