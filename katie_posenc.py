# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:52:02 2024

@author: vkarmarkar
"""

import numpy as np
import torch.nn as nn

safe_sin = lambda x: np.sin(x % (100 * np.pi))

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
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

class MLP(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
  
    @nn.compact
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)

        return out