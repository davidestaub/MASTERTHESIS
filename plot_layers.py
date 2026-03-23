import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Define the parameters
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Number of layers
num_layers = 5
layer_height = 2 / num_layers

# Construct the parameter map
param_map = np.zeros_like(X)

for i in range(num_layers):
    lower_bound = -1 + i * layer_height
    upper_bound = lower_bound + layer_height
    sine_boundary = np.sin(np.pi * X) * layer_height / 2
    in_layer = (Y >= lower_bound + sine_boundary) & (Y < upper_bound + sine_boundary)
    param_map[in_layer] = i + 1

# Apply a Gaussian filter with a smaller sigma to reduce the effect on the top and bottom layers
param_map_smoothed = gaussian_filter(param_map, sigma=0.5)

# To maintain the edge values, we can manually set the top and bottom rows to the max and min values respectively
param_map_smoothed[0, :] = param_map[0, :]
param_map_smoothed[-1, :] = param_map[-1, :]

# Re-plot the smoothed parameter map with adjusted smoothing
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, param_map_smoothed, levels=np.linspace(param_map_smoothed.min(), param_map_smoothed.max(), 100), cmap='viridis')
plt.colorbar(label='Layer Value')
plt.title('Adjusted Smoothed Sine Boundary Parameter Map')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()