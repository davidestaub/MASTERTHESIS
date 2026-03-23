import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.colors as mcolors

def generate_mixture(num_mixtures=100, amplitude=0.3, width=0.15):
    np.random.seed(42)
    torch.manual_seed(42)
    return torch.tensor(np.concatenate([
        np.random.uniform(-1, 1, (num_mixtures, 2)), # location
        width * np.ones((num_mixtures, 1)), # width
        amplitude * np.random.uniform(10, 50, (num_mixtures, 1)), # amplitude
    ], axis=1), dtype=torch.float32)

def compute_param(X, Y, mixture):
    print("X.shape = ",X.shape)
    param_mixture = torch.exp(-0.5 * ((X.unsqueeze(-1) - mixture[:, 0])**2 + (Y.unsqueeze(-1) - mixture[:, 1])**2) / (mixture[:, 2]**2))
    param = torch.sum(mixture[:, 3] * param_mixture, dim=-1)
    return param

def generate_acoustic_mixture(num_mixtures=100, width=0.15):
    np.random.seed(42)
    torch.manual_seed(42)
    return torch.tensor(np.concatenate([
        np.random.uniform(-1, 1, (num_mixtures, 2)), # location
        width * np.ones((num_mixtures, 1)), # width
        0.25 * np.random.uniform(0.9, 1.0, (num_mixtures, 1)), # velocity
    ], axis=1), dtype=torch.float32)

def compute_acoustic_param(X, Y, mixture):
    param_mixture = torch.exp(-0.5 * ((X.unsqueeze(-1) - mixture[:, 0])**2 + (Y.unsqueeze(-1) - mixture[:, 1])**2) / (mixture[:, 2]**2))
    param = torch.sum(mixture[:, 3] * param_mixture, dim=-1)
    return param

def compute_image(mixture, res=100):
    x_vals = torch.linspace(-1, 1, res)
    y_vals = torch.linspace(-1, 1, res)
    X, Y = torch.meshgrid(x_vals, y_vals)
    image = compute_param(X, Y, mixture)
    return image


def sigmoid(x, a=1, b=0):
    return 1 / (1 + torch.exp(-a * (x - b)))


def compute_lambda_mu_layers(X, Y, num_layers, smoothing_fraction=0.2):
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    np.random.seed(42)
    torch.manual_seed(42)

    min_val = 10
    max_val = 50
    layer_thickness = X.shape[0] // num_layers
    smoothing_thickness = int(layer_thickness * smoothing_fraction)

    lambda_s = torch.zeros_like(X)
    mu_s = torch.zeros_like(Y)

    current_lambda_val = np.random.uniform(min_val, max_val)
    current_mu_val = np.random.uniform(min_val, max_val)

    for i in range(num_layers):
        next_lambda_val = np.random.uniform(min_val, max_val)
        next_mu_val = np.random.uniform(min_val, max_val)

        start_idx = i * layer_thickness
        end_idx = (i + 1) * layer_thickness if i != num_layers - 1 else X.shape[0]

        # Set the base value for this layer
        lambda_s[start_idx:end_idx, :] = current_lambda_val
        mu_s[start_idx:end_idx, :] = current_mu_val

        # Sigmoid smoothing at the upper boundary of this layer
        interp_start = end_idx - smoothing_thickness
        interp_end = end_idx

        x = torch.linspace(-5, 5, steps=interp_end - interp_start)
        transition = sigmoid(x)

        interpolated_lambda = (1 - transition) * current_lambda_val + transition * next_lambda_val
        interpolated_mu = (1 - transition) * current_mu_val + transition * next_mu_val

        # Make these interpolated values compatible for 2D assignment
        interpolated_lambda = interpolated_lambda[:, None].expand(-1, X.shape[1])
        interpolated_mu = interpolated_mu[:, None].expand(-1, X.shape[1])

        lambda_s[interp_start:interp_end, :] = interpolated_lambda
        mu_s[interp_start:interp_end, :] = interpolated_mu

        current_lambda_val = next_lambda_val
        current_mu_val = next_mu_val

    return lambda_s.squeeze(), mu_s.squeeze()

def compute_lambda_mu_layers_accoustic_slow(X, Y, num_layers, smoothing_fraction=0.2):
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    np.random.seed(42)
    torch.manual_seed(42)

    min_val = 1
    max_val = 2
    layer_thickness = X.shape[0] // num_layers
    smoothing_thickness = int(layer_thickness * smoothing_fraction)

    lambda_s = torch.zeros_like(X)
    mu_s = torch.zeros_like(Y)

    current_lambda_val = np.random.uniform(min_val, max_val)
    current_mu_val = np.random.uniform(min_val, max_val)

    for i in range(num_layers):
        next_lambda_val = np.random.uniform(min_val, max_val)
        next_mu_val = np.random.uniform(min_val, max_val)

        start_idx = i * layer_thickness
        end_idx = (i + 1) * layer_thickness if i != num_layers - 1 else X.shape[0]

        # Set the base value for this layer
        lambda_s[start_idx:end_idx, :] = current_lambda_val
        mu_s[start_idx:end_idx, :] = current_mu_val

        # Sigmoid smoothing at the upper boundary of this layer
        interp_start = end_idx - smoothing_thickness
        interp_end = end_idx

        x = torch.linspace(-5, 5, steps=interp_end - interp_start)
        transition = sigmoid(x)

        interpolated_lambda = (1 - transition) * current_lambda_val + transition * next_lambda_val
        interpolated_mu = (1 - transition) * current_mu_val + transition * next_mu_val

        # Make these interpolated values compatible for 2D assignment
        interpolated_lambda = interpolated_lambda[:, None].expand(-1, X.shape[1])
        interpolated_mu = interpolated_mu[:, None].expand(-1, X.shape[1])
        print(interpolated_lambda.shape)

        lambda_s[interp_start:interp_end, :] = interpolated_lambda.unsqueeze(-1)
        mu_s[interp_start:interp_end, :] = interpolated_mu.unsqueeze(-1)

        current_lambda_val = next_lambda_val
        current_mu_val = next_mu_val

    return lambda_s.squeeze(), mu_s.squeeze()

def compute_lambda_mu_layers_accoustic_fast(X, Y, num_layers, smoothing_fraction=0.2):
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    np.random.seed(42)
    torch.manual_seed(42)

    min_val = 1
    max_val = 5
    layer_thickness = X.shape[0] // num_layers
    smoothing_thickness = int(layer_thickness * smoothing_fraction)

    lambda_s = torch.zeros_like(X)
    mu_s = torch.zeros_like(Y)

    current_lambda_val = np.random.uniform(min_val, max_val)
    current_mu_val = np.random.uniform(min_val, max_val)

    for i in range(num_layers):
        next_lambda_val = np.random.uniform(min_val, max_val)
        next_mu_val = np.random.uniform(min_val, max_val)

        start_idx = i * layer_thickness
        end_idx = (i + 1) * layer_thickness if i != num_layers - 1 else X.shape[0]

        # Set the base value for this layer
        lambda_s[start_idx:end_idx, :] = current_lambda_val
        mu_s[start_idx:end_idx, :] = current_mu_val

        # Sigmoid smoothing at the upper boundary of this layer
        interp_start = end_idx - smoothing_thickness
        interp_end = end_idx

        x = torch.linspace(-5, 5, steps=interp_end - interp_start)
        transition = sigmoid(x)

        interpolated_lambda = (1 - transition) * current_lambda_val + transition * next_lambda_val
        interpolated_mu = (1 - transition) * current_mu_val + transition * next_mu_val

        # Make these interpolated values compatible for 2D assignment
        interpolated_lambda = interpolated_lambda[:, None].expand(-1, X.shape[1])
        interpolated_mu = interpolated_mu[:, None].expand(-1, X.shape[1])

        lambda_s[interp_start:interp_end, :] = interpolated_lambda
        mu_s[interp_start:interp_end, :] = interpolated_mu

        current_lambda_val = next_lambda_val
        current_mu_val = next_mu_val

    return lambda_s.squeeze(), mu_s.squeeze()



xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0
tmin = 0.0
tmax = 1.0

domain_extrema = torch.tensor(
    [[tmin, tmax],  # Time dimension
     [xmin, xmax],
     [ymin, ymax],
     ])  # Space dimension

def manual_gaussian_smoothing(map, kernel_size=10):
    """
    Apply manual Gaussian-like smoothing to a 2D map.

    :param map: The 2D map to be smoothed.
    :param kernel_size: Size of the smoothing kernel, must be an odd number.
    :return: Smoothed 2D map.
    """
    padded_map = np.pad(map, pad_width=kernel_size // 2, mode='edge')
    smoothed_map = np.copy(map)

    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            kernel = padded_map[i:i+kernel_size, j:j+kernel_size]
            smoothed_map[i, j] = np.mean(kernel)

    return smoothed_map


def create_sine_layer_map(X, Y, num_layers=5, min_val=10, max_val=50, kernel_size=10):
    """
    Creates a horizontal sine layer map with increasing layer values and manual Gaussian-like smoothing.

    :param X: A 2D tensor or numpy array representing the X coordinates.
    :param Y: A 2D tensor or numpy array representing the Y coordinates.
    :param num_layers: The number of layers to create in the map.
    :param min_val: The minimum value for lambda and mu.
    :param max_val: The maximum value for lambda and mu.
    :param kernel_size: The size of the kernel used for smoothing.
    :return: A tuple of 2D arrays (lambda_map, mu_map), representing the smoothed values of lambda and mu.
    """

    # Normalizing X and Y
    X_normalized = 2 * ((X - X.min()) / (X.max() - X.min())) - 1
    Y_normalized = 2 * ((Y - Y.min()) / (Y.max() - Y.min())) - 1

    np.random.seed(42)
    layer_height = 2 / num_layers
    lambda_map = np.zeros_like(X_normalized)
    mu_map = np.zeros_like(Y_normalized)

    for i in range(num_layers):
        # Define the bounds for this layer
        lower_bound = -1 + i * layer_height
        upper_bound = lower_bound + layer_height

        # Sine function for horizontal layers
        sine_boundary = np.sin(np.pi * Y_normalized) * layer_height / 2

        # Assign increasing values to the current layer
        lambda_value = min_val + (max_val - min_val) * (i / (num_layers - 1))
        mu_value = min_val + (max_val - min_val) * (i / (num_layers - 1))

        # Determine points in the current layer
        in_layer = (X_normalized >= lower_bound + sine_boundary) & (X_normalized < upper_bound + sine_boundary)
        lambda_map[in_layer] = lambda_value
        mu_map[in_layer] = mu_value

    # Ensure that all points are assigned to a layer
    # Get the middle index on the Y-axis
    middle_index = Y.shape[0] // 2

    print(lambda_map.shape)
    print(lambda_map)
    for i in range(0,lambda_map.shape[0]):
        for j in range(0,lambda_map.shape[1]):
            if lambda_map[i,j] == 0 and i >= lambda_map.shape[0]/2:
                #print("upper",i,j)
                lambda_map[i, j] = min_val
            elif lambda_map[i,j] == 0 and i < lambda_map.shape[0]/2:
                lambda_map[i, j] = max_val
                #print("lower", i, j)
            elif i>=(lambda_map.shape[0] * (7/8)) and j>=(lambda_map.shape[0] * (7/8)):
                lambda_map[i, j] = max_val


    # Fill the top half with 50 where lambda_map is 0
    #lambda_map[:middle_index][lambda_map[:middle_index] == 0] = 50
   # mu_map[:middle_index][mu_map[:middle_index] == 0] = 50

    # Fill the bottom half with 10 where lambda_map is 0
    #lambda_map[middle_index:][lambda_map[middle_index:] == 0] = 10
    #mu_map[middle_index:][mu_map[middle_index:] == 0] = 10
    # Apply manual Gaussian smoothing
    lambda_map_smoothed = manual_gaussian_smoothing(lambda_map, kernel_size=kernel_size)
    mu_map_smoothed = manual_gaussian_smoothing(mu_map, kernel_size=kernel_size)

    for i in range(0, lambda_map.shape[0]):
        for j in range(0, lambda_map.shape[1]):
            if i >= (lambda_map.shape[0] * (7 / 8)) and j >= (lambda_map.shape[0] * (7 / 8)):
                lambda_map_smoothed[i, j] = max_val
    mu_map_smoothed = lambda_map_smoothed
    return lambda_map_smoothed, mu_map_smoothed

def extract_values_from_2d_maps(input_s, num_layers=5, min_val=10, max_val=50, kernel_size=10):
    """
    Extracts lambda and mu values from 2D sine layer maps at specified points.

    :param input_s: A 2D tensor containing (x, y) coordinates.
    :param num_layers, min_val, max_val, kernel_size: Parameters for the sine layer map.
    :return: Two tensors of extracted lambda and mu values.
    """
    # Create a 2D grid covering the range of x and y
    x_vals = torch.linspace(-1, 1, 1024)
    y_vals = torch.linspace(-1, 1, 1024)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')

    # Generate the 2D sine layer map and apply smoothing
    lambda_map, mu_map = create_sine_layer_map(X.numpy(), Y.numpy(), num_layers, min_val, max_val, kernel_size)
    lambda_map_tensor = torch.tensor(manual_gaussian_smoothing(lambda_map, kernel_size)).unsqueeze(0).unsqueeze(0)
    mu_map_tensor = torch.tensor(manual_gaussian_smoothing(mu_map, kernel_size)).unsqueeze(0).unsqueeze(0)

    # Normalize input_s coordinates to be in the range [-1, 1] for grid sampling
    x_normalized = 2 * ((input_s[:, 1] - input_s[:, 1].min()) / (input_s[:, 1].max() - input_s[:, 1].min())) - 1
    y_normalized = 2 * ((input_s[:, 2] - input_s[:, 2].min()) / (input_s[:, 2].max() - input_s[:, 2].min())) - 1
    normalized_coords = torch.stack([x_normalized, y_normalized], dim=-1).unsqueeze(0).unsqueeze(0)

    # Extract lambda and mu values at specified points using grid sampling
    lambda_extracted_values = F.grid_sample(lambda_map_tensor, normalized_coords, mode='bilinear', align_corners=True)
    mu_extracted_values = F.grid_sample(mu_map_tensor, normalized_coords, mode='bilinear', align_corners=True)

    return lambda_extracted_values.squeeze(), mu_extracted_values.squeeze()

def direct_extract_values(input_s, num_layers=5, min_val=10, max_val=50):
    """
    Vectorized calculation of lambda values for a batch of (x, y) coordinates.

    :param input_s: A 2D tensor containing (x, y) coordinates normalized between [-1, 1].
    :param num_layers: The number of sine layers.
    :param min_val: The minimum lambda value.
    :param max_val: The maximum lambda value.
    :return: A tensor of lambda values for the given coordinates.
    """
    # Ensure input_s is a float tensor for proper division
    input_s = input_s.float()

    # Layer height based on the number of layers
    layer_height = 2.0 / num_layers

    # Sine function to modulate the layer boundaries
    # Assuming y coordinates are the second column of input_s
    y_sine = torch.sin(np.pi * input_s[:, 2]) * (layer_height / 2.0)

    # Layer index calculation adjusted by sine modulation
    # Assuming x coordinates are the first column of input_s
    layer_index = torch.floor((input_s[:, 1] - y_sine + 1) / layer_height).clamp(0, num_layers - 1)

    # Linear interpolation of lambda values based on layer index
    lambda_values = min_val + (max_val - min_val) * (layer_index / (num_layers - 1))

    return lambda_values

# Example usage
#input_s = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Example input coordinates
#lambda_values, mu_values = direct_extract_values(input_s)

def convert(tens):
    #assert (tens.shape[1] == self.domain_extrema.shape[0])
    return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]

