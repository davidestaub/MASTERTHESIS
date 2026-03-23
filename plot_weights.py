import PINNs
import sys
import torch
import configparser
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
config = configparser.ConfigParser()
if len(sys.argv) < 2:
    raise Exception("Please provide an input folder containing a model path and a config file")
input_folder = sys.argv[1]

config.read(input_folder+"/config.ini")
pth = input_folder +"/model.pth"

my_network = PINNs.MorletWavelet_and_FCN(input_dimension=3, output_dimension=2, n_hidden_layers=int(config['Network']['n_hidden_layers']), neurons=int(config['Network']['n_neurons']),
                       regularization_param=0., regularization_exp=2., retrain_seed=42,activation='tanh',n_1=int(config['Network']['n_1']))

my_network.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))


# Initialize parameters for a WaveletLayer
def initialize_wavelet(wavelet_layer, seed):
    torch.manual_seed(seed)
    nn.init.uniform_(wavelet_layer.amplitude, -1, 1)
    nn.init.uniform_(wavelet_layer.translation, -1, 1)
    nn.init.uniform_(wavelet_layer.scaling, 0.5, 1.5)
    nn.init.uniform_(wavelet_layer.frequency, 0, 1)
    nn.init.uniform_(wavelet_layer.wavelength, -2, 2)
    nn.init.uniform_(wavelet_layer.w_x, -1, 1)
    nn.init.uniform_(wavelet_layer.w_y, -1, 1)

# Generate t, x, y values
t_values = torch.linspace(0, 1, 1000)
x_values = torch.tensor([0.5])  # Near the center
y_values = torch.tensor([0.5])  # Near the center

# Compute wavelet values using the trained parameters
input_values = torch.stack(torch.meshgrid(t_values, x_values, y_values), dim=-1).reshape(-1, 3)
wavelet_trained = my_network.waveletlayer(input_values)

# Save trained parameters
trained_state_dict = my_network.waveletlayer.state_dict().copy()

# Initialize wavelet layer to get untrained values
initialize_wavelet(my_network.waveletlayer, 3)
wavelet_initialized = my_network.waveletlayer(input_values)

# Restore trained parameters
my_network.waveletlayer.load_state_dict(trained_state_dict)

neuron_idx = 3  # Indexing starts from 0, so neuron 4 is at index 3

print("Parameters for Neuron 4:")

# Print the parameters
print(f"Frequency: {my_network.waveletlayer.frequency[neuron_idx].item()}")
print(f"Wavelength: {my_network.waveletlayer.wavelength[neuron_idx].item()}")
print(f"Scaling: {my_network.waveletlayer.scaling[neuron_idx].item()}")
print(f"Translation: {my_network.waveletlayer.translation[neuron_idx].item()}")
print(f"Amplitude: {my_network.waveletlayer.amplitude[neuron_idx].item()}")
print(f"w_x: {my_network.waveletlayer.w_x[neuron_idx].item()}")
print(f"w_y: {my_network.waveletlayer.w_y[neuron_idx].item()}")

# Plot comparison
plt.figure(figsize=(10, 5))
for i in range(my_network.waveletlayer.neurons):
   # plt.plot(t_values, wavelet_initialized[:, i].detach(), label=f"Neuron {i+1} (Initialized)")
    plt.plot(t_values, wavelet_trained[:, i].detach(), label=f"Neuron {i+1} (Trained)", linestyle='--')
#plt.legend(loc='best')
plt.title("Wavelets at x=0.5, y=0.5 - Trained")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


for i in range(my_network.waveletlayer.neurons):
    plt.plot(t_values, wavelet_initialized[:, i].detach(), label=f"Neuron {i+1} (Initialized)")
    #plt.plot(t_values, wavelet_trained[:, i].detach(), label=f"Neuron {i+1} (Trained)", linestyle='--')
#plt.legend(loc='best')
plt.title("Wavelets at x=0.5, y=0.5 - Initialized")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# Define the wavelet functions as before
def ricker_wavelet(t, a=4):
    return (1 - 2 * a ** 2 * t ** 2) * torch.exp(-a ** 2 * t ** 2)


def modulated_ricker_wavelet(t, x, y, scaling, translation, amp, w_x, w_y):

    spatial_term = x * w_x + y * w_y #+ x**2 * w_x + y**2 * w_y
    adjusted_time_translation = t + translation + spatial_term
    wavelet_value = amp * ricker_wavelet(scaling * adjusted_time_translation)
    return wavelet_value


# Initialize the initial parameters based on the given scheme
def init_xavier(neurons):
    torch.manual_seed(3)  # Setting the seed as given

    amplitude_init = torch.nn.init.uniform_(torch.empty(neurons), -1, 1)
    translation_init = torch.nn.init.uniform_(torch.empty(neurons), -1, 1)
    scaling_init = torch.nn.init.uniform_(torch.empty(neurons), 0.5, 1.5)
    w_x_init = torch.nn.init.uniform_(torch.empty(neurons), -2, 2)
    w_y_init = torch.nn.init.uniform_(torch.empty(neurons), -2, 2)

    return amplitude_init, translation_init, scaling_init, w_x_init, w_y_init


num_neurons = my_network.waveletlayer.amplitude.shape[0]

# Get initialized parameters
amplitude_init, translation_init, scaling_init, w_x_init, w_y_init = init_xavier(num_neurons)

# For plotting
t = torch.tensor(0.5)
x_range = torch.linspace(-1, 1, 100)
y_range = torch.linspace(-1, 1, 100)
X, Y = torch.meshgrid(x_range, y_range)

for neuron_idx in range(num_neurons):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(f"2D Scatter plot of Wavelet values for t=0.5, Neuron {neuron_idx + 1}")

    # Extract trained parameters
    scaling_trained = my_network.waveletlayer.scaling[neuron_idx]
    translation_trained = my_network.waveletlayer.translation[neuron_idx]
    amp_trained = my_network.waveletlayer.amplitude[neuron_idx]
    w_x_trained = my_network.waveletlayer.w_x[neuron_idx]
    w_y_trained = my_network.waveletlayer.w_y[neuron_idx]

    for idx, (scaling, translation, amp, w_x, w_y, ax, title) in enumerate([(scaling_init[neuron_idx],
                                                                             translation_init[neuron_idx],
                                                                             amplitude_init[neuron_idx],
                                                                             w_x_init[neuron_idx], w_y_init[neuron_idx],
                                                                             axes[0], "Initialized"),
                                                                            (scaling_trained, translation_trained,
                                                                             amp_trained, w_x_trained, w_y_trained,
                                                                             axes[1], "Trained")]):
        values = modulated_ricker_wavelet(t, X, Y, scaling, translation, amp, w_x, w_y)
        ax.scatter(X.numpy().flatten(), Y.numpy().flatten(), c=values.detach().numpy().flatten(), cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()





