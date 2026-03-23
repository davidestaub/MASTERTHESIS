import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # Assuming scienceplots is correctly installed and available


def extract_neurons(folder_name):
    print(folder_name.split('_'))
    return int(folder_name.split('_')[10])

def extract_layers(folder_name):
    return int(folder_name.split('_')[9])

def extract_t1(folder_name):
    'PINN_t1_experiment_100000_200_1.0_100_100_100_5_128_tanh_0.01_PINNS_1_constant'
    return float(folder_name.split('_')[-4])


plt.style.use(['science', 'ieee'])
plt.rcParams.update({'font.size': 20})

# Modify the functions to accept an ax parameter for plotting
def plot_n_neurons_vs_average_l2(folder_path):
    neuron_values = []
    error_values = []
    for subdir, _, _ in os.walk(folder_path):
        file_path = os.path.join(subdir, 'averages.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'average_l2' in df.columns:
                neuron = extract_neurons(subdir)
                layers = extract_layers(subdir)
                if int(layers) == 5:
                    average_l2 = df['average_l2'].values[0]
                    neuron_values.append(neuron)
                    error_values.append(average_l2)

    sorted_pairs = sorted(zip(neuron_values, error_values), key=lambda x: x[0])
    neuron_values, error_values = zip(*sorted_pairs)

    plt.figure(figsize=(6, 5))
    plt.scatter(neuron_values, error_values, color='purple')
    plt.plot(neuron_values, error_values, color='purple')
    plt.xlabel('$N_{Neurons}$')
    plt.ylabel('Average $L_2$ error')
    #plt.title('$N_{Neurons}$ vs. $L_2$ error')
    plt.ylim([0.999, 101.0])
    plt.yscale('log')
    plt.tight_layout(pad=1.0)
    plt.savefig('hyper_neurons.pdf', dpi=500)
    plt.show()
    plt.close()

def plot_n_layers_vs_average_l2(folder_path):
    layer_values = []
    error_values = []
    for subdir, _, _ in os.walk(folder_path):
        file_path = os.path.join(subdir, 'averages.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'average_l2' in df.columns:
                neuron = extract_neurons(subdir)
                layers = extract_layers(subdir)
                if int(neuron) == 128 or int(neuron) == 127:
                    average_l2 = df['average_l2'].values[0]
                    layer_values.append(layers)
                    error_values.append(average_l2)

    sorted_pairs = sorted(zip(layer_values, error_values), key=lambda x: x[0])
    layer_values, error_values = zip(*sorted_pairs)

    plt.figure(figsize=(6, 5))
    plt.scatter(layer_values, error_values, color='purple')
    plt.plot(layer_values,error_values,color='purple')
    plt.xlabel('$N_{Layers}$')
    plt.ylabel('Average $L_2$ error')
    #plt.title('$N_{Layers}$ vs. $L_2$ error')
    plt.ylim([0.999, 101.0])
    plt.yscale('log')
    plt.tight_layout(pad=1.0)
    plt.savefig('hyper_layers.pdf', dpi=500)
    plt.show()
    plt.close()


plot_n_neurons_vs_average_l2('../../Results/Elastic/hyper')
plot_n_layers_vs_average_l2('../../Results/Elastic/hyper')
