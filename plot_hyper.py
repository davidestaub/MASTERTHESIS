import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use(['science','ieee'])
plt.rcParams.update({'font.size': 20})


def extract_neurons(folder_name):
    print(folder_name.split('_'))
    return int(folder_name.split('_')[10])

def extract_layers(folder_name):
    return int(folder_name.split('_')[9])

def extract_t1(folder_name):
    'PINN_t1_experiment_100000_200_1.0_100_100_100_5_128_tanh_0.01_PINNS_1_constant'
    return float(folder_name.split('_')[-4])



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

    plt.scatter(neuron_values, error_values,color='purple')
    #plt.plot(np.unique(neuron_values),np.poly1d(np.polyfit(neuron_values, error_values, 4))(np.unique(neuron_values)),color='purple')


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
                print("layer = ",layers)
                if int(neuron) == 128 or int(neuron)==127:
                    average_l2 = df['average_l2'].values[0]
                    layer_values.append(layers)
                    error_values.append(average_l2)
                else:
                    print(int(neuron))
    plt.scatter(layer_values, error_values, color='purple')
    #plt.plot(np.unique(layer_values), np.poly1d(np.polyfit(layer_values, error_values, 4))(np.unique(layer_values)),
             #color='purple')



def plot_t1_vs_average_l2(folder_path):
    t1_values = []
    error_values = []
    for subdir, _, _ in os.walk(folder_path):
        print(subdir)
        file_path = os.path.join(subdir, 'averages.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'average_l2' in df.columns:
                t1 = extract_t1(subdir)
                average_l2 = df['average_l2'].values[0]
                print("t1= ",t1)
                t1_values.append(t1)
                error_values.append(average_l2)

    sorted_pairs = sorted(zip(t1_values, error_values), key=lambda x: x[0])
    t1_values, error_values = zip(*sorted_pairs)

    plt.scatter(t1_values, error_values, color='purple')
    plt.plot(t1_values, error_values, color='purple')
    #plt.plot(np.unique(layer_values), np.poly1d(np.polyfit(layer_values, error_values, 4))(np.unique(layer_values)),
             #color='purple')


# Plot for 'wavelet' and 'vanilla' folders
plot_n_layers_vs_average_l2('../../Results/Elastic/hyper')
plt.xlabel('$N_{Layers}$')
plt.ylabel('Average $L_2$ error')
plt.title('$N_{Layers}$ vs. $L_2$ error - $N_{Neurons}$ = 128')
plt.tight_layout(pad=1.0)
plt.ylim([0.999,101.0])
plt.yscale('log')

#plt.yticks([0.0,1.0,10.0,100.0])
plt.savefig('hyper_layers.png',dpi=500)
plt.show()
plt.close()

#plt.savefig('strong_scaling_vanilla_vs_best.png',dpi=500)




plot_n_neurons_vs_average_l2('../../Results/Elastic/hyper')
plt.xlabel('$N_{Neurons}$')
plt.ylabel('Average $L_2$ error')
plt.title('$N_{Neurons}$ vs. $L_2$ error - $N_{Layers}$ = 5')
plt.tight_layout(pad=1.0)
plt.ylim([0.999,101.0])
plt.yscale('log')
plt.savefig('hyper_neurons.png',dpi=500)
plt.show()
plt.close()


#plt.savefig('strong_scaling_vanilla_vs_best.png',dpi=500)

plt.figure(figsize=(10,5))
plot_t1_vs_average_l2('../../Results/Elastic/t1_experiment/[tensor(0.)_tensor(0.)]')
plt.xlabel('$t_{1}$')
plt.ylabel('Average $L_2$ error')
#plt.title('$t_{1}$ vs. $L_2$ error')
plt.tight_layout(pad=1.0)
plt.ylim([0.999,101.0])
plt.yscale('log')

plt.savefig('hyper_t1.pdf',dpi=400)
plt.show()
plt.close()

print("hello")