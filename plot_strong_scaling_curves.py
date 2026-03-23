import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use(['science','ieee'])
plt.rcParams.update({'font.size': 20})

def extract_sigma(folder_name):
    return float(folder_name.split('_')[-1])

custom_order_and_labels = [
        ('PINN', 'PINN-tanh'),
        ('FCN_ALL_PARAMS_WAVELET_FCN','$\it{wED}$-PINN'),
    ]

color_mapping = {
    'FCN_ALL_PARAMS_WAVELET': 'blue',
    'FCN_ALL_PARAMS_WAVELET_FCN': 'orange',
    'PLANE_WAVE_FCN': 'green',
    'FCN_ALL_PARAMS_PLANEWAVE': 'red',
    'PINN': 'purple',
    'FCN_AMPLITUDE_WAVELET': 'brown',
    'FCN_ALL_PARAMS_PLANEWAVE_FCN': 'pink',
    'SIREN': 'grey',
    'FCN_AMPLITUDE_PLANEWAVE': 'lightgreen',
    'MORLET_WAVELET_FCN': 'lightblue'
}

def plot_sigma_vs_average_l2(folder_path, label,cut):
    nn_type = custom_order_and_labels[label][0]
    sigma_values = []
    average_l2_values = []

    for subdir, _, _ in os.walk(folder_path):
        file_path = os.path.join(subdir, 'averages.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'average_l2' in df.columns:
                sigma = extract_sigma(subdir)
                if cut:
                    if sigma >= 0.06:
                        average_l2 = df['average_l2'].values[0]
                        sigma_values.append(sigma)
                        average_l2_values.append(average_l2)
                else:
                    average_l2 = df['average_l2'].values[0]
                    sigma_values.append(sigma)
                    average_l2_values.append(average_l2)

                if sigma ==0.01:
                    print("HEHEHEHEHE")


    plt.scatter(sigma_values, average_l2_values, label=nn_type, color=color_mapping[nn_type])
    plt.plot(np.unique(sigma_values), np.poly1d(np.polyfit(sigma_values, average_l2_values, 4))(np.unique(sigma_values)),color=color_mapping[nn_type])


    #plt.yscale('log')
plt.figure(figsize=(8,5))
# Plot for 'wavelet' and 'vanilla' folders
plot_sigma_vs_average_l2('../../Results/Elastic/strong_scaling/256/vanilla', 0,True)
plot_sigma_vs_average_l2('../../Results/Elastic/strong_scaling/256/wavelet', 1,True)




plt.xlabel('$\sigma$')
plt.ylabel('Average $L_2$ error')
plt.title('Varing Source size $\sigma$, PINN-tanh vs. $\it{wED}$-PINN')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                             markerfacecolor=color_mapping[nn_type], markersize=10)
                  for nn_type, label in custom_order_and_labels]
plt.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2)
plt.tight_layout()
plt.savefig('strong_scaling_vanilla_vs_best.pdf',dpi=400)
plt.show()
print("hello")

plt.figure(figsize=(8,5))
plot_sigma_vs_average_l2('../../Results/Elastic/strong_scaling/256/vanilla', 0,True)

plt.xlabel('$\sigma$')

plt.ylabel('Average $L_2$ error')
plt.title('Varing Source size, $\sigma \in [0.06,0.2]$')
plt.tight_layout()
plt.savefig('strong_scaling_cut.pdf')
plt.show()
print("hello")
plt.close()


plt.figure(figsize=(8,5))
plot_sigma_vs_average_l2('../../Results/Elastic/strong_scaling/256/vanilla', 0,False)
plt.xticks([0.01,0.05,0.1,0.15,0.2])
plt.xlabel('$\sigma$')
plt.ylabel('Average $L_2$ error')
plt.title('Varing Source size, $\sigma \in [0.01,0.2]$')
plt.tight_layout()
plt.savefig('strong_scaling_full.pdf')
plt.show()
print("hello")