import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
import scienceplots
plt.style.use(['science','ieee'])
plt.rcParams.update({'font.size': 20})


constant_all_df = pd.read_csv('../pre_trained_models/constant_thesis/constant_all_results.csv')
mixture_all_df = pd.read_csv('../pre_trained_models/mixture_thesis/mixture_all_results.csv')
layered_all_df = pd.read_csv('../pre_trained_models/layered_thesis/layered_all_results.csv')

constant_top_df = pd.read_csv('../pre_trained_models/constant_thesis/constant_top_results.csv')
mixture_top_df = pd.read_csv('../pre_trained_models/mixture_thesis/mixture_top_results.csv')
layered_top_df = pd.read_csv('../pre_trained_models/layered_thesis/layered_top_results.csv')

constant_top_007_df = pd.read_csv('../pre_trained_models/constant_thesis/constant_top_results_007.csv')
mixture_top_007_df = pd.read_csv('../pre_trained_models/mixture_thesis/mixture_top_results_007.csv')
layered_top_007_df = pd.read_csv('../pre_trained_models/layered_thesis/layered_top_results_007.csv')

constant_top_01_df = pd.read_csv('../pre_trained_models/constant_thesis/constant_top_results_01.csv')
mixture_top_01_df = pd.read_csv('../pre_trained_models/mixture_thesis/mixture_top_results_01.csv')
layered_top_01_df = pd.read_csv('../pre_trained_models/layered_thesis/layered_top_results_01.csv')

constant_top_02_df = pd.read_csv('../pre_trained_models/constant_thesis/constant_top_results_02.csv')
mixture_top_02_df = pd.read_csv('../pre_trained_models/mixture_thesis/mixture_top_results_02.csv')
layered_top_02_df = pd.read_csv('../pre_trained_models/layered_thesis/layered_top_results_02.csv')


#combined_runs_df = pd.concat([constant_data, layered_sine_data, mixture_data])

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

experiment_y_positions = {
    'Constant': 1.0,  # Adjust these values as needed
    'Layered Sine': 2.0,
    'Mixture': 1.5
}

dataframes = [constant_all_df, mixture_all_df, layered_all_df, constant_top_df, mixture_top_df, layered_top_df, constant_top_007_df, mixture_top_007_df, layered_top_007_df, constant_top_01_df, mixture_top_01_df, layered_top_01_df, constant_top_02_df, mixture_top_02_df, layered_top_02_df]

def map_nn_type_to_spectrum(nn_type):
    if nn_type == 'PINN':
        return 0
    elif nn_type == 'SIREN':
        return 1
    elif nn_type in ['PLANE_WAVE_FCN', 'MORLET_WAVELET_FCN']:
        return 2
    elif nn_type in ['FCN_ALL_PARAMS_WAVELET_FCN', 'FCN_ALL_PARAMS_PLANEWAVE_FCN']:
        return 3
    elif nn_type in ['FCN_ALL_PARAMS_PLANEWAVE', 'FCN_ALL_PARAMS_WAVELET']:
        return 4
    elif 'AMPLITUDE' in nn_type:
        return 5
    else:
        return -1  # Or any other default value for nn_types not listed

for df in dataframes:
    df['spectrum'] = df['nn_type'].apply(map_nn_type_to_spectrum)

data_experiments = {
    'Constant': constant_top_df,
    'Mixture': mixture_top_df,
    'Layered Sine': layered_top_df

}



# Function to plot the lowest test loss for each nn_type across spectrum numbers with a best fit line
def plot_lowest_loss_with_best_fit(data, title, color_mapping):
    plt.figure(figsize=(14, 7))
    nn_types = data['nn_type'].unique()

    # Prepare the plot for the best fit lines
    fig, ax = plt.subplots(figsize=(15, 7))

    for nn_type in nn_types:
        type_data = data[data['nn_type'] == nn_type]
        if not type_data.empty:
            # Calculate the lowest test loss for each spectrum
            spectrum_lowest = type_data.groupby('spectrum')['l2_error'].min()
            print(nn_type)
            print(spectrum_lowest)
            # Plot the lowest test loss with circles
            ax.scatter(spectrum_lowest.index, spectrum_lowest.values, label=nn_type, color=color_mapping[nn_type],s=500)
            # Fit a line through the points if there are at least two points
            #z = np.polyfit(spectrum_lowest.index, spectrum_lowest.values, 5)
            #p = np.poly1d(z)
            #ax.plot(spectrum_lowest.index, p(spectrum_lowest.index), color=color_mapping[nn_type])

    # Adding labels and title
    custom_order_and_labels = [
        ('PINN', 'PINN-tanh'),
        ('SIREN', '$\it{sT}$-PINN'),
        ('PLANE_WAVE_FCN', '$\it{pT}$-PINN'),
        ('MORLET_WAVELET_FCN', '$\it{wT}$-PINN'),
        ('FCN_ALL_PARAMS_PLANEWAVE_FCN', '$\it{pED}$-PINN'),
        ('FCN_ALL_PARAMS_WAVELET_FCN', '$\it{wED}$-PINN'),
        ('FCN_ALL_PARAMS_PLANEWAVE', '$\it{pE}$-PINN'),
        ('FCN_ALL_PARAMS_WAVELET', '$\it{wE}$-PINN'),
        ('FCN_AMPLITUDE_PLANEWAVE', '$\it{pA}$-PINN'),
        ('FCN_AMPLITUDE_WAVELET', '$\it{wA}$-PINN'),
    ]

    # Create custom legend handles based on the defined order and labels
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                 markerfacecolor=color_mapping[nn_type], markersize=12)
                      for nn_type, label in custom_order_and_labels]
    # ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))


    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=5)
    ax.set_xlabel("Spectrum Number / Increasing Restrictiveness")
    ax.set_ylabel("Relative $L_2$ Error in %")
    #ax.set_ylim(0.0,12.0)

    ax.set_yscale('log')
    plt.ylim((0.0, 2.1*pow(10, 1)))
    ax.set_yticks([1*pow(10,0),2*pow(10,0),3*pow(10,0),4*pow(10,0),5*pow(10,0),6*pow(10,0),7*pow(10,0),8*pow(10,0),9*pow(10,0),1*pow(10,1),2*pow(10,1)])
    #ax.set_title(title)
    # Adjust layout to ensure the legend is not cut off

    #ax.legend()
    ax.grid(True)

    plt.tight_layout(pad=2.0)  # Adjust padding as needed
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin

    # Show and return the figure
    #plt.show()
    plt.savefig('best_fit_{}.png'.format(title.split(" ")[0]),dpi=400)
    return fig

# Function to create a bubble plot for cross-experiment data with proper bubbles and varying sizes
# Function to create a bubble plot for cross-experiment data with proper bubbles and varying sizes


def plot_cross_experiment_bubble_proper(data_dict, title, color_mapping):
    plt.figure(figsize=(14, 7))
    fig, ax = plt.subplots(figsize=(15, 7))

    # Create a list to store all bubbles with their sizes, colors, and positions
    bubble_list = []

    for experiment, data in data_dict.items():
        nn_types = data['nn_type'].unique()

        for nn_type in nn_types:
            type_data = data[data['nn_type'] == nn_type]
            aggregated_data = type_data.groupby('spectrum')['l2_error'].mean()
            sizes = (1/(aggregated_data**2/100))*50
            sizes = sizes.fillna(sizes.min() / 2)
            print(sizes)
            print("hii")
            for spectrum, size in zip(aggregated_data.index, sizes):
                y_position = experiment_y_positions[experiment]
                color = color_mapping[nn_type]
                bubble_list.append((size, spectrum, y_position, color, nn_type))
    # Sort the bubble list by size in descending order
    bubble_list.sort(reverse=True, key=lambda x: x[0])
    # Plot the bubbles from largest to smallest
    for size, spectrum, y_position, color, nn_type in bubble_list:
        ax.scatter(spectrum, y_position, s=size, alpha=1.0, color=color, label=nn_type)
    # Adding labels and title
    ax.set_xlabel("Spectrum Number / Increasing Restrictiveness")
    ax.set_ylabel("Experiment Difficulty")
    ax.set_title(title)
    ax.grid(True)

    # Create custom legend
    # Define custom order and labels for the legend


    custom_order_and_labels = [
        ('PINN', 'PINN-tanh'),
        ('SIREN', '$\it{st}$-PINN'),
        ('PLANE_WAVE_FCN','$\it{pt}$-PINN'),
        ('MORLET_WAVELET_FCN','$\it{wt}$-PINN'),
        ('FCN_ALL_PARAMS_PLANEWAVE_FCN','$\it{pED}$-PINN'),
        ('FCN_ALL_PARAMS_WAVELET_FCN','$\it{wED}$-PINN'),
        ('FCN_ALL_PARAMS_PLANEWAVE','$\it{pE}$-PINN'),
        ('FCN_ALL_PARAMS_WAVELET','$\it{wE}$-PINN'),
        ('FCN_AMPLITUDE_PLANEWAVE','$\it{pA}$-PINN'),
        ('FCN_AMPLITUDE_WAVELET','$\it{wA}$-PINN'),
    ]

    # Create custom legend handles based on the defined order and labels
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                 markerfacecolor=color_mapping[nn_type], markersize=10)
                      for nn_type, label in custom_order_and_labels]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=5)
    y_min = min(experiment_y_positions.values()) - 0.275  # Add padding below
    y_max = max(experiment_y_positions.values()) + 0.275  # Add padding above
    # Set y-axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Spectrum Number / Increasing Restrictiveness")
    ax.set_ylabel("Experiment Difficulty")
    ax.set_title(title)
    ax.set_yticks(list(experiment_y_positions.values()))
    ax.set_yticklabels(experiment_y_positions.keys())
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.3)
   # plt.show()
    plt.savefig('bubbles.png',dpi=400)
    return fig





# Generate and save the corrected bubble plot
fig_bubble_proper = plot_cross_experiment_bubble_proper(data_experiments, "Cross Experiment Spectrum Analysis",color_mapping)
print("done")
plt.show()
# Plot the lowest test loss with best fit lines for each experiment
fig_constant = plot_lowest_loss_with_best_fit(constant_top_df, "Constant Parameters", color_mapping)
fig_layered_sine = plot_lowest_loss_with_best_fit(layered_top_df, "Layered Setting",color_mapping)
fig_mixture = plot_lowest_loss_with_best_fit(mixture_top_df, "Mixture Setting", color_mapping)

print(layered_top_df)


# Extracting L2 error for specific nn_types and t1 values
nn_types = ['PINN', 'FCN_ALL_PARAMS_PLANEWAVE_FCN', 'FCN_ALL_PARAMS_WAVELET_FCN']
t1_values = [0.07, 0.1, 0.2]
dataframes = {
    'constant': [constant_top_007_df, constant_top_01_df, constant_top_02_df],
    'mixture': [mixture_top_007_df, mixture_top_01_df, mixture_top_02_df],
    'layered': [layered_top_007_df, layered_top_01_df, layered_top_02_df]
}

# Preparing the data for plotting
plot_data = []
for experiment, dfs in dataframes.items():
    for df, t1 in zip(dfs, t1_values):
        for nn_type in nn_types:
            l2_error = df[df['nn_type'] == nn_type]['l2_error'].values
            if l2_error.size > 0:
                plot_data.append((experiment, nn_type, t1, l2_error[0]))
print("all data",plot_data)
# Plotting
plt.figure(figsize=(5, 5))
for data in plot_data:
    plt.scatter(data[2], data[3], label=f"{data[0]}: {data[1]} (t1={data[2]})", alpha=0.7)

plt.xlabel('t1 Value')
plt.ylabel('L2 Error')
plt.title('L2 Error for Different NN Types and t1 Values')
plt.legend()
plt.show()

# Initialize a dictionary to store the aggregated L2 error data
aggregated_data = {}

# Aggregating the L2 error data
for data in plot_data:
    key = (data[0], data[1])  # Key is a tuple of (experiment, nn_type)
    if key not in aggregated_data:
        aggregated_data[key] = []
    aggregated_data[key].append(data[3])  # Appending L2 error

# Calculating mean and standard deviation
mean_std_data = {}
for key, values in aggregated_data.items():
    mean_std_data[key] = {
        'mean': np.mean(values),
        'std': np.std(values)
    }

print(mean_std_data)

# Define a color map for nn_types
color_map = {
    'PINN': 'purple',
    'FCN_ALL_PARAMS_PLANEWAVE_FCN': 'pink',
    'FCN_ALL_PARAMS_WAVELET_FCN': 'orange'
}

# Organizing the data for error bar plotting
experiments = ['constant', 'mixture', 'layered']
nn_types = ['PINN', 'FCN_ALL_PARAMS_PLANEWAVE_FCN', 'FCN_ALL_PARAMS_WAVELET_FCN']

# Plotting Error Bars
fig, ax = plt.subplots(figsize=(8, 8))


print("mean_std_data",mean_std_data)

for i, exp in enumerate(experiments):
    for j, nn_type in enumerate(nn_types):
        key = (exp, nn_type)
        if key in mean_std_data:
            mean = mean_std_data[key]['mean']
            std = mean_std_data[key]['std']
            x = i + j * 0.1  # Adjust x to separate the groups
            color = color_map[nn_type]

            ax.errorbar(x, mean, std, linestyle='None', marker='^', color=color)

# Improving the plot
custom_order_and_labels = [
        ('PINN', 'PINN-tanh'),
        ('FCN_ALL_PARAMS_PLANEWAVE_FCN','$\it{pED}$-PINN'),
        ('FCN_ALL_PARAMS_WAVELET_FCN','$\it{wED}$-PINN'),
    ]

ax.set_xticks([i + 0.1 for i in range(len(experiments))], experiments)
ax.set_ylabel('Mean $L_2$ Error')
ax.set_xlabel('Experiment Type')
ax.set_title('$t_1$ Sensitivity Across Experiments',fontsize=20)
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,markerfacecolor=color_mapping[nn_type], markersize=10)for nn_type, label in custom_order_and_labels]
ax.legend(handles=legend_handles, loc='lower center',ncol=3, bbox_to_anchor=(0.5, -0.25))
plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.3)
plt.savefig('t1-sensitivity.pdf',dpi=400)
plt.show()


