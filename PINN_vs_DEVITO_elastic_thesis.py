import matplotlib.patches as patches
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import torch.nn as nn
import os
import configparser
import numpy as np
import matplotlib.pyplot as plt
import torch
import PINNs
import time as timer
import pickle
import pandas as pd
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()

def relative_l2_error(true, pred):
    # Ensure true and pred are numpy arrays for correct operations
    true = np.array(true)
    pred = np.array(pred)

    # Calculate the L2 norm of the error
    error_norm = np.linalg.norm(pred - true)

    # Calculate the L2 norm of the true (observed) values
    true_norm = np.linalg.norm(true)

    # Compute the Relative L2 Error
    relative_error = error_norm / true_norm

    return relative_error


constant_folder = '../pre_trained_models/constant_thesis/'
csubfolders = [ f.path for f in os.scandir(constant_folder) if f.is_dir() ]

mixture_folder = '../pre_trained_models/mixture_thesis/'
msubfolders = [ f.path for f in os.scandir(mixture_folder) if f.is_dir() ]

layered_folder = '../pre_trained_models/layered_thesis/'
lsubfolders = [ f.path for f in os.scandir(layered_folder) if f.is_dir() ]

subfolders = csubfolders + msubfolders + lsubfolders

hyper_folder = '../pre_trained_models/hyper/'
hsubfolders = [ f.path for f in os.scandir(hyper_folder) if f.is_dir() ]

t1_folders = '../pre_trained_models/t1_experiment/'
t1subfolders = [ f.path for f in os.scandir(t1_folders) if f.is_dir() ]


models = ['../pre_trained_models/constant_thesis/PINN_constant_thesis_100000_200_1.0_100_100_100_5_128_tanh_0.1_PINN_1_constant_0.1','../pre_trained_models/mixture_thesis/PINN_mixture_thesis_100000_200_1.0_100_100_100_5_128_tanh_0.1_PINN_1_mixture_0.1','../pre_trained_models/layered_thesis/PINN_layered_thesis_100000_400_2.0_200_200_100_5_128_tanh_0.2_PINN_1_layered_sine_simpler_3_0.1']

new_source_1_wavlete_model = 'PINN_new_source_100000_200_1.0_100_100_100_4_32_tanh_0.1_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_constant_0.1'
new_source_1_vanilla_model = 'PINN_new_source_100000_200_1.0_100_100_100_5_128_tanh_0.1_PINN_1_constant_0.1'

models = [new_source_1_wavlete_model,new_source_1_vanilla_model]
for input_folder in models:
#input_folder = '../pre_trained_models/constant_thesis/PINN_constant_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.07_FCN_ALL_PARAMS_WAVELET_FCN_128_4_8_constant_0.1'



    dt=1.095e-3
    # Define the range for random numbers
    min_val, max_val = -0.45, 0.45

    mu_quake_x_list = [0.0]
    mu_quake_y_list = [0.0]


    FD_prep_1 = timer.time()
    config.read("../pre_trained_models/"+input_folder+"/config.ini")

    model_type = config['parameters']['model_type']



    mu_quake = [0.0, 0.0]
    mu_quake = torch.tensor(mu_quake)

    input_str = str(input_folder)
    if 'layered' in input_str:
        str_path = input_str.replace("../pre_trained_models/layered_thesis/", "")
    elif 'mixture' in input_str:
        str_path = input_str.replace("../pre_trained_models/mixture_thesis/", "")
    elif 'constant' in input_str:
        str_path = input_str.replace("../pre_trained_models/hyper/", "")
    else:
        print("unknong input str",input_str)
        raise Exception




    save_path = '../../Results/Elastic/hyper/[{}_{}]/{}/'.format( str(mu_quake[0]), str(mu_quake[1]),str_path)
    os.makedirs(save_path, exist_ok=True)

    config.read(input_folder + "/config.ini")
    t1 = float(config['initial_condition']['t1'])

    devito_prep_time = 0.0

    # Parameters
    rho_solid = float(config['parameters']['rho_solid'])

    model_path =  "../pre_trained_models/"+input_folder + "/model.pth"
    print("model path = ", model_path)

    model_type = config["parameters"]["model_type"]


    print("Model type not specified, reading from config file instead")
    network_type = config['Network']['model_type']
    if network_type == "Pinns":
        pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
    elif network_type == "Global_NSources_Conditioned_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
    elif network_type == "Relative_Distance_NSources_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                  config=config)
    elif network_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                    config=config)
    elif network_type == "Global_NSources_Conditioned_Lame_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                            config=config)
    elif network_type == "Global_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                         config=config)
    else:
        raise Exception("Model type {} not supported".format(network_type))

    tag = str_path.replace(".pth", "")
    tag = tag.replace("/model", "")

    n_neurons = int(config['Network']['n_neurons'])
    n_hidden_layers = int(config['Network']['n_hidden_layers'])
    print(n_neurons, n_hidden_layers)
    if config['Network']['activation'] == 'tanh':
        activation = nn.Tanh()
    elif config['Network']['activation'] == 'sin':
        activation = PINNs.SinActivation()
    elif config['Network']['activation'] == 'wavelet':
        activation = PINNs.WaveletActivation(int(config['Network']['n_neurons']))
    elif config['Network']['activation'] == 'wavelet2':
        activation = PINNs.WaveletActivation_2(int(config['Network']['n_neurons']))
    else:
        print("unknown activation function", config['Network']['activation'])
        raise Exception

    if config['Network']['nn_type'] == 'SIREN':
        # self, input_dimension, output_dimension, n_hidden_layers, neurons,
        # regularization_param, regularization_exp, retrain_seed, activation, n_1
        my_network = PINNs.SIREN_NeuralNet(input_dimension=3, output_dimension=2,
                                                    n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                    neurons=int(config['Network']['n_neurons']),
                                                    regularization_param=0.,
                                                    regularization_exp=2.,
                                                    retrain_seed=3, activation=activation,
                                                    n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'PLANE_WAVE_FCN':
        my_network = PINNs.Planewave_and_FCN_NeuralNet(input_dimension=3, output_dimension=2,
                                                                n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                                neurons=int(config['Network']['n_neurons']),
                                                                regularization_param=0.,
                                                                regularization_exp=2.,
                                                                retrain_seed=3, activation=activation,
                                                                n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'MORLET_WAVELET_FCN':
        my_network = PINNs.MorletWavelet_and_FCN(input_dimension=3, output_dimension=2,
                                                          n_hidden_layers=int(
                                                              config['Network']['n_hidden_layers']),
                                                          neurons=int(config['Network']['n_neurons']),
                                                          regularization_param=0.,
                                                          regularization_exp=2.,
                                                          retrain_seed=3, activation=activation,
                                                          n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_PLANEWAVE':
        my_network = PINNs.FCN_Amplitude_Planewave(input_dimension=3, output_dimension=2,
                                                            n_hidden_layers=int(
                                                                config['Network']['n_hidden_layers']),
                                                            neurons=int(config['Network']['n_neurons']),
                                                            regularization_param=0.,
                                                            regularization_exp=2.,
                                                            retrain_seed=3, activation=activation,
                                                            n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_WAVELET':
        my_network = PINNs.FCN_Amplitude_Wavelet(input_dimension=3, output_dimension=2,
                                                          n_hidden_layers=int(
                                                              config['Network']['n_hidden_layers']),
                                                          neurons=int(config['Network']['n_neurons']),
                                                          regularization_param=0.,
                                                          regularization_exp=2.,
                                                          retrain_seed=3, activation=activation,
                                                          n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET':
        my_network = PINNs.FCN_all_params_Wavelet_Modulation(input_dimension=3, output_dimension=2,
                                                                      n_hidden_layers=int(
                                                                          config['Network']['n_hidden_layers']),
                                                                      neurons=int(config['Network']['n_neurons']),
                                                                      regularization_param=0.,
                                                                      regularization_exp=2.,
                                                                      retrain_seed=3, activation=activation,
                                                                      n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE':
        my_network = PINNs.FCN_all_params_Planewave_Modulation(input_dimension=3,
                                                                        output_dimension=2,
                                                                        n_hidden_layers=int(
                                                                            config['Network']['n_hidden_layers']),
                                                                        neurons=int(config['Network']['n_neurons']),
                                                                        regularization_param=0.,
                                                                        regularization_exp=2.,
                                                                        retrain_seed=3, activation=activation,
                                                                        n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
        print("correct")
        my_network = PINNs.FCN_all_params_Wavelet_FCN(input_dimension=3, output_dimension=2,
                                                      n_hidden_layers=int(
                                                          config['Network']['n_hidden_layers']),
                                                      neurons=int(config['Network']['n_neurons']),
                                                      regularization_param=0.,
                                                      regularization_exp=2.,
                                                      retrain_seed=3, activation=activation,
                                                      n_1=int(config['Network']['n_1']),
                                                      n_hidden_layers_after=int(
                                                          config['Network']['n_hidden_layers_after']),
                                                      n_neurons_after=int(
                                                          config['Network']['n_neurons_after']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE_FCN':
        print("it's a plane")
        my_network = PINNs.FCN_all_params_Planewave_FCN(input_dimension=3, output_dimension=2,
                                                        n_hidden_layers=int(
                                                            config['Network']['n_hidden_layers']),
                                                        neurons=int(config['Network']['n_neurons']),
                                                        regularization_param=0.,
                                                        regularization_exp=2.,
                                                        retrain_seed=3, activation=activation,
                                                        n_1=int(config['Network']['n_1']),
                                                        n_hidden_layers_after=int(
                                                            config['Network']['n_hidden_layers_after']),
                                                        n_neurons_after=int(
                                                            config['Network']['n_neurons_after']))
    else:
        my_network = PINNs.NeuralNet(input_dimension=3, output_dimension=2,
                                     n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                     neurons=int(config['Network']['n_neurons']),
                                     regularization_param=0.,
                                     regularization_exp=2.,
                                     retrain_seed=3, activation=activation)

    my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    pinn.approximate_solution = my_network

    if model_type == 'constant':
        numpoints_sqrt = 256
    elif model_type == 'mixture':
        numpoints_sqrt = 256
    elif (model_type == 'layered') or (model_type == 'layered_sine') or (model_type == 'layered_sine_simpler') or (
            model_type == 'layered_sine_simpler_3'):
        numpoints_sqrt = 256
    else:
        print("unsuported model type", model_type)
        raise Exception

    inferece_time = 0.0
    FD_time = 0.0

    soboleng = torch.quasirandom.SobolEngine(dimension=pinn.domain_extrema.shape[0])
    inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))
    inputs_repeated = inputs.repeat(100, 1)

    grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                    torch.linspace(-1.0, 1.0, numpoints_sqrt))
    grid_x = torch.reshape(grid_x, (-1,))
    grid_y = torch.reshape(grid_y, (-1,))

    inputs[:, 1] = grid_x
    inputs[:, 2] = grid_y

    t0 = timer.time()
    ux = pinn.pinn_model_eval(inputs_repeated)[:, 0]
    uy = pinn.pinn_model_eval(inputs_repeated)[:, 1]
    t1 = timer.time()
    print("TIME =",t1-t0)


    # inputs[:, 3] = mu_quake[0]
    # inputs[:, 4] = mu_quake[1]

    res_list_ux = []
    res_list_uy = []
    res_list_u = []
    n = 102
    dt = 1.096e-3

    time_list = np.linspace(0, 1, n).tolist()
    total_t = 0.0
    for i in time_list:
        time = i
        inputs[:, 0] = time

        NN_time_0 = timer.time()
        t0 = timer.time()
        ux = pinn.pinn_model_eval(inputs)[:, 0]
        uy = pinn.pinn_model_eval(inputs)[:, 1]
        t1 = timer.time()
        print(t1-t0)
        total_t = total_t+(t1-t0)
        ux_out = ux.detach()
        uy_out = uy.detach()

        np_ux_out = ux_out.numpy()
        np_uy_out = uy_out.numpy()

        B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
        B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
        B = np.sqrt(B_uy ** 2 + B_ux ** 2)
        res_list_ux.append(B_ux)
        res_list_uy.append(B_uy)
        res_list_u.append(B)
        NN_time_1 = timer.time()
        inferece_time = inferece_time + (NN_time_1 - NN_time_0)
    print("Total t=",total_t)
    res_ux = np.dstack(res_list_ux)
    res_uy = np.dstack(res_list_uy)
    res_ux = np.rollaxis(res_ux, -1)
    res_uy = np.rollaxis(res_uy, -1)
    res_u = np.dstack(res_list_u)
    res_u = np.rollaxis(res_u, -1)

    RMSE = 0
    RRMSE = 0
    index = 1
    total_relative_l2_error = 0.0
    total_RRMSE = 0.0
    relative_l2_err = 0.0

    error_file = save_path + 'running_errors.csv'

    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=['time', 'relative_l2', 'RRMSE'])

    # Define the specific time steps to plot
    time_steps_to_plot = [
        0,
        int(len(res_list_uy) / 4),
        int(len(res_list_uy) / 2),
        int(3 * int(len(res_list_uy) / 4)),
        len(res_list_uy) - 2
    ]
    time_values_to_plot = [time_list[h] for h in time_steps_to_plot]

    # Create figure and GridSpec with appropriate layout
    n_columns = len(time_steps_to_plot)
    f = plt.figure(figsize=(10, 20))  # Adjust figure size as needed
    f.subplots_adjust(left=.1, right=.9)
    main_gs = gridspec.GridSpec(3, n_columns, hspace=0.1, wspace=0)  # Adjust hspace for spacing between groups

    # Set row titles for the leftmost column
    titles = ["PINN $u_x$", "FD $u_x$", r"$\triangle u_x$", r"PINN $u_y$", r"FD $u_y$", r"$\triangle u_y$",
              r"PINN $|u|$", r"FD $|u|$", r"$\triangle |u|$"]

    if model_type == 'constant':
        with open("constant_devito_01_256_x", "rb") as fp:
            res_list_devito_x = pickle.load(fp)
        with open("constant_devito_01_256_y", "rb") as fp:
            res_list_devito_y = pickle.load(fp)
        with open("constant_devito_01_256_u", "rb") as fp:
            res_list_devito_u = pickle.load(fp)
    elif model_type == 'mixture':
        with open("mixture_devito_01_512_x", "rb") as fp:
            res_list_devito_x = pickle.load(fp)
        with open("mixture_devito_01_512_y", "rb") as fp:
            res_list_devito_y = pickle.load(fp)
        with open("mixture_devito_01_512_u", "rb") as fp:
            res_list_devito_u = pickle.load(fp)
    elif (model_type == 'layered') or (model_type == 'layered_sine') or (model_type == 'layered_sine_simpler') or (
            model_type == 'layered_sine_simpler_3'):
        with open("layered_devito_01_512_x", "rb") as fp:
            res_list_devito_x = pickle.load(fp)
        with open("layered_devito_01_512_y", "rb") as fp:
            res_list_devito_y = pickle.load(fp)
        with open("layered_devito_01_512_u", "rb") as fp:
            res_list_devito_u = pickle.load(fp)
    else:
        print("unsuported model type", model_type)
        raise Exception

    for h in range(0, len(res_list_uy)):
        diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
        diffx[0:9, :] = 0
        diffx[:, 0:9] = 0
        diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
        diffy[0:9, :] = 0
        diffy[:, 0:9] = 0
        diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
        diffu[0:9, :] = 0
        diffu[:, 0:9] = 0
        diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        RMSE = np.sqrt(np.mean(diffu ** 2))
        RRMSE = np.sqrt(np.mean(diffu ** 2)) / np.sqrt(np.mean(res_list_devito_u[h] ** 2))
        # print("RMSE:", RMSE, "RRMSE:", RRMSE * 100.0, "%")
        relative_l2_err = relative_l2_error(res_list_devito_u[h], res_u[h, :, :])
        # print("Relative L2 error: ", relative_l2_err * 100.0, "%")
        total_relative_l2_error = total_relative_l2_error + relative_l2_err
        total_RRMSE = total_RRMSE + RRMSE
        index = index + 1
        new_row = {'time': f'time_{h}', 'relative_l2': relative_l2_err, 'RRMSE': RRMSE}
        df = df.append(new_row, ignore_index=True)

    s = 12 * np.mean(np.abs(res_list_devito_x[0]))

    # Plot the data for each selected time step
    for idx, (h, time_value) in enumerate(zip(time_steps_to_plot, time_values_to_plot)):
        # Calculate differences for time step 'h'
        diffx = res_uy[h, :, :] - res_list_devito_x[h]
        diffy = res_ux[h, :, :] - res_list_devito_y[h]
        diffu = res_u[h, :, :] - res_list_devito_u[h]

        relative_l2_err = relative_l2_error(res_list_devito_u[h], res_u[h, :, :])

        # Set the time title for the top of each column
        title_gs = main_gs[0, idx].get_gridspec()
        ax = f.add_subplot(title_gs.new_subplotspec((0, idx)))
        ax.set_title(f"Time: {time_value:.2f}s", fontsize=15, pad=20)
        ax.axis('off')  # Hide this axis

        # Apply zero-padding to differences
        for diff in [diffx, diffy, diffu]:
            diff[0:9, :] = 0
            diff[:, 0:9] = 0
            diff[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diff[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        for group_idx in range(3):
            # Create a subgrid for each group in the current column (time step)
            sub_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[group_idx, idx], hspace=0.0,
                                                      wspace=0.0)

            for row_idx in range(3):
                ax = f.add_subplot(sub_gs[row_idx, 0])

                # Calculate the overall row index
                i = group_idx * 3 + row_idx

                # Select data to plot
                if i < 3:  # First group (PINN)
                    data = res_uy[h, :, :] if i == 0 else res_list_devito_x[h] if i == 1 else diffx
                elif i < 6:  # Second group (FD)
                    data = res_ux[h, :, :] if i == 3 else res_list_devito_y[h] if i == 4 else diffy
                else:  # Third group (Difference)
                    data = res_u[h, :, :] if i == 6 else res_list_devito_u[h] if i == 7 else diffu

                # Plot data
                ax.imshow(data, cmap='bwr', vmin=-s, vmax=s)

                # Remove axis ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Set the row title for the first column
                if idx == 0:
                    ax.text(-0.15, 0.5, titles[i], transform=ax.transAxes, fontsize=15, va='center', ha='right',
                            rotation=0)

                # In the bottom row of each column, add the rectangle with L2 error text
                if group_idx == 2 and row_idx == 2:
                    # Determine the size of the rectangle
                    rect_width, rect_height = 0.25, 0.25  # Adjust these values as needed

                    # Calculate the position of the bottom-right corner
                    rect_x = 1 - rect_width
                    rect_y = 0

                    # Create and add the rectangle
                    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                             transform=ax.transAxes, color='black', clip_on=False)
                    ax.add_patch(rect)

                    # Add L2 error text inside the rectangle
                    error_text = f"{round(float(relative_l2_err * 100), 1)}\%"  # Format the L2 error
                    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, error_text,
                            transform=ax.transAxes, fontsize=12, color='white', va='center', ha='center')

    # Create an axes for the colorbar
    cbar_ax = f.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]

    # Create a ScalarMappable with the same colormap and limits as your plots
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-s, vmax=s))

    # Create the colorbar
    f.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    f.tight_layout()
    #f.show()
    f.savefig(save_path + "selected_time_steps_comparison.png")
    #f.show()

    df.to_csv(error_file, index=False)
    RMSE = RMSE / len(res_list_uy)
    RRMSE = RRMSE / len(res_list_uy)
    print(len(res_list_uy))
    print("Average relative l2 error:", (total_relative_l2_error / len(res_list_uy)) * 100.0, "%", " dt = ", dt, "n = ",
          n)
    print("Average RRMSE:", (total_RRMSE / len(res_list_uy)) * 100.0, "%", "dt = ", dt)

    average_file = save_path + 'averages.csv'
    # Create an empty DataFrame with specified columns
    df_a = pd.DataFrame(columns=['average_l2', 'average_RRMSE'])
    new_row_a = {'average_l2': (total_relative_l2_error / len(res_list_uy)) * 100.0,
                 'average_RRMSE': (total_RRMSE / len(res_list_uy)) * 100.0}
    df_a = df_a.append(new_row_a, ignore_index=True)
    df_a.to_csv(average_file, index=False)

