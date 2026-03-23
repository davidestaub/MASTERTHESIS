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

constant_folder = '../pre_trained_models/constant_thesis/PINN_constant_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.1_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_constant_0.1'
mixture_folder = '../pre_trained_models/mixture_thesis/PINN_mixture_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.1_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_mixture_0.1'
layered_folder = '../pre_trained_models/layered_thesis/PINN_layered_thesis_100000_400_2.0_200_200_100_4_32_tanh_0.07_FCN_ALL_PARAMS_PLANEWAVE_FCN_128_3_8_layered_sine_simpler_3_0.1'


def mother_wavelet(t, a=4):
    result = (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))
    if torch.isnan(result).any():
        print("found nan in result")
        if torch.isnan(torch.exp(-(a ** 2) * (t ** 2))).any():
            print("nan in second term")
            print(t)
            if torch.isnan(t).any():
                print("nan in t ????")
        elif torch.isnan((1 - 2 * (a ** 2) * (t ** 2))).any():
            print("nan in first term")
            if torch.isnan(t).any():
                print("nan in t ????")
    return (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))

def compute_wavelet_transform(x, scaling, translation, w_x, w_y):
    t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]
    spatial_term = x_val.unsqueeze(-1) * w_x + y_val.unsqueeze(-1) * w_y
    adjusted_time_translation = t.unsqueeze(-1) - translation - spatial_term
    modulated_wavelet = mother_wavelet(scaling * adjusted_time_translation)
    return modulated_wavelet




def visualize_wavelet_output(x,wavelet_params, n_1):
    # Extract parameters
    scaling, translation, w_x, w_y = wavelet_params['params'].chunk(4, dim=1)

    for i in range(n_1):
        # Dummy example for wavelet transform computation
        # You will need to replace this with your actual computation using scaling[i], translation[i], w_x[i], w_y[i]
        wavelet_output = compute_wavelet_transform(x, scaling, translation, w_x, w_y)

        # Visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(wavelet_output,
                   extent=[-1,1-1,1],
                   aspect='auto')
        plt.colorbar()
        plt.title(f'Wavelet Neuron {i} Output')
        plt.show()


new_folder = [constant_folder,mixture_folder,layered_folder]

with torch.no_grad():

    for input_folder in new_folder:
    #input_folder = '../pre_trained_models/constant_thesis/PINN_constant_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.07_FCN_ALL_PARAMS_WAVELET_FCN_128_4_8_constant_0.1'




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

        model_path =  input_folder + "/model.pth"
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

        # Attach the hook to the params_layer
        my_network.params_layer.register_forward_hook(capture_params_hook)

        soboleng = torch.quasirandom.SobolEngine(dimension=pinn.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        # inputs[:, 3] = mu_quake[0]
        # inputs[:, 4] = mu_quake[1]

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        n = 102

        time_list = np.linspace(0, 1, n).tolist()
        for i in time_list:
            time = i
            inputs[:, 0] = time

            visualize_wavelet_output(inputs,wavelet_params,128)

            NN_time_0 = timer.time()
            ux = pinn.pinn_model_eval(inputs)[:, 0]
            uy = pinn.pinn_model_eval(inputs)[:, 1]
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

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)

        # Define the specific time steps to plot
        time_steps_to_plot = [
            0,
            int(len(res_list_uy) / 4),
            int(len(res_list_uy) / 2),
            int(3 * int(len(res_list_uy) / 4)),
            len(res_list_uy) - 2
        ]
        time_values_to_plot = [time_list[h] for h in time_steps_to_plot]

        for h in time_steps_to_plot:
            s = 0.1
            data = res_ux[h, :, :]
            plt.imshow(data, cmap='bwr', vmin=-s, vmax=s)
            plt.show()








