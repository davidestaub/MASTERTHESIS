import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
torch.manual_seed(2)
import torch.nn as nn
import wandb
import mixture_model
import torch
from torch.utils.data import DataLoader
torch.manual_seed(128)
import os
import initial_conditions
import numpy as np
import torch
import FD_devito
from devito import *
import pickle
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)



torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class WaveletActivation(nn.Module):
    def __init__(self, in_features):
        super(WaveletActivation, self).__init__()
        self.amplitude = nn.Parameter(torch.Tensor(in_features))
        self.frequency = nn.Parameter(torch.Tensor(in_features))
        self.phase = nn.Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.amplitude, -1, 1)
        nn.init.uniform_(self.frequency, 0, 1)
        nn.init.uniform_(self.phase, 0, 2*torch.pi)

    def forward(self, input):
        # Ricker wavelet as the mother wavelet
        a = 4
        wavelet_output = (1 - 2 * a ** 2 * (input - self.phase) ** 2) * torch.exp(-a ** 2 * (input - self.phase) ** 2)
        return self.amplitude * wavelet_output #* torch.cos(2 * torch.pi * self.frequency * input)

class WaveletActivation_2(nn.Module):
    def __init__(self, input_features):
        super(WaveletActivation_2, self).__init__()

        # Assuming input_features is the size of the input vector to this layer
        self.input_features = input_features

        # Learnable parameters for each input feature
        self.amplitude = nn.Parameter(torch.randn(input_features))
        self.scaling = nn.Parameter(torch.randn(input_features))
        self.translation = nn.Parameter(torch.randn(input_features))

    def mother_wavelet(self, t):
        a = 4  # Constant for Ricker wavelet
        return (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))

    def forward(self, x):
        # Apply wavelet transformation element-wise
        # Assuming x has shape [batch_size, input_features]
        adjusted_input = self.scaling.unsqueeze(0) * x + self.translation.unsqueeze(0)
        modulated_wavelet = self.amplitude * self.mother_wavelet(adjusted_input)

        return modulated_wavelet


class SinusoidalMappingLayer(nn.Module):
    def __init__(self, d, n_1):
        super(SinusoidalMappingLayer, self).__init__()
        self.w1 = nn.Parameter(torch.randn(d, n_1))
        self.b1 = nn.Parameter(torch.zeros(n_1))

    def forward(self, x):
        return torch.sin(2 * torch.pi * (x @ self.w1 + self.b1))

class SinusoidalMappingLayer_params_from_FCN(nn.Module):
    def __init__(self, n_1):
        super(SinusoidalMappingLayer_params_from_FCN, self).__init__()
        self.n_1 = n_1

    def forward(self, x, w1, b1):
        return torch.sin(2 * torch.pi * (x @ w1 + b1))

class AdvancedBesselLayer(nn.Module):
    def __init__(self, n_1):
        super(AdvancedBesselLayer, self).__init__()

        self.neurons = n_1
        self.A = nn.Parameter(torch.randn(self.neurons))
        self.k = nn.Parameter(torch.randn(self.neurons))
        self.omega = nn.Parameter(torch.randn(self.neurons))
        self.alpha = nn.Parameter(torch.randn(self.neurons))
        self.beta = 0.0  # This is scalar
        self.gamma = nn.Parameter(torch.randn(self.neurons))
        self.kappa = nn.Parameter(torch.randn(self.neurons))
        self.alpha_gaussian = nn.Parameter(torch.randn(self.neurons))
        self.eps = 1e-5

        self.lambda_0 = 0.08152
        self.p1 = -0.2622096091
        self.q = 0.5701109315
        self.p_tilda1 = -0.3945665468
        self.p0 = 0.5948274174
        self.p2 = -0.04932081835
        self.p_tilda0 = 0.4544933991
        self.p_tilda2 = 0.03277620113

    def j0_approximation(self, x):
        factor1 = 1.0 / (((1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.25) * (1.0 + self.q * (x ** 2)))
        cosine_term = (self.p0 + self.p1 * (x ** 2) + self.p2 * (
                    1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) * torch.cos(x)
        sine_term = (self.p_tilda0 + self.p_tilda1 * (x ** 2)) * (
                    (1 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) + self.p_tilda2 * (
                            x ** 2) * torch.sin(x) / x
        result = factor1 * (cosine_term + sine_term)
        return result

    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Calculate R
        R = torch.sqrt(x_val ** 2 + y_val ** 2 + self.eps).unsqueeze(
            -1)  # Broadcasting for operations with neuron weights

        # Gaussian envelope
        term_inside_sqrt = torch.sqrt(self.gamma ** 2 + self.eps) * (x_val.unsqueeze(-1) ** 2) + torch.sqrt(self.kappa.unsqueeze(0) ** 2 + self.eps) * (y_val.unsqueeze(-1) ** 2) + self.eps

        gaussian_envelope = torch.exp(
            -torch.sqrt(self.alpha_gaussian ** 2 + self.eps) *
            (self.k * torch.sqrt(term_inside_sqrt) - self.omega.unsqueeze(0) * t.unsqueeze(-1)) ** 2
        )

        # Bessel value
        atan_term = torch.atan2(y_val.unsqueeze(-1) + self.eps, x_val.unsqueeze(-1) + self.eps)
        bessel_value_x = gaussian_envelope * torch.cos(
            atan_term - self.alpha + (self.beta * atan_term ** 2)
        ) * self.A * torch.exp(-t.unsqueeze(-1)) * self.j0_approximation(
            self.k * R - self.omega * t.unsqueeze(-1) + self.eps)

        return bessel_value_x

class PlaneWaveLayer(nn.Module):
    def __init__(self, n_1):
        super(PlaneWaveLayer, self).__init__()

        self.neurons = n_1
        self.k = nn.Parameter(torch.randn(self.neurons))
        self.l = nn.Parameter(torch.randn(self.neurons))
        self.v = nn.Parameter(torch.randn(self.neurons))
        self.A = nn.Parameter(torch.randn(self.neurons))
        self.phi = nn.Parameter(torch.randn(self.neurons))

    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        distance_term = torch.sqrt(x_val ** 2 + y_val ** 2 + 1e-8).unsqueeze(
            -1)  # Broadcasting for operations with neuron weights

        theta = self.k.unsqueeze(0) * x_val.unsqueeze(-1) + self.l.unsqueeze(0) * y_val.unsqueeze(
            -1) - self.v.unsqueeze(0) * t.unsqueeze(-1) + self.phi.unsqueeze(0) * distance_term
        complex_exp = self.A.unsqueeze(0) * torch.exp(1j * theta)

        return torch.real(complex_exp)

class PlaneWaveLayer_no_amplitude(nn.Module):
    def __init__(self, n_1):
        super(PlaneWaveLayer_no_amplitude, self).__init__()

        self.neurons = n_1
        self.k = nn.Parameter(torch.randn(self.neurons))
        self.l = nn.Parameter(torch.randn(self.neurons))
        self.v = nn.Parameter(torch.randn(self.neurons))
        self.phi = nn.Parameter(torch.randn(self.neurons))

    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        distance_term = torch.sqrt(x_val ** 2 + y_val ** 2 + 1e-8).unsqueeze(
            -1)  # Broadcasting for operations with neuron weights

        theta = self.k.unsqueeze(0) * x_val.unsqueeze(-1) + self.l.unsqueeze(0) * y_val.unsqueeze(
            -1) - self.v.unsqueeze(0) * t.unsqueeze(-1) + self.phi.unsqueeze(0) * distance_term
        complex_exp = torch.exp(1j * theta)

        return torch.real(complex_exp)

class WaveletLayer(nn.Module):
    def __init__(self, n_1):
        super(WaveletLayer, self).__init__()

        # Neuron count
        self.neurons = n_1

        # Learnable parameters for each neuron
        self.frequency = nn.Parameter(torch.randn(self.neurons))  # Frequency modulation
        self.wavelength = nn.Parameter(torch.randn(self.neurons))  # Wavelength modulation
        self.scaling = nn.Parameter(torch.randn(self.neurons))  # Scaling for the wavelet
        self.translation = nn.Parameter(torch.randn(self.neurons))  # Translating the wavelet in time
        self.amplitude = nn.Parameter(torch.randn(self.neurons))  # Amplitude scaling

        # Weights for spatial adjustment
        self.w_x = nn.Parameter(torch.randn(self.neurons))
        self.w_y = nn.Parameter(torch.randn(self.neurons))

    def mother_wavelet(self, t):
        # Ricker wavelet with a=4
        a = 4
        return (1 - 2 * a ** 2 * t ** 2) * torch.exp(-a ** 2 * t ** 2)


    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Spatial term (you can modify this based on your requirements)
        spatial_term = x_val.unsqueeze(-1) * self.w_x + y_val.unsqueeze(-1) * self.w_y

        # Calculate adjusted time translation based on spatial and other learned parameters
        adjusted_time_translation = t.unsqueeze(-1) + self.translation.unsqueeze(0) + spatial_term

        # Apply the learned transformations to the mother wavelet
        modulated_wavelet = self.amplitude * self.mother_wavelet(self.scaling.unsqueeze(0) * adjusted_time_translation)

        return modulated_wavelet

class WaveletLayer_no_amplitude(nn.Module):
    def __init__(self, n_1):
        super(WaveletLayer_no_amplitude, self).__init__()

        # Neuron count
        self.neurons = n_1

        # Learnable parameters for each neuron
        self.scaling = nn.Parameter(torch.randn(self.neurons))  # Scaling for the wavelet
        self.translation = nn.Parameter(torch.randn(self.neurons))  # Translating the wavelet in time

        # Weights for spatial adjustment
        self.w_x = nn.Parameter(torch.randn(self.neurons))
        self.w_y = nn.Parameter(torch.randn(self.neurons))

    def mother_wavelet(self, t):
        # Ricker wavelet with a=4
        a = 4
        return (1 - 2 * a ** 2 * t ** 2) * torch.exp(-a ** 2 * t ** 2)

    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Spatial term
        spatial_term = x_val.unsqueeze(-1) * self.w_x + y_val.unsqueeze(-1) * self.w_y

        # Calculate adjusted time translation based on spatial and other learned parameters
        adjusted_time_translation = t.unsqueeze(-1) + self.translation.unsqueeze(0) + spatial_term

        # Apply the learned transformations to the mother wavelet
        modulated_wavelet = self.mother_wavelet(self.scaling.unsqueeze(0) * adjusted_time_translation)

        return modulated_wavelet

class WaveletLayer_params_from_FCN(nn.Module):
    def __init__(self, n_1):
        super(WaveletLayer_params_from_FCN, self).__init__()
        self.neurons = n_1

    def mother_wavelet(self, t, a=4):
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

    def forward(self, x, scaling, translation, w_x, w_y):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]
        spatial_term = x_val.unsqueeze(-1) * w_x + y_val.unsqueeze(-1) * w_y
        adjusted_time_translation = t.unsqueeze(-1) - translation - spatial_term
        modulated_wavelet = self.mother_wavelet(scaling * adjusted_time_translation)
        return modulated_wavelet

class WaveletLayer_gaussian_params_from_FCN(nn.Module):
    def __init__(self, n_1):
        super(WaveletLayer_gaussian_params_from_FCN, self).__init__()
        self.neurons = n_1

    def mother_wavelet(self, t, a=4):
        return (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))

    def gaussian_envelope(self, x, mean, variance):
        #print(torch.isnan(torch.exp(-0.5 * ((x - mean) ** 2) / (torch.abs(variance)+1e-5))).any())
        return torch.exp(-0.5 * ((x - mean) ** 2) / (torch.abs(variance)+1e-5))

    def forward(self, x, scaling, translation, w_x, w_y, gaussian_mean, gaussian_variance):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]
        spatial_term = x_val.unsqueeze(-1) * w_x + y_val.unsqueeze(-1) * w_y
        adjusted_time_translation = t.unsqueeze(-1) - translation - spatial_term
        wavelet = self.mother_wavelet(scaling * adjusted_time_translation)

        # Apply Gaussian envelope
        gaussian = self.gaussian_envelope(x_val.unsqueeze(-1), gaussian_mean, gaussian_variance)
        modulated_wavelet = wavelet * gaussian

        return modulated_wavelet

class WaveletLayer_params_from_FCN_v1(nn.Module):
    def __init__(self, n_1,density,lambda_model,mu_model):
        super(WaveletLayer_params_from_FCN_v1, self).__init__()
        self.neurons = n_1
        #self.precompute_wave_speeds(density,lambda_m,mu_m)
        self.density = density
        self.lambda_model = lambda_model  # This should be a function or nn.Module that computes lambda
        self.mu_model = mu_model  # This should be a function or nn.Module that computes mu
        self.lambda_model = self.lambda_model.to(device)
        self.mu_model = self.mu_model.to(device)

    def mother_wavelet(self, t, a=4):
        return (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))

    #def precompute_wave_speeds(self, density, lambda_m, mu_m):
        # Compute wave speeds using Lamé parameters
        # p_speed = sqrt((lambda_m + 2 * mu_m) / density)
        # s_speed = sqrt(mu_m / density)
        #self.p_speeds = torch.sqrt((lambda_m + 2 * mu_m) / density)
        #self.s_speeds = torch.sqrt(mu_m / density)

    def forward(self, x, scaling, translation, w_x, w_y):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Compute lambda_m and mu_m dynamically based on the current batch size
        lambda_m = mixture_model.compute_param(x_val,y_val, self.lambda_model)
        print("start")
        print("lambda",lambda_m)
        mu_m = mixture_model.compute_param(x_val, y_val, self.mu_model)
        print("mu",mu_m)
        p_speeds = torch.sqrt((lambda_m + 2 * mu_m) / self.density)
        print("p",p_speeds)
        s_speeds = torch.sqrt(mu_m / self.density)
        print("s",s_speeds)
        print("rho",self.density)
        print("done")
        print(torch.min(p_speeds),torch.max(p_speeds),torch.min(s_speeds),torch.max(s_speeds))
        p_speeds =  p_speeds - torch.min(p_speeds)
        p_speeds = p_speeds / torch.max(p_speeds)

        s_speeds = s_speeds - torch.min(s_speeds)
        s_speeds = s_speeds / torch.max(s_speeds)

        # Using precomputed wave speeds
        spatial_scaling = scaling * (p_speeds.unsqueeze(-1) + s_speeds.unsqueeze(-1)) / 2

        spatial_term = x_val.unsqueeze(-1) * w_x + y_val.unsqueeze(-1) * w_y
        adjusted_time_translation = t.unsqueeze(-1) - translation - spatial_term / spatial_scaling

        modulated_wavelet = self.mother_wavelet(spatial_scaling * adjusted_time_translation)
        return modulated_wavelet

class WaveletLayer_params_from_FCN_v1_Lame(nn.Module):
    def __init__(self, n_1):
        super(WaveletLayer_params_from_FCN_v1_Lame, self).__init__()
        self.neurons = n_1
    def mother_wavelet(self, t, a=4):
        return (1 - 2 * (a ** 2) * (t ** 2)) * torch.exp(-(a ** 2) * (t ** 2))
    def forward(self, x, scaling, translation, w_x, w_y):
        t, x_val, y_val,lambda_val,mu_val = x[:, 0], x[:, 1], x[:, 2], x[:,3],x[:,4]

        p_speeds = torch.sqrt((lambda_val + 2 * mu_val) / 100.0)
        s_speeds = torch.sqrt(mu_val / 100.0)
        spatial_scaling = scaling * (p_speeds.unsqueeze(-1) + s_speeds.unsqueeze(-1)) / 2

        spatial_term = x_val.unsqueeze(-1) * w_x + y_val.unsqueeze(-1) * w_y
        adjusted_time_translation = t.unsqueeze(-1) - translation - spatial_term / spatial_scaling

        modulated_wavelet = self.mother_wavelet(spatial_scaling * adjusted_time_translation)
        return modulated_wavelet

class PlaneWaveLayer_params_from_FCN(nn.Module):
    def __init__(self, n_1):
        super(PlaneWaveLayer_params_from_FCN, self).__init__()
        self.n_1 = n_1

    def forward(self, x, amplitude, k_x, k_y, omega, phi):
        # x[:, 0] is time, x[:, 1] is x-coordinate, x[:, 2] is y-coordinate
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Calculate the spatial and temporal components of the plane waves
        spatial_term = x_val.unsqueeze(-1) * k_x + y_val.unsqueeze(-1) * k_y
        temporal_term = omega * t.unsqueeze(-1)

        # Compute plane waves with dynamic amplitude and parameters
        plane_waves = amplitude * torch.cos(spatial_term + temporal_term + phi)

        return plane_waves

class NonLinear_WaveletLayer(nn.Module):
    def __init__(self, n_1):
        super(NonLinear_WaveletLayer, self).__init__()

        # Neuron count
        self.neurons = n_1

        # Learnable parameters for each neuron
        self.scaling = nn.Parameter(torch.randn(self.neurons))  # Scaling for the wavelet
        self.translation = nn.Parameter(torch.randn(self.neurons))  # Translating the wavelet in time
        self.amplitude = nn.Parameter(torch.randn(self.neurons))  # Amplitude scaling

        # Weights for spatial adjustment
        self.w_x = nn.Parameter(torch.randn(self.neurons))
        self.w_y = nn.Parameter(torch.randn(self.neurons))

        self.w_x2 =nn.Parameter(torch.randn(self.neurons))
        self.w_y2 =nn.Parameter(torch.randn(self.neurons))



    def mother_wavelet(self, t):
        # Ricker wavelet with a=4
        a = 4
        return (1 - 2 * a ** 2 * t ** 2) * torch.exp(-a ** 2 * t ** 2)

    def forward(self, x):
        t, x_val, y_val = x[:, 0], x[:, 1], x[:, 2]

        # Spatial term (you can modify this based on your requirements)
        spatial_term = x_val.unsqueeze(-1) * self.w_x + y_val.unsqueeze(-1) * self.w_y \
                       + x_val.unsqueeze(-1)**2 * self.w_x2 + y_val.unsqueeze(-1)**2 * self.w_y2 # NON linear terms

        # Calculate adjusted time translation based on spatial and other learned parameters
        adjusted_time_translation = t.unsqueeze(-1) + self.translation.unsqueeze(0) + spatial_term

        # Apply the learned transformations to the mother wavelet
        modulated_wavelet = self.amplitude * self.mother_wavelet(self.scaling.unsqueeze(0) * adjusted_time_translation)

        return modulated_wavelet

class DirectWaveletLayer(nn.Module):
    def __init__(self, input_dim, n_1):
        super(DirectWaveletLayer, self).__init__()

        self.neurons = n_1

        # Learnable parameters for each neuron
        self.amplitude = nn.Parameter(torch.randn(self.neurons))
        self.frequency = nn.Parameter(torch.randn(self.neurons))
        self.phase = nn.Parameter(torch.randn(self.neurons))

        # Transformation weights for input data
        self.transformation_weights = nn.Parameter(torch.randn(input_dim, self.neurons))

    def mother_wavelet(self, x):
        a = 4
        return (1 - 2 * a ** 2 * x ** 2) * torch.exp(-a ** 2 * x ** 2)

    def forward(self, x):
        # Linear transformation of input
        transformed_x = torch.matmul(x, self.transformation_weights)

        modulated_wavelet = self.amplitude * self.mother_wavelet(self.frequency * transformed_x + self.phase)

        return modulated_wavelet

class Fixed_AdvancedBesselLayer(nn.Module):
    def __init__(self, input_dim, n_1):
        super(Fixed_AdvancedBesselLayer, self).__init__()

        self.w_t = nn.Parameter(torch.randn(input_dim, n_1))
        self.b_t = nn.Parameter(torch.zeros(n_1))

        self.w_x = nn.Parameter(torch.randn(input_dim, n_1))
        self.b_x = nn.Parameter(torch.zeros(n_1))

        self.w_y = nn.Parameter(torch.randn(input_dim, n_1))
        self.b_y = nn.Parameter(torch.zeros(n_1))

        # Fixed parameters
        self.lambda_0 = 0.08152
        self.p1 = -0.2622096091
        self.q = 0.5701109315
        self.p_tilda1 = -0.3945665468
        self.p0 = 0.5948274174
        self.p2 = -0.04932081835
        self.p_tilda0 = 0.4544933991
        self.p_tilda2 = 0.03277620113
        self.eps = 1e-5


    def j0_approximation(self, x):
        factor1 = 1.0 / (((1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.25) * (1.0 + self.q * (x ** 2)))
        cosine_term = (self.p0 + self.p1 * (x ** 2) + self.p2 * (
                    1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) * torch.cos(x)
        sine_term = (self.p_tilda0 + self.p_tilda1 * (x ** 2)) * (
                    (1 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) + self.p_tilda2 * (
                            x ** 2) * torch.sin(x) / x
        result = factor1 * (cosine_term + sine_term)
        return result

    def forward(self, x):
        t = x @ self.w_t + self.b_t
        x_val = x @ self.w_x + self.b_x
        y_val = x @ self.w_y + self.b_y

        return self.bessel_approximation(t, x_val, y_val)

    def bessel_approximation(self, t, x_val, y_val):
        # At this point, t, x_val, y_val should all have shape [batch_size, n_1]

        # Calculate R
        R = torch.sqrt(x_val ** 2 + y_val ** 2 + self.eps)

        # Gaussian envelope
        term_inside_sqrt = 2.0 * x_val ** 2 + 2.0 * y_val ** 2 + self.eps
        gaussian_envelope = torch.exp(-0.5642 * (9.8208 * torch.sqrt(term_inside_sqrt) - 8.7387 * t) ** 2)

        # Bessel value
        atan_term = torch.atan2(y_val + self.eps, x_val + self.eps)
        bessel_value_x = gaussian_envelope * torch.cos(atan_term) * -1.0 * torch.exp(-t) * self.j0_approximation(
            9.8208 * R - 8.7387 * t + self.eps)

        return bessel_value_x

class AdvancedBesselLayer_params_from_FCN(nn.Module):
    def __init__(self, n_1):
        super(AdvancedBesselLayer_params_from_FCN, self).__init__()
        self.neurons = n_1

        # Fixed Bessel function approximation coefficients
        self.lambda_0 = 0.08152
        self.p1 = -0.2622096091
        self.q = 0.5701109315
        self.p_tilda1 = -0.3945665468
        self.p0 = 0.5948274174
        self.p2 = -0.04932081835
        self.p_tilda0 = 0.4544933991
        self.p_tilda2 = 0.03277620113
        self.eps = 1e-5  # Small constant to avoid numerical issues
        self.beta = 0.0  # This is scalar


    def j0_approximation(self, x):
        factor1 = 1.0 / (((1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.25) * (1.0 + self.q * (x ** 2)))
        cosine_term = (self.p0 + self.p1 * (x ** 2) + self.p2 * (
                1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) * torch.cos(x)
        sine_term = (self.p_tilda0 + self.p_tilda1 * (x ** 2)) * (
                (1 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) + self.p_tilda2 * (
                            x ** 2) * torch.sin(x) / x
        result = factor1 * (cosine_term + sine_term)
        return result

    def forward(self, x, A, k, omega, alpha, gamma, kappa, alpha_gaussian):
        # Extract individual variables from input
        t = x[:, 0].unsqueeze(-1)  # Adding last dimension for broadcasting
        x_val = x[:, 1].unsqueeze(-1)
        y_val = x[:, 2].unsqueeze(-1)

        # Calculate R with broadcasting
        R = torch.sqrt(x_val ** 2 + y_val ** 2 + self.eps)

        # Calculate the Gaussian envelope
        term_inside_sqrt = (gamma ** 2 * x_val ** 2) + (kappa ** 2 * y_val ** 2) + self.eps
        #print("gamma",gamma.shape,"x",x_val.shape,"kappa",kappa.shape,"y_val",y_val.shape)
        #print("alpha",alpha_gaussian.shape,"k",k.shape,"inside",term_inside_sqrt.shape,"omega",omega.shape,"t",t.shape)
        gaussian_envelope = torch.exp(-alpha_gaussian * (k * torch.sqrt(term_inside_sqrt) - omega * t) ** 2)

        # Calculate the angle for Bessel function argument
        atan_term = torch.atan2(y_val, x_val + self.eps)


        #DEBUGGING
        print("gaussian_envelope",torch.isnan(gaussian_envelope).any())
        print("atan",torch.isnan(atan_term).any())
        print("alpha",torch.isnan(alpha).any())
        print("cos",torch.isnan(torch.cos(atan_term - alpha)).any())
        print("A",torch.isnan(A).any())
        print("exp-t",torch.isnan(torch.exp(-t)).any())
        print("j0",torch.isnan( self.j0_approximation(k * R - omega * t + self.eps)).any())
        print("k",torch.isnan(k).any())
        print("R",torch.isnan(R).any())
        print("omega",torch.isnan(omega).any())



        # Calculate Bessel values with broadcasting
        bessel_value_x = gaussian_envelope * torch.cos(
            atan_term - alpha ) * A * torch.exp(-t) * self.j0_approximation(
            k * R - omega * t + self.eps
        )


        # Output shape: [batch_size, n_1]
        return bessel_value_x.squeeze()



class SIREN_NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(SIREN_NeuralNet, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers

        self.activation = activation

        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization


        # Sinusoidal Mapping Layer (replaces the traditional input layer)
        self.sinusoidal_layer = SinusoidalMappingLayer(input_dimension, n_1)

        # Adjust the first hidden layer to accept output from sinusoidal_layer
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.init_xavier()

    def forward(self, x):
        # Pass the input through the sinusoidal mapping layer first
        x = self.sinusoidal_layer(x)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            if isinstance(m, SinusoidalMappingLayer):
                # Example: Initialize weights with normal distribution
                nn.init.normal_(m.w1, mean=0.0, std=1.0)
                nn.init.zeros_(m.b1)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class FCN_all_params_SIREN_FCN(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1, n_hidden_layers_after, n_neurons_after):
        super(FCN_all_params_SIREN_FCN, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.neurons_after = n_neurons_after
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for sinusoidal mapping
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 2 * n_1)  # Output all sinusoidal mapping parameters

        # Sinusoidal mapping layer with parameters from FCN
        self.sinusoidal_layer = SinusoidalMappingLayer_params_from_FCN(n_1)

        # The subsequent hidden layers after sinusoidal mapping
        self.hidden_layers_after_sinusoidal = nn.ModuleList([nn.Linear(n_1, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for sinusoidal mapping from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        sinusoidal_params = self.params_layer(x_fcn)

        # Split parameters for passing to sinusoidal layer
        w1 = sinusoidal_params[:, :self.n_1]
        b1 = sinusoidal_params[:, self.n_1:]

        # Sinusoidal processing with the new parameters
        x_sinusoidal = self.sinusoidal_layer(input, w1, b1)

        # Now, pass through the subsequent hidden layers
        x_after = x_sinusoidal
        for l in self.hidden_layers_after_sinusoidal:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class Advanced_Bessel_and_FCN_NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(Advanced_Bessel_and_FCN_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        self.advanced_bessel_layer = AdvancedBesselLayer(n_1)
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        #print(input.shape)

        # Pass the input through the sinusoidal mapping layer first
        x = self.advanced_bessel_layer(input)
        #print(x.shape)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)



    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, AdvancedBesselLayer):  # Adjust for BesselActivation
                nn.init.uniform_(m.A, -1.0, 1.0)
                nn.init.uniform_(m.k, 9.0, 10.0)
                nn.init.uniform_(m.omega, 8.0, 9.0)
                nn.init.uniform_(m.alpha, 0.0, 3.0)
                # nn.init.uniform_(m.beta, 0.0, 0.01)
                nn.init.uniform_(m.gamma, 1.9, 2.1)
                nn.init.uniform_(m.kappa, 1.9, 2.1)
                nn.init.uniform_(m.alpha_gaussian, 0.50, 0.6)
        self.apply(init_weights)

class Fixed_Advanced_Bessel_and_FCN_NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(Fixed_Advanced_Bessel_and_FCN_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        self.advanced_bessel_layer = Fixed_AdvancedBesselLayer(input_dimension,n_1)
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        print("pre",input.shape)

        # Pass the input through the sinusoidal mapping layer first
        x = self.advanced_bessel_layer(input)
        print("post",x.shape)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)

class Planewave_and_FCN_NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(Planewave_and_FCN_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        self.plane_wave_layer = PlaneWaveLayer(n_1)
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        #print(input.shape)

        # Pass the input through the sinusoidal mapping layer first
        x = self.plane_wave_layer(input)
        #print(x.shape)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, PlaneWaveLayer):
                # Initialize parameters for PlaneWaveLayer
                nn.init.uniform_(m.k, -2, 2)  # Wavenumbers for x direction
                nn.init.uniform_(m.l, -2, 2)  # Wavenumbers for y direction
                nn.init.uniform_(m.v, 0.5, 1.5)  # Velocity of the wave
                nn.init.uniform_(m.A, -1, 1)  # Amplitude of the wave
                nn.init.uniform_(m.phi, 0, np.pi / 4)  # Phase term, initialized to a quarter of the full phase range
        self.apply(init_weights)

class MorletWavelet_and_FCN(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(MorletWavelet_and_FCN, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        self.waveletlayer = WaveletLayer(n_1)
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        #print(input.shape)

        # Pass the input through the sinusoidal mapping layer first
        x = self.waveletlayer(input)
        #print(x.shape)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, WaveletLayer):
                # Initialize parameters for WaveletLayer
                nn.init.uniform_(m.amplitude, -1, 1)  # Values between -1 and 1 for amplitude
                nn.init.uniform_(m.translation, -1, 1)  # Values between -1 and 1 to accommodate full range
                nn.init.uniform_(m.scaling, 0.5, 1.5)  # Values between 0.5 and 1.5 for scaling
                nn.init.uniform_(m.frequency, 0, 1)  # Values between 0 and 1 for frequency
                nn.init.uniform_(m.wavelength, -2, 2)  # Values between -2 and 2 to handle the spatial domain of [-1,1]
                nn.init.uniform_(m.w_x, -0.1, 0.1)
                nn.init.uniform_(m.w_y, -0.1, 0.1)
        self.apply(init_weights)

class FCN_Preceding_Dual_Wavelet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(FCN_Preceding_Dual_Wavelet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        # Initialize FCN layers
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.pre_wavelet_layer = nn.Linear(neurons, 3)  # To feed into the wavelet layers

        # Dual wavelet layers for ux and uy displacements
        self.waveletlayer_ux = WaveletLayer(n_1)
        self.waveletlayer_uy = WaveletLayer(n_1)

        self.init_xavier()


    def forward(self, input):
        # Pass through the FCN
        x = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        x = self.activation(self.pre_wavelet_layer(x))

        # Get the ux and uy displacements from the wavelet layers
        ux_displacement = self.waveletlayer_ux(x)
        uy_displacement = self.waveletlayer_uy(x)

        # Concatenate the outputs along the feature dimension
        final_output = torch.cat([ux_displacement, uy_displacement], dim=1)

        return final_output

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, WaveletLayer):
                # Initialize parameters for WaveletLayer
                nn.init.uniform_(m.amplitude, -1, 1)  # Values between -1 and 1 for amplitude
                nn.init.uniform_(m.translation, -1, 1)  # Values between -1 and 1 to accommodate full range
                nn.init.uniform_(m.scaling, 0.5, 1.5)  # Values between 0.5 and 1.5 for scaling
                nn.init.uniform_(m.frequency, 0, 1)  # Values between 0 and 1 for frequency
                nn.init.uniform_(m.wavelength, -2, 2)  # Values between -2 and 2 to handle the spatial domain of [-1,1]
                nn.init.uniform_(m.w_x, -0.1, 0.1)
                nn.init.uniform_(m.w_y, -0.1, 0.1)

        self.apply(init_weights)

class FCN_Direct_Dual_Wavelet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_Direct_Dual_Wavelet, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])

        # Direct dual wavelet layers
        self.waveletlayer_ux = DirectWaveletLayer(neurons, n_1)
        self.waveletlayer_uy = DirectWaveletLayer(neurons, n_1)

        self.init_xavier()

    def forward(self, input):
        x = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x = self.activation(l(x))

        ux_displacement = self.waveletlayer_ux(x)
        uy_displacement = self.waveletlayer_uy(x)

        final_output = torch.cat([ux_displacement, uy_displacement], dim=1)

        return final_output

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, DirectWaveletLayer):
                nn.init.uniform_(m.amplitude, -1, 1)  # Values between -1 and 1 for amplitude
                nn.init.uniform_(m.frequency, 0, 1)  # Values between 0 and 1 for frequency
                nn.init.uniform_(m.phase, 0, 2 * torch.pi)  # Values between 0 and 2π for phase

        self.apply(init_weights)

class FCN_Direct_Singular_Wavelet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_Direct_Singular_Wavelet, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])

        # Singular wavelet layer
        self.waveletlayer = DirectWaveletLayer(neurons, n_1)

        # Final linear layer to output 2D result
        self.final_layer = nn.Linear(n_1, output_dimension)

        self.init_xavier()

    def forward(self, input):
        x = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x = self.activation(l(x))

        x_wavelet = self.waveletlayer(x)

        final_output = self.final_layer(x_wavelet)

        return final_output

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, DirectWaveletLayer):
                nn.init.uniform_(m.amplitude, -1, 1)  # Values between -1 and 1 for amplitude
                nn.init.uniform_(m.frequency, 0, 1)  # Values between 0 and 1 for frequency
                nn.init.uniform_(m.phase, 0, 2 * torch.pi)  # Values between 0 and 2π for phase

        self.apply(init_weights)

class Nonlinear_Wavelet_and_FCN(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers,neurons, regularization_param, regularization_exp,
                 retrain_seed,activation,n_1):
        super(Nonlinear_Wavelet_and_FCN, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        # Adjust the first hidden layer to accept output from BESSEL layer

        self.waveletlayer = NonLinear_WaveletLayer(n_1)
        self.first_hidden_layer = nn.Linear(n_1, self.neurons)
        # The subsequent hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 2)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        #print(input.shape)

        # Pass the input through the sinusoidal mapping layer first
        x = self.waveletlayer(input)
        #print(x.shape)

        # Now, pass through the first hidden layer
        x = self.activation(self.first_hidden_layer(x))

        for l in self.hidden_layers:
            x = self.activation(l(x))

        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, WaveletLayer):
                # Initialize parameters for WaveletLayer
                nn.init.uniform_(m.amplitude, -1, 1)  # Values between -1 and 1 for amplitude
                nn.init.uniform_(m.translation, -1, 1)  # Values between -1 and 1 to accommodate full range
                nn.init.uniform_(m.scaling, 0.5, 1.5)  # Values between 0.5 and 1.5 for scaling

                nn.init.uniform_(m.w_x, -0.1, 0.1)
                nn.init.uniform_(m.w_y, -0.1, 0.1)
                nn.init.uniform_(m.w_x2, -0.05, 0.05)
                nn.init.uniform_(m.w_y2, -0.05, 0.05)

        self.apply(init_weights)

class FCN_Amplitude_Wavelet(nn.Module):

    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_Amplitude_Wavelet, self).__init__()

        self.input_dimension = input_dimension
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        self.output_dimension = output_dimension

        # FCN to produce amplitude
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.output_amplitude_layer = nn.Linear(neurons, 1)  # 1D amplitude output

        # Wavelet layer
        self.waveletlayer = WaveletLayer_no_amplitude(n_1)

        # Linear combination of wavelets to produce ux, uy
        self.final_layer = nn.Linear(n_1, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get amplitude from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        amplitude = self.output_amplitude_layer(x_fcn)
        print("amplitude pre",amplitude.shape)

        # Wavelet processing
        x_wavelet = self.waveletlayer(input)
        print("wavelet",x_wavelet.shape)

        # Modulate amplitude
        x_modulated = amplitude * x_wavelet
        print("x_modulated",x_modulated.shape)

        final_output = self.final_layer(x_modulated)

        return final_output

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, WaveletLayer):
                nn.init.uniform_(m.scaling, 0, 1)
                nn.init.uniform_(m.translation, 0, 2 * torch.pi)

        self.apply(init_weights)

class FCN_Amplitude_Planewave(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_Amplitude_Planewave, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        self.n_1 = n_1  # Number of planewave components

        # FCN to produce amplitude
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.output_amplitude_layer = nn.Linear(neurons, n_1)  # Amplitude for each planewave component

        # Planewave layer
        self.planewave_layer = PlaneWaveLayer(n_1)

        # Linear combination of planewaves to produce final output
        self.final_layer = nn.Linear(n_1, output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get amplitude from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        amplitude = self.output_amplitude_layer(x_fcn)

        # Planewave processing
        x_planewave = self.planewave_layer(input)

        # Modulate amplitude
        x_modulated = amplitude * x_planewave

        final_output = self.final_layer(x_modulated)

        return final_output

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, PlaneWaveLayer):
                # Initialize parameters for PlaneWaveLayer
                nn.init.uniform_(m.k, -2, 2)  # Wavenumbers for x direction
                nn.init.uniform_(m.l, -2, 2)  # Wavenumbers for y direction
                nn.init.uniform_(m.v, 0.5, 1.5)  # Velocity of the wave
                nn.init.uniform_(m.phi, 0, np.pi / 4)  # Phase term, initialized to a quarter of the full phase range

        self.apply(init_weights)

class FCN_all_params_Wavelet_Modulation(nn.Module):
    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_all_params_Wavelet_Modulation, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for wavelet
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 4 * n_1)  # Output all wavelet parameters

        # Wavelet layer
        self.waveletlayer = WaveletLayer_params_from_FCN(n_1)
        # Linear combination of wavelets to produce final output
        self.final_layer = nn.Linear(n_1, self.output_dimension)  # 2 for 2D outputs, ux and uy

        self.init_xavier()

    def forward(self, input):
        # Get parameters for wavelets from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        wavelet_params = self.params_layer(x_fcn)

        # Split parameters for passing to wavelet layer
        scaling = wavelet_params[:, :self.n_1]
        translation = wavelet_params[:, self.n_1:2 * self.n_1]
        w_x = wavelet_params[:, 2 * self.n_1:3 * self.n_1]
        w_y = wavelet_params[:, 3 * self.n_1:]

        # Wavelet processing with the new parameters
        x_wavelet = self.waveletlayer(input, scaling, translation, w_x, w_y)

        # Linear combination of wavelets to produce final output
        final_output = self.final_layer(x_wavelet)

        return final_output

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_Planewave_Modulation(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1):
        super(FCN_all_params_Planewave_Modulation, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for planewaves
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 5 * n_1)  # 5 * n_1 for amplitude, k_x, k_y, omega, phi

        # Planewave layer
        self.planewavelayer = PlaneWaveLayer_params_from_FCN(n_1)

        # Linear combination of planewaves to produce final output
        self.final_layer = nn.Linear(n_1, output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for planewaves from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        planewave_params = self.params_layer(x_fcn)

        # Split parameters for passing to planewave layer
        amplitude = planewave_params[:, :self.n_1]
        k_x = planewave_params[:, self.n_1:2 * self.n_1]
        k_y = planewave_params[:, 2 * self.n_1:3 * self.n_1]
        omega = planewave_params[:, 3 * self.n_1:4 * self.n_1]
        phi = planewave_params[:, 4 * self.n_1:]

        # Planewave processing with the new parameters
        x_planewave = self.planewavelayer(input, amplitude, k_x, k_y, omega, phi)

        # Linear combination of planewaves to produce final output
        final_output = self.final_layer(x_planewave)

        return final_output

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_Wavelet_FCN(nn.Module):
    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,n_hidden_layers_after,n_neurons_after):
        super(FCN_all_params_Wavelet_FCN, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.neurons_after = n_neurons_after
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for wavelet
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 4 * n_1)  # Output all wavelet parameters

        # Wavelet layer
        self.waveletlayer = WaveletLayer_params_from_FCN(n_1)
        # Linear combination of wavelets to produce final output
        self.first_hidden_layer_after_wavelet = nn.Linear(n_1, self.neurons_after)
        # The subsequent hidden layers
        self.hidden_layers_after_wavelet  = nn.ModuleList([nn.Linear(self.neurons_after, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for wavelets from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        wavelet_params = self.params_layer(x_fcn)

        # Split parameters for passing to wavelet layer
        scaling = wavelet_params[:, :self.n_1]
        translation = wavelet_params[:, self.n_1:2 * self.n_1]
        w_x = wavelet_params[:, 2 * self.n_1:3 * self.n_1]
        w_y = wavelet_params[:, 3 * self.n_1:]

        # Wavelet processing with the new parameters
        x_wavelet = self.waveletlayer(input, scaling, translation, w_x, w_y)

        # Now, pass through the first hidden layer
        x_after = self.activation(self.first_hidden_layer_after_wavelet(x_wavelet))

        for l in self.hidden_layers_after_wavelet:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_Wavelet_gaussian_FCN(nn.Module):
    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,n_hidden_layers_after,n_neurons_after):
        super(FCN_all_params_Wavelet_gaussian_FCN, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.neurons_after = n_neurons_after
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for wavelet
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 6 * n_1)  # Output all wavelet parameters

        # Wavelet layer
        self.waveletlayer = WaveletLayer_gaussian_params_from_FCN(n_1)
        # Linear combination of wavelets to produce final output
        self.first_hidden_layer_after_wavelet = nn.Linear(n_1, self.neurons_after)
        # The subsequent hidden layers
        self.hidden_layers_after_wavelet  = nn.ModuleList([nn.Linear(self.neurons_after, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # FCN processing to get parameters
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        wavelet_params = self.params_layer(x_fcn)

        # Extract wavelet and Gaussian parameters
        scaling = wavelet_params[:, :self.n_1]
        translation = wavelet_params[:, self.n_1:2 * self.n_1]
        w_x = wavelet_params[:, 2 * self.n_1:3 * self.n_1]
        w_y = wavelet_params[:, 3 * self.n_1:4 * self.n_1]
        gaussian_mean = wavelet_params[:, 4 * self.n_1:5 * self.n_1]
        gaussian_variance = wavelet_params[:, 5 * self.n_1:6 * self.n_1]

        # Wavelet processing
        x_wavelet = self.waveletlayer(input, scaling, translation, w_x, w_y, gaussian_mean, gaussian_variance)

        # Post-wavelet FCN processing
        x_after = self.activation(self.first_hidden_layer_after_wavelet(x_wavelet))
        for l in self.hidden_layers_after_wavelet:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class Lame_FCN_all_params_Wavelet_FCN_v1(nn.Module):
    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,n_hidden_layers_after,n_neurons_after):
        super(Lame_FCN_all_params_Wavelet_FCN_v1, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.neurons_after = n_neurons_after
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for wavelet
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 4 * n_1)  # Output all wavelet parameters

        # Wavelet layer
        self.waveletlayer = WaveletLayer_params_from_FCN_v1_Lame(n_1)
        # Linear combination of wavelets to produce final output
        self.first_hidden_layer_after_wavelet = nn.Linear(n_1, self.neurons_after)
        # The subsequent hidden layers
        self.hidden_layers_after_wavelet  = nn.ModuleList([nn.Linear(self.neurons_after, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for wavelets from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        wavelet_params = self.params_layer(x_fcn)

        # Split parameters for passing to wavelet layer
        scaling = wavelet_params[:, :self.n_1]
        translation = wavelet_params[:, self.n_1:2 * self.n_1]
        w_x = wavelet_params[:, 2 * self.n_1:3 * self.n_1]
        w_y = wavelet_params[:, 3 * self.n_1:]

        # Wavelet processing with the new parameters
        x_wavelet = self.waveletlayer(input, scaling, translation, w_x, w_y)

        # Now, pass through the first hidden layer
        x_after = self.activation(self.first_hidden_layer_after_wavelet(x_wavelet))

        for l in self.hidden_layers_after_wavelet:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_Wavelet_FCN_v1(nn.Module):
    def __init__(self, input_dimension,output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,n_hidden_layers_after,n_neurons_after,density,lambda_m,mu_m):
        super(FCN_all_params_Wavelet_FCN_v1, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1
        self.neurons_after = n_neurons_after
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for wavelet
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 4 * n_1)  # Output all wavelet parameters

        # Wavelet layer
        self.waveletlayer = WaveletLayer_params_from_FCN_v1(n_1,density,lambda_m,mu_m)
        # Linear combination of wavelets to produce final output
        self.first_hidden_layer_after_wavelet = nn.Linear(n_1, self.neurons_after)
        # The subsequent hidden layers
        self.hidden_layers_after_wavelet  = nn.ModuleList([nn.Linear(self.neurons_after, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for wavelets from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        wavelet_params = self.params_layer(x_fcn)

        # Split parameters for passing to wavelet layer
        scaling = wavelet_params[:, :self.n_1]
        translation = wavelet_params[:, self.n_1:2 * self.n_1]
        w_x = wavelet_params[:, 2 * self.n_1:3 * self.n_1]
        w_y = wavelet_params[:, 3 * self.n_1:]

        # Wavelet processing with the new parameters
        x_wavelet = self.waveletlayer(input, scaling, translation, w_x, w_y)

        # Now, pass through the first hidden layer
        x_after = self.activation(self.first_hidden_layer_after_wavelet(x_wavelet))

        for l in self.hidden_layers_after_wavelet:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_Planewave_FCN(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,n_hidden_layers_after,n_neurons_after):
        super(FCN_all_params_Planewave_FCN, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.neurons_after = n_neurons_after
        self.n_1 = n_1
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # FCN to produce parameters for planewaves
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 5 * n_1)  # 5 * n_1 for amplitude, k_x, k_y, omega, phi

        # Planewave layer
        self.planewavelayer = PlaneWaveLayer_params_from_FCN(n_1)

        self.first_hidden_layer_after_planewave = nn.Linear(n_1, self.neurons_after)
        # The subsequent hidden layers
        self.hidden_layers_after_planewave = nn.ModuleList([nn.Linear(self.neurons_after, self.neurons_after) for _ in range(n_hidden_layers_after - 2)])
        self.output_layer = nn.Linear(self.neurons_after, self.output_dimension)

        self.init_xavier()

    def forward(self, input):
        # Get parameters for planewaves from FCN
        x_fcn = self.activation(self.first_hidden_layer(input))
        for l in self.hidden_layers:
            x_fcn = self.activation(l(x_fcn))
        planewave_params = self.params_layer(x_fcn)

        # Split parameters for passing to planewave layer
        amplitude = planewave_params[:, :self.n_1]
        k_x = planewave_params[:, self.n_1:2 * self.n_1]
        k_y = planewave_params[:, 2 * self.n_1:3 * self.n_1]
        omega = planewave_params[:, 3 * self.n_1:4 * self.n_1]
        phi = planewave_params[:, 4 * self.n_1:]

        # Planewave processing with the new parameters
        x_planewave = self.planewavelayer(input, amplitude, k_x, k_y, omega, phi)

        # Now, pass through the first hidden layer
        x_after = self.activation(self.first_hidden_layer_after_planewave(x_planewave))

        for l in self.hidden_layers_after_planewave:
            x_after = self.activation(l(x_after))

        return self.output_layer(x_after)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)

class FCN_all_params_AdvancedBessel_Modulation(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, activation, n_1,):
        super(FCN_all_params_AdvancedBessel_Modulation, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_1 = n_1

        # FCN to produce parameters for Bessel
        self.first_hidden_layer = nn.Linear(input_dimension, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(n_hidden_layers - 1)])
        self.params_layer = nn.Linear(neurons, 7 * n_1)  # Output all Bessel parameters

        # Bessel layer
        self.advanced_bessel_layer = AdvancedBesselLayer_params_from_FCN(n_1)
        # Output layer
        self.output_layer = nn.Linear(n_1, self.output_dimension)
        self.init_xavier()

    def forward(self, x):
        # Generate Bessel parameters from FCN
        x_fcn = torch.tanh(self.first_hidden_layer(x))
        for l in self.hidden_layers:
            x_fcn = torch.tanh(l(x_fcn))
        bessel_params = self.params_layer(x_fcn)

        # Extract parameters for Bessel layer
        A = bessel_params[:, :self.n_1]
        k = bessel_params[:, self.n_1:2 * self.n_1]
        omega = bessel_params[:, 2 * self.n_1:3 * self.n_1]
        alpha = bessel_params[:, 3 * self.n_1:4 * self.n_1]
        gamma = bessel_params[:, 4 * self.n_1:5 * self.n_1]
        kappa = bessel_params[:, 5 * self.n_1:6 * self.n_1]
        alpha_gaussian = bessel_params[:, 6 * self.n_1:]

        # Bessel processing with the new parameters
        x_bessel = self.advanced_bessel_layer(x, A, k, omega, alpha, gamma, kappa, alpha_gaussian)

        # Final output
        return self.output_layer(x_bessel)

    def init_xavier(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)


#BASE NN CLAS FOR PINNS
class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed,activation):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function changed to softplus to match paper
        #if config['Network']['activation'] == 'tanh':
            #self.activation = nn.Tanh()
        self.activation = activation
        #else:
            #print("unknown activation function", config['Network'].activation)
            #exit()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, WaveletActivation):
                nn.init.uniform_(m.amplitude, -1, 1)
                nn.init.uniform_(m.frequency, 0, 1)
                nn.init.uniform_(m.phase, 0, 2*torch.pi)

            elif isinstance(m, WaveletActivation_2):
                nn.init.uniform_(m.amplitude, -1, 1)
                nn.init.uniform_(m.scaling, 0, 1)
                nn.init.uniform_(m.translation, -1, 1)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class PlaneWave_NeuralNet_old(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed):
        super(PlaneWave_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        self.activation = PlaneWaveActivation(self.neurons)
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.input_layer = nn.Linear(3, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, input):
        txy = input[:, :3].contiguous().view(-1, 3)
        sx, sy = input[:, 3], input[:, 4]

        x = txy
        x = self.activation(self.input_layer(x), sx, sy)
        for k, l in enumerate(self.hidden_layers):
            # After processing each hidden layer, concatenate the output with the original txy
            x = torch.cat((txy, x), dim=1)
            x = l(x)
            x = self.activation(x, sx, sy)
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                if isinstance(self.activation, PlaneWaveActivation):
                    g = 1.0  # or some other value based on empirical results
                    torch.nn.init.xavier_uniform_(m.weight, gain=g)
                    m.bias.data.fill_(0.01)
                else:
                    g = nn.init.calculate_gain('tanh')
                    torch.nn.init.xavier_uniform_(m.weight, gain=g)
                    m.bias.data.fill_(0.01)
            elif isinstance(m, PlaneWaveActivation):
                # Initialize parameters of PlaneWaveActivation to uniform random values between 0 and 1
                nn.init.uniform_(m.k, 0, 1)
                nn.init.uniform_(m.l, 0, 1)
                nn.init.uniform_(m.v, 0, 1)
                nn.init.uniform_(m.A, 0, 1)
                if hasattr(m, 'phi'):  # Check if phi exists, and initialize it as well
                    nn.init.uniform_(m.phi, 0, 1)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class PlaneWave_NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed):
        super(PlaneWave_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        self.activation = PlaneWaveActivation(neurons)
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed

        # Layers for t, x, y
        self.input_layer_txy = nn.Linear(3, self.neurons)
        self.hidden_layers_txy = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])

        # Layers for wave patterns
        self.input_layer_wave = nn.Linear(1, self.neurons)  # For the initial wave pattern
        self.hidden_layers_wave = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])

        # Final output layer
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        txy = input[:, :3].contiguous().view(-1, 3)
        sx, sy = input[:, 3], input[:, 4]

        # Initial transformation
        transformed_txy = self.input_layer_txy(txy)
        wave_pattern = self.activation(transformed_txy, sx, sy)

        for l_txy, l_wave in zip(self.hidden_layers_txy, self.hidden_layers_wave):
            # Transforming the t,x,y and wave_pattern separately
            transformed_txy = l_txy(transformed_txy)
            transformed_wave = l_wave(wave_pattern)

            # Producing new wave pattern
            new_wave_pattern = self.activation(transformed_txy, sx, sy)

            # Modifying the new wave pattern with transformed wave pattern from previous layer
            new_wave_pattern += transformed_wave

            # Combining the wave patterns
            wave_pattern = wave_pattern + new_wave_pattern

        # Producing the final output
        return self.output_layer(wave_pattern)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, PlaneWaveActivation):
                nn.init.uniform_(m.k, 0, 1)
                nn.init.uniform_(m.l, 0, 1)
                nn.init.uniform_(m.v, 0, 1)
                nn.init.uniform_(m.A, 0, 1)
                if hasattr(m, 'phi'):
                    nn.init.uniform_(m.phi, 0, 1)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

#JUST ONE SINGLE PLANE WAVE ACTIVTAION LAYER, NO TXY TRANSOFMRATION OR SUCCSSESSFUL WAVE SUMMATION
class SimplePlaneWave_NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, neurons, regularization_param, regularization_exp,
                 retrain_seed):
        super(SimplePlaneWave_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = PlaneWaveActivation(neurons)
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed

        # Final output layer
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):
        txy = input[:, :3].contiguous().view(-1, 3)
        sx, sy = input[:, 3], input[:, 4]

        # Passing the input through the plane wave activation
        wave_pattern = self.activation(txy, sx, sy)

        # Producing the final output
        return self.output_layer(wave_pattern)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, PlaneWaveActivation):
                nn.init.uniform_(m.k, 6, 60)
                nn.init.uniform_(m.l, 6, 60)
                nn.init.uniform_(m.v, 0, 1)
                nn.init.uniform_(m.A, -1.0, 1.0)
                if hasattr(m, 'phi'):
                    nn.init.uniform_(m.phi, 0, 6.2831853)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class SimpleBesselWave_NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, neurons, regularization_param, regularization_exp,
                 retrain_seed):
        super(SimpleBesselWave_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = BesselActivation(neurons)  # Use the BesselActivation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)


        # Final output layer
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):

        txy = input[:, :3].contiguous().view(-1, 3)
        sx, sy = input[:, 3], input[:, 4]

        txy = self.input_layer(input)

        # Passing the input through the bessel wave activation
        wave_pattern = self.activation(txy, sx, sy)

        # Producing the final output
        return self.output_layer(wave_pattern)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, BesselActivation):  # Adjust for BesselActivation
                nn.init.uniform_(m.A, -1, 1)
                nn.init.uniform_(m.k, 5, 65)
                nn.init.uniform_(m.omega, 3, 60)
                nn.init.uniform_(m.alpha,-3.1415926,3.1415926)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class SimpleFarFieldWave_NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, neurons, regularization_param, regularization_exp,
                 retrain_seed):
        super(SimpleFarFieldWave_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = FarFieldActivation(neurons)  # Use the BesselActivation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.output_layer = nn.Linear(2 * self.neurons, self.output_dimension)

        # Final output layer
        #self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):

        #txy = input[:, :3].contiguous().view(-1, 3)
        #sx, sy = input[:, 3], input[:, 4]

        #print("input pre",input)
        #input = self.input_layer(input)

        # Passing the input through the bessel wave activation
        #print("input post",input)
        wave_pattern = self.activation(input)
        wave_pattern = wave_pattern.view(-1, 2 * self.neurons)

        # Producing the final output
        return self.output_layer(wave_pattern)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, FarFieldActivation):  # Adjust for BesselActivation
                nn.init.uniform_(m.theta_1, 0.0, 2 * np.pi)
                nn.init.uniform_(m.theta_2, 0.0, 2 * np.pi)
                nn.init.uniform_(m.theta_3, 0.0, 2 * np.pi)
                nn.init.uniform_(m.theta_4, 0.0, 2 * np.pi)
                nn.init.uniform_(m.M0, -1.0, 1.0)
                nn.init.uniform_(m.T,-1.0, 1.0)
                nn.init.uniform_(m.offset, 0.0, 0.1)
        self.apply(init_weights)


class Advanced_Bessel_NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, neurons, regularization_param, regularization_exp,
                 retrain_seed):
        super(Advanced_Bessel_NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons in the hidden layer
        self.neurons = neurons
        self.activation = Advanced_Bessel_Activation(neurons)  # Use the BesselActivation
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        self.retrain_seed = retrain_seed
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.output_layer = nn.Linear(2 * self.neurons, self.output_dimension)

        # Initialize weights
        self.init_xavier()

    def forward(self, input):

        wave_pattern = self.activation(input)
        self.activation.gamma.data.clamp_(min=0)
        self.activation.kappa.data.clamp_(min=0)
        # Splitting the tensor into ux and uy parts
        ux_parts = wave_pattern[:, :self.neurons]
        uy_parts = wave_pattern[:, self.neurons:]

        # Summing over neurons
        ux_summed = ux_parts.sum(dim=1, keepdim=True)
        uy_summed = uy_parts.sum(dim=1, keepdim=True)

        # Concatenating the summed results
        output = torch.cat([ux_summed, uy_summed], dim=1)
        #print("wave pattern shape pre",wave_pattern.shape)
        #wave_pattern = wave_pattern.view(-1, 2 * self.neurons)
        #print("wave pattern shape post", wave_pattern.shape)
        return output


    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.01)
            elif isinstance(m, Advanced_Bessel_Activation):  # Adjust for BesselActivation
                nn.init.uniform_(m.A, -1.0, 1.0)
                nn.init.uniform_(m.k, 9.6, 9.8)
                nn.init.uniform_(m.omega, 8.4, 8.8)
                nn.init.uniform_(m.alpha, 0.0, 0.01)
                #nn.init.uniform_(m.beta, 0.0, 0.01)
                nn.init.uniform_(m.gamma,1.9, 2.1)
                nn.init.uniform_(m.kappa, 1.9, 2.1)
                nn.init.uniform_(m.alpha_gaussian, 0.54, 0.58)
        self.apply(init_weights)


class PlaneWaveActivation_old(nn.Module):
    def __init__(self,neurons):
        super(PlaneWaveActivation, self).__init__()
        # Initialize parameters for each neuron
        self.neurons = neurons
        self.k = nn.Parameter(torch.randn(self.neurons))
        self.l = nn.Parameter(torch.randn(self.neurons))
        self.v = nn.Parameter(torch.randn(self.neurons))
        self.A = nn.Parameter(torch.randn(self.neurons))
        self.phi = nn.Parameter(torch.randn(self.neurons))

    def forward(self, txy, sx, sy):
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]
        distance_term = torch.sqrt((x - sx) ** 2 + (y - sy) ** 2)

        total_wave = torch.zeros_like(t)
        for i in range(self.neurons):
            theta = self.k[i] * x + self.l[i] * y - self.v[i] * t + self.phi[i] * distance_term
            complex_exp = self.A[i] * torch.exp(1j * theta)
            total_wave += torch.real(complex_exp)

        return total_wave

class PlaneWaveActivation(nn.Module):
    def __init__(self, neurons):
        super(PlaneWaveActivation, self).__init__()
        # Initialize parameters for each neuron
        self.neurons = neurons
        self.k = nn.Parameter(torch.randn(self.neurons))
        self.l = nn.Parameter(torch.randn(self.neurons))
        self.v = nn.Parameter(torch.randn(self.neurons))
        self.A = nn.Parameter(torch.randn(self.neurons))
        self.phi = nn.Parameter(torch.randn(self.neurons))

    def forward(self, txy, sx, sy):
        self.log_parameters_to_wandb()
        self.plot_plane_wave_function()
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]
        distance_term = torch.sqrt(((x - sx) ** 2 + (y - sy) ** 2)+ 1e-8)


        waves = []
        for i in range(self.neurons):
            theta = self.k[i] * x + self.l[i] * y - self.v[i] * t + self.phi[i] * distance_term
            complex_exp = self.A[i] * torch.exp(1j * theta)
            waves.append(torch.real(complex_exp).unsqueeze(-1))

        return torch.cat(waves, dim=-1) # Concatenate along the last dimension

    def plot_plane_wave_function(self, sx=0.0, sy=0.0, t_range=(0, 1), x_range=(-1, 1), y_range=(-1, 1)):
        x_vals = torch.linspace(x_range[0], x_range[1], 100)
        y_vals = torch.linspace(y_range[0], y_range[1], 100)
        t_vals = torch.linspace(t_range[0], t_range[1], 3)
        X, Y = torch.meshgrid(x_vals, y_vals)

        for i, (a, k, l, v, phi) in enumerate(zip(self.A.detach().cpu().numpy(),
                                                  self.k.detach().cpu().numpy(),
                                                  self.l.detach().cpu().numpy(),
                                                  self.v.detach().cpu().numpy(),
                                                  self.phi.detach().cpu().numpy())):
            for t in t_vals:
                distance_term = torch.sqrt(((X - sx) ** 2 + (Y - sy) ** 2) + 1e-8)
                theta = k * X + l * Y - v * t + phi * distance_term
                plane_wave_vals = (a * torch.exp(1j * theta)).real.detach().cpu().numpy()

                plt.imshow(plane_wave_vals, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                           origin='lower', aspect='auto')
                plt.colorbar(label='Plane Wave Value')
                plt.title(f'Neuron {i} at time {t:.2f}')
                plt.xlabel('X')
                plt.ylabel('Y')

                # Log the image to wandb
                wandb.log({f"PlaneWave_plot_neuron_{i}_time_{t:.2f}": wandb.Image(plt)})

                # Clear the plot for the next image
                plt.clf()

    def log_parameters_to_wandb(self):
        # Extracting parameters
        A_values = self.A.cpu().detach().numpy()
        k_values = self.k.cpu().detach().numpy()
        l_values = self.l.cpu().detach().numpy()
        v_values = self.v.cpu().detach().numpy()
        phi_values = self.phi.cpu().detach().numpy()

        # Neuron numbers for x-axis
        neurons = list(range(1, len(A_values) + 1))

        # Log each parameter value per neuron
        for i, neuron in enumerate(neurons):
            wandb.log({f"A_neuron_{neuron}": A_values[i],
                       f"k_neuron_{neuron}": k_values[i],
                       f"l_neuron_{neuron}": l_values[i],
                       f"v_neuron_{neuron}": v_values[i],
                       f"phi_neuron_{neuron}": phi_values[i]})

class BesselActivation(nn.Module):
    def __init__(self, neurons):
        super(BesselActivation, self).__init__()
        # Initialize parameters for each neuron
        self.neurons = neurons
        self.k = nn.Parameter(torch.randn(self.neurons))  # Wave number
        self.A = nn.Parameter(torch.randn(self.neurons))  # Amplitude
        self.omega = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
        self.alpha = nn.Parameter(torch.randn(self.neurons))

    def forward(self, txy, sx, sy):

        self.log_parameters_to_wandb()
        self.plot_bessel_function()
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]
        r = torch.sqrt(((x - sx) ** 2 + (y - sy) ** 2) +1e-8)  # Distance from source

        bessel_waves = []
        for i in range(self.neurons):
            #theta = self.k[i] * r - self.omega[i] * t
            bessel_wave = torch.cos(torch.atan2(y,x) - self.alpha[i]) *self.A[i] * torch.special.bessel_j0(self.k[i] * r - self.omega[i] * t)
            bessel_waves.append(bessel_wave)

        return torch.stack(bessel_waves, dim=-1)

    def log_weights(self):
        for i, (a, k, omega) in enumerate(zip(self.A, self.k, self.omega)):
            wandb.log({f"A_neuron_{i}": a.item(), f"k_neuron_{i}": k.item(), f"omega_neuron_{i}": omega.item()})

    def plot_bessel_function(self, sx=0.0, sy=0.0, t_range=(0, 1), x_range=(-1, 1), y_range=(-1, 1)):
        x_vals = torch.linspace(x_range[0], x_range[1], 100)
        y_vals = torch.linspace(y_range[0], y_range[1], 100)
        t_vals = torch.linspace(t_range[0], t_range[1], 3)
        X, Y = torch.meshgrid(x_vals, y_vals)

        for i, (a, k, omega,alpha) in enumerate(zip(self.A.detach().cpu().numpy(),
                                              self.k.detach().cpu().numpy(),
                                              self.omega.detach().cpu().numpy(),
                                                    self.alpha.detach().cpu().numpy())):
            for t in t_vals:
                R = torch.sqrt(((X - sx) ** 2 + (Y - sy) ** 2) + 1e-8)
                theta = k * R - omega * t
                bessel_vals = torch.cos(torch.atan2(Y,X) - alpha) *a * torch.special.bessel_j0(theta).detach().cpu().numpy()

                plt.imshow(bessel_vals, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                           origin='lower', aspect='auto')
                plt.colorbar(label='Bessel Value')
                plt.title(f'Neuron {i} at time {t:.2f}')
                plt.xlabel('X')
                plt.ylabel('Y')

                # Log the image to wandb
                wandb.log({f"Bessel_plot_neuron_{i}_time_{t:.2f}": wandb.Image(plt)})

                # Clear the plot for the next image
                plt.clf()

    def log_parameters_to_wandb(self):
        # Extracting parameters
        A_values = self.A.cpu().detach().numpy()
        k_values = self.k.cpu().detach().numpy()
        omega_values = self.omega.cpu().detach().numpy()

        # Neuron numbers for x-axis
        neurons = list(range(1, len(A_values) + 1))

        # Plotting and logging A values
        plt.figure()
        plt.bar(neurons, A_values)
        plt.xlabel('Neuron Number')
        plt.ylabel('A Value')
        plt.title('A Values per Neuron')
        wandb.log({"A Values": plt})

        # Plotting and logging k values
        plt.figure()
        plt.bar(neurons, k_values)
        plt.xlabel('Neuron Number')
        plt.ylabel('k Value')
        plt.title('k Values per Neuron')
        wandb.log({"k Values": plt})

        # Plotting and logging omega values
        plt.figure()
        plt.bar(neurons, omega_values)
        plt.xlabel('Neuron Number')
        plt.ylabel('omega Value')
        plt.title('omega Values per Neuron')
        wandb.log({"omega Values": plt})

class FarFieldActivation(nn.Module):

    def __init__(self, neurons):
        super(FarFieldActivation, self).__init__()
        # Initialize parameters for each neuron
        self.neurons = neurons
        self.T = nn.Parameter(torch.randn(self.neurons))  # Wave number
        self.M0 = nn.Parameter(torch.randn(self.neurons))  # Amplitude
        self.theta_1 = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
        self.theta_2 = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
        self.theta_3 = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
        self.theta_4 = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
        self.offset = nn.Parameter(torch.randn(self.neurons))  # Angular frequency
    def forward(self, input):
        #self.plot_farfield_activation()
        #self.log_parameters_to_wandb()
        alpha_ = torch.tensor(np.sqrt((20.0 + 2.0 * 30.0) / 100.0))
        beta_ = torch.tensor(np.sqrt(30.0 / 100.0))

        #print(input)

        t = input[:, 0]
        x = input[:, 1]
        y = input[:, 2]
        sx = input[:,3]
        sy = input[:,4]
        r = torch.sqrt(((x - sx) ** 2 + (y - sy) ** 2) +1e-8)  # Distance from source
        r = r + 1e-5

        farfield_waves_x = []
        farfield_waves_y = []
        for i in range(self.neurons):

            r_hat_x = (x - sx) / r
            r_hat_y = (y - sy) / r
            phi_hat_x = -1.0 * r_hat_y
            phi_hat_y = r_hat_x
            # phi = torch.atan2(y-mu_quake[1],x - mu_quake[0])

            M0_dot_input1 = (t + 1 + self.offset[i]) - r / alpha_
            M0_dot_input2 = (t + 1 + self.offset[i]) - r / beta_

            M_dot1 = self.M0[i] / (self.T[i] ** 2) * (M0_dot_input1 - 3.0 * self.T[i] / 2.0) * torch.exp(
                -(M0_dot_input1 - 3.0 * self.T[i] / 2.0) ** 2 / self.T[i] ** 2)
            M_dot2 = self.M0[i] / (self.T[i] ** 2) * (M0_dot_input2 - 3.0 * self.T[i] / 2.0) * torch.exp(
                -(M0_dot_input2 - 3.0 * self.T[i] / 2.0) ** 2 / self.T[i] ** 2)

            A_FP_x = torch.sin(self.theta_1[i]+1e-5) * torch.cos(self.theta_3[i]+1e-5) * r_hat_x
            A_FS_x = -torch.cos(self.theta_2[i]+1e-5) * torch.sin(self.theta_4[i]+1e-5) * phi_hat_x

            A_FP_y = torch.sin(self.theta_1[i]+1e-5) * torch.cos(self.theta_3[i]+1e-5) * r_hat_y
            A_FS_y = -torch.cos(self.theta_2[i]+1e-5) * torch.sin(self.theta_4[i]+1e-5) * phi_hat_y

            far_field_x = (1.0 / (4.0 * torch.pi * alpha_ ** 3)) * A_FP_x * (1.0 / r) * M_dot1 + (
                    1.0 / (4.0 * torch.pi * beta_ ** 3)) * A_FS_x * (1.0 / r) * M_dot2

            far_field_y = (1.0 / (4.0 * torch.pi * alpha_ ** 3)) * A_FP_y * (1.0 / r) * M_dot1 + (
                    1.0 / (4.0 * torch.pi * beta_ ** 3)) * A_FS_y * (1.0 / r) * M_dot2

            print(torch.max(torch.max(torch.abs(far_field_x)), torch.max(torch.max(torch.abs(far_field_y)))))

            far_field_x = far_field_x / (torch.max(torch.max(torch.abs(far_field_x)))+1e-5)
            far_field_y = far_field_y / (torch.max(torch.max(torch.abs(far_field_y)))+1e-5)


            farfield_waves_x.append(far_field_x)
            farfield_waves_y.append(far_field_y)

        # Stack results for each neuron
        combined_output_x = torch.stack(farfield_waves_x, dim=-1)
        combined_output_y = torch.stack(farfield_waves_y, dim=-1)

        # Concatenate results for u_x and u_y
        combined_output = torch.cat((combined_output_x, combined_output_y), dim=-1)

        return combined_output


        #return torch.stack(farfield_waves_x,farfield_waves_y, dim=-1)

    def plot_farfield_activation(self, sx=0.0, sy=0.0, t_range=(0, 1), x_range=(-1, 1), y_range=(-1, 1)):
        x_vals = torch.linspace(x_range[0], x_range[1], 100)
        y_vals = torch.linspace(y_range[0], y_range[1], 100)
        t_vals = torch.linspace(t_range[0], t_range[1], 3)
        X, Y = torch.meshgrid(x_vals, y_vals)

        alpha_ = torch.tensor(np.sqrt((20.0 + 2.0 * 30.0) / 100.0))
        beta_ = torch.tensor(np.sqrt(30.0 / 100.0))

        for i, (T, M0, theta_1, theta_2, theta_3, theta_4,offset) in enumerate(zip(
                self.T.detach().cpu(),
                self.M0.detach().cpu(),
                self.theta_1.detach().cpu(),
                self.theta_2.detach().cpu(),
                self.theta_3.detach().cpu(),
                self.theta_4.detach().cpu(),
                self.offset.detach().cpu())):

            for t in t_vals:
                r = torch.sqrt(((X - sx) ** 2 + (Y - sy) ** 2) + 1e-8)  # Distance from source
                r = r + 1e-15
                r_hat_x = (X - sx) / r
                r_hat_y = (Y - sy) / r
                phi_hat_x = -1.0 * r_hat_y

                M0_dot_input1 = (t + 1 + offset) - r / alpha_
                M0_dot_input2 = (t + 1 + offset) - r / beta_

                M_dot1 = M0 / (T ** 2) * (M0_dot_input1 - 3.0 * T / 2.0) * torch.exp(
                    -(M0_dot_input1 - 3.0 * T / 2.0) ** 2 / T ** 2)
                M_dot2 = M0 / (T ** 2) * (M0_dot_input2 - 3.0 * T / 2.0) * torch.exp(
                    -(M0_dot_input2 - 3.0 * T / 2.0) ** 2 / T ** 2)

                A_FP_x = torch.sin(theta_1) * torch.cos(theta_3) * r_hat_x
                A_FS_x = -torch.cos(theta_2) * torch.sin(theta_4) * phi_hat_x

                far_field_x = (1.0 / (4.0 * torch.pi * alpha_ ** 3)) * A_FP_x * (1.0 / r) * M_dot1 + (
                        1.0 / (4.0 * torch.pi * beta_ ** 3)) * A_FS_x * (1.0 / r) * M_dot2
                far_field_vals = (far_field_x / torch.max(torch.abs(far_field_x))).detach().cpu().numpy()

                plt.imshow(far_field_vals, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                           origin='lower', aspect='auto')
                plt.colorbar(label='Far Field Value')
                plt.title(f'Neuron {i} at time {t:.2f}')
                plt.xlabel('X')
                plt.ylabel('Y')

                # Log the image to wandb
                wandb.log({f"FarField_plot_neuron_{i}_time_{t:.2f}": wandb.Image(plt)})

                # Clear the plot for the next image
                plt.clf()

    def log_parameters_to_wandb(self):
        # Extracting parameters
        T_values = self.T.cpu().detach().numpy()
        M0_values = self.M0.cpu().detach().numpy()
        theta_1_values = self.theta_1.cpu().detach().numpy()
        theta_2_values = self.theta_2.cpu().detach().numpy()
        theta_3_values = self.theta_3.cpu().detach().numpy()
        theta_4_values = self.theta_4.cpu().detach().numpy()

        # Neuron numbers for x-axis
        neurons = list(range(1, len(T_values) + 1))

        # Log each parameter value per neuron
        for i, neuron in enumerate(neurons):
            wandb.log({f"T_neuron_{neuron}": T_values[i],
                       f"M0_neuron_{neuron}": M0_values[i],
                       f"theta_1_neuron_{neuron}": theta_1_values[i],
                       f"theta_2_neuron_{neuron}": theta_2_values[i],
                       f"theta_3_neuron_{neuron}": theta_3_values[i],
                       f"theta_4_neuron_{neuron}": theta_4_values[i]})
            
class Advanced_Bessel_Activation(nn.Module):
    def __init__(self, neurons):
        super(Advanced_Bessel_Activation, self).__init__()
        # Initialize parameters for each neuron
        self.neurons = neurons
        self.A = nn.Parameter(torch.randn(self.neurons))
        self.k = nn.Parameter(torch.randn(self.neurons)) 
        self.omega = nn.Parameter(torch.randn(self.neurons))
        self.alpha = nn.Parameter(torch.randn(self.neurons))
        #self.beta = nn.Parameter(torch.randn(self.neurons))
        self.beta = 0.0
        self.gamma = nn.Parameter(torch.randn(self.neurons)) 
        self.kappa = nn.Parameter(torch.randn(self.neurons))
        self.alpha_gaussian = nn.Parameter(torch.randn(self.neurons))

        self.lambda_0 = 0.08152
        self.p1 = -0.2622096091
        self.q = 0.5701109315
        self.p_tilda1 = -0.3945665468
        self.p0 = 0.5948274174
        self.p2 = -0.04932081835
        self.p_tilda0 = 0.4544933991
        self.p_tilda2 = 0.03277620113
        self.eps = 1e-5  # Small constant to avoid numerical issues

    def j0_approximation(self,x):
        factor1 = 1.0 / (((1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.25) * (1.0 + self.q * (x ** 2)))
        cosine_term = (self.p0 + self.p1 * (x ** 2) + self.p2 * (1.0 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) * torch.cos(x)
        sine_term = (self.p_tilda0 + self.p_tilda1 * (x ** 2)) * ((1 + (self.lambda_0 ** 4) * (x ** 2)) ** 0.5) + self.p_tilda2 * (
                x ** 2) * torch.sin(x) / x
        result = factor1 * (cosine_term + sine_term)

        return result
    def forward(self, input):
        print("A = ",self.A)
        print("k = ", self.k)
        print("omega = ", self.omega)
        print("alpha = ", self.alpha)
        print("gamma = ", self.gamma)
        print("kappa = ", self.kappa)
        print("alpha_gaussian = ", self.alpha_gaussian)
        #self.plot_advanced_bessel_function()

        t = input[:, 0]
        x = input[:, 1]
        y = input[:, 2]
        R = torch.sqrt(x ** 2 + y ** 2 + self.eps)
        #sx = input[:,3]
        #sy = input[:,4]


        bessel_waves_x = []
        bessel_waves_y = []
        for i in range(self.neurons):

            gaussian_envelope = torch.exp(-torch.sqrt(self.alpha_gaussian[i]**2 + self.eps) * (self.k[i] * torch.sqrt(torch.sqrt(self.gamma[i]**2 +self.eps) * (x ** 2) + torch.sqrt(self.kappa[i]**2 +self.eps) * (y ** 2) + self.eps) - self.omega[i] * t) ** 2)
            bessel_value_x = gaussian_envelope * torch.cos(torch.atan2(y + self.eps, x + self.eps) - self.alpha[i] + (self.beta * torch.atan2(y + self.eps, x + self.eps) ** 2)) * self.A[i] * torch.exp(-t) * self.j0_approximation(self.k[i] * R - self.omega[i] * t + self.eps)
            bessel_value_y = gaussian_envelope * torch.sin(torch.atan2(y + self.eps, x + self.eps) - self.alpha[i] + (self.beta * torch.atan2(y + self.eps, x + self.eps) ** 2)) * self.A[i] * torch.exp(-t) * self.j0_approximation(self.k[i] * R - self.omega[i] * t + self.eps)
            bessel_waves_x.append(bessel_value_x)
            bessel_waves_y.append(bessel_value_y)

        # Stack results for each neuron
        combined_output_x = torch.stack(bessel_waves_x, dim=-1)
        combined_output_y = torch.stack(bessel_waves_y, dim=-1)

        # Concatenate results for u_x and u_y
        combined_output = torch.cat((combined_output_x, combined_output_y), dim=-1)
        print("combined output inside activation",combined_output.shape,combined_output)

        return combined_output




        #return torch.stack(farfield_waves_x,farfield_waves_y, dim=-1)

    def plot_advanced_bessel_function(self, sx=0.0, sy=0.0, t_range=(0, 1), x_range=(-1, 1), y_range=(-1, 1)):
        x_vals = torch.linspace(x_range[0], x_range[1], 100)
        y_vals = torch.linspace(y_range[0], y_range[1], 100)
        t_vals = torch.linspace(t_range[0], t_range[1], 3)
        X, Y = torch.meshgrid(x_vals, y_vals)

        self.k = nn.Parameter(torch.randn(self.neurons))
        self.omega = nn.Parameter(torch.randn(self.neurons))
        self.alpha = nn.Parameter(torch.randn(self.neurons))
        self.beta = nn.Parameter(torch.randn(self.neurons))
        self.gamma = nn.Parameter(torch.randn(self.neurons))
        self.kappa = nn.Parameter(torch.randn(self.neurons))
        self.alpha_gaussian = nn.Parameter(torch.randn(self.neurons))

        for i, (k, omega,alpha,beta,gamma,kappa,alpha_gaussian) in enumerate(zip(
                                            self.k.detach().cpu().numpy(),
                                            self.omega.detach().cpu().numpy(),
                                            self.alpha.detach().cpu().numpy(),
                                            self.beta.detach().cpu().numpy(),
                                            self.gamma.detach().cpu().numpy(),
                                            self.kappa.detach().cpu().numpy(),
                                            self.alpha_gaussian.detach().cpu().numpy(),)):
            R = torch.sqrt(X ** 2 + Y ** 2 + self.eps)
            for t in t_vals:
                gaussian_envelope = torch.exp(-alpha_gaussian * (
                            k * torch.sqrt(gamma * X ** 2 + kappa * Y ** 2 + self.eps) -
                            omega * t) ** 2)
                bessel_value_x = gaussian_envelope * torch.cos(
                    torch.atan2(Y + self.eps, X + self.eps) - alpha + (
                                beta * torch.atan2(Y + self.eps, X + self.eps) ** 2)) * (
                                             self.A * torch.exp(-t)) * self.j0_approximation(
                    k * R - omega * t + self.eps)
                bessel_value_y = gaussian_envelope * torch.sin(
                    torch.atan2(Y + self.eps, X + self.eps) - alpha + (
                                beta * torch.atan2(Y + self.eps, X + self.eps) ** 2)) * (
                                             self.A * torch.exp(-t)) * self.j0_approximation(
                    k * R - omega * t + self.eps)

                plt.imshow(bessel_value_x, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                           origin='lower', aspect='auto')
                plt.colorbar(label='Bessel Value')
                plt.title(f'Neuron {i} at time {t:.2f}')
                plt.xlabel('X')
                plt.ylabel('Y')

                # Log the image to wandb
                wandb.log({f"Advanced_bessel_activation_plot_neuron_{i}_time_{t:.2f} X": wandb.Image(plt)})

                # Clear the plot for the next image
                plt.clf()
                plt.imshow(bessel_value_y, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                           origin='lower', aspect='auto')
                plt.colorbar(label='Bessel Value')
                plt.title(f'Neuron {i} at time {t:.2f}')
                plt.xlabel('X')
                plt.ylabel('Y')

                # Log the image to wandb
                wandb.log({f"Advanced_bessel_activation_plot_neuron_{i}_time_{t:.2f} Y": wandb.Image(plt)})

                # Clear the plot for the next image
                plt.clf()




class NeuralNet_increasing(NeuralNet):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed, activation):
        super().__init__(input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed, activation)
        self.hidden_layers = nn.ModuleList([nn.Linear(64, 64),nn.Linear(64, 64),nn.Linear(64, 96),nn.Linear(96, 128),nn.Linear(128, 256)])
        self.output_layer = nn.Linear(256, self.output_dimension)




class Pinns:

    def __init__(self, n_collocation_points,wandb_on,config):
        self.config = config
        if 'accoustic' in config:
            if config['accoustic']['accoustic_on'] == 'True':
                output_dimension=1
            else:
                output_dimension = 2
        else:
            output_dimension=2
        if config['Network']['activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif config['Network']['activation'] == 'sin':
            self.activation = SinActivation()
        elif config['Network']['activation'] == 'wavelet':
            self.activation = WaveletActivation(int(config['Network']['n_neurons']))
        elif config['Network']['activation'] == 'wavelet2':
            self.activation = WaveletActivation_2(int(config['Network']['n_neurons']))
        else:
            print("unknown activation function", config['Network']['activation'])
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        if config['Network']['nn_type'] == 'SIREN':
            #self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 #regularization_param, regularization_exp, retrain_seed, activation, n_1
            self.approximate_solution = SIREN_NeuralNet(input_dimension=3, output_dimension=output_dimension,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation,
                                                  n_1 = int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'BESSEL_FCN':
            self.approximate_solution = Advanced_Bessel_and_FCN_NeuralNet(input_dimension=3,output_dimension=output_dimension,
                                                                          n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                                          neurons=int(config['Network']['n_neurons']),
                                                                          regularization_param=0.,
                                                                          regularization_exp=2.,
                                                                          retrain_seed=3, activation=self.activation,
                                                                          n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'PLANE_WAVE_FCN':
            self.approximate_solution = Planewave_and_FCN_NeuralNet(input_dimension=3,output_dimension=output_dimension,
                                                                    n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'MORLET_WAVELET_FCN':
            self.approximate_solution = MorletWavelet_and_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'NONLINEAR_WAVELET_FCN':
            self.approximate_solution = Nonlinear_Wavelet_and_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FIXED_BESSEL_FCN':
            self.approximate_solution = Fixed_Advanced_Bessel_and_FCN_NeuralNet(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_PRECEEDING_DUAL_WAVELET':
            self.approximate_solution = FCN_Preceding_Dual_Wavelet(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_DIRECT_DUAL_WAVELET':
            self.approximate_solution = FCN_Direct_Dual_Wavelet(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_DIRECT_SINGULAR_WAVELET':
            self.approximate_solution = FCN_Direct_Singular_Wavelet(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_PLANEWAVE':
            self.approximate_solution = FCN_Amplitude_Planewave(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_WAVELET':
            self.approximate_solution = FCN_Amplitude_Wavelet(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET':
            self.approximate_solution = FCN_all_params_Wavelet_Modulation(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE':
            self.approximate_solution = FCN_all_params_Planewave_Modulation(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
            self.approximate_solution =  FCN_all_params_Wavelet_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']),
                                                                    n_hidden_layers_after=int(config['Network']['n_hidden_layers_after']),
                                                                    n_neurons_after=int(config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_SIREN_FCN':
            self.approximate_solution =  FCN_all_params_SIREN_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']),
                                                                    n_hidden_layers_after=int(config['Network']['n_hidden_layers_after']),
                                                                    n_neurons_after=int(config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_GAUSSIAN_FCN':
            self.approximate_solution =  FCN_all_params_Wavelet_gaussian_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']),
                                                                    n_hidden_layers_after=int(config['Network']['n_hidden_layers_after']),
                                                                    n_neurons_after=int(config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE_FCN':
            self.approximate_solution =  FCN_all_params_Planewave_FCN(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1']),
                                                                    n_hidden_layers_after=int(config['Network']['n_hidden_layers_after']),
                                                                    n_neurons_after=int(config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_ADVANCED_BESSEL':
            self.approximate_solution =  FCN_all_params_AdvancedBessel_Modulation(input_dimension=3, output_dimension=output_dimension,
                                                                    n_hidden_layers=int(
                                                                        config['Network']['n_hidden_layers']),
                                                                    neurons=int(config['Network']['n_neurons']),
                                                                    regularization_param=0.,
                                                                    regularization_exp=2.,
                                                                    retrain_seed=3, activation=self.activation,
                                                                    n_1=int(config['Network']['n_1'])
                                                                    )
        elif config['Network']['nn_type'] == 'PINN_REFERENCE_SINGLE_SOURCE_CONSTANT_sin':
            SinActivation()
            self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=output_dimension,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation)
        elif config['Network']['nn_type'] == 'PINN_REFERENCE_SINGLE_SOURCE_CONSTANT_tanh':
            self.activation = nn.Tanh()
            self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=output_dimension,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation)
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN_V1':

            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            density = float(config['parameters']['rho_solid'])
            density = torch.tensor(density)
            density = density.to(device)
            self.approximate_solution = FCN_all_params_Wavelet_FCN_v1(input_dimension=3, output_dimension=output_dimension,
                                                                   n_hidden_layers=int(
                                                                       config['Network']['n_hidden_layers']),
                                                                   neurons=int(config['Network']['n_neurons']),
                                                                   regularization_param=0.,
                                                                   regularization_exp=2.,
                                                                   retrain_seed=3, activation=self.activation,
                                                                   n_1=int(config['Network']['n_1']),
                                                                   n_hidden_layers_after=int(
                                                                       config['Network']['n_hidden_layers_after']),
                                                                   n_neurons_after=int(
                                                                       config['Network']['n_neurons_after']),density=density,lambda_m=lambda_mixture,mu_m=mu_mixture)
        elif config['Network']['nn_type'] == 'Lame_FCN_ALL_PARAMS_WAVELET_FCN':
            self.domain_extrema = torch.tensor(
                [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                 [float(config['domain']['xmin']), float(config['domain']['xmax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                 ])  # Space dimension
            self.approximate_solution = FCN_all_params_Wavelet_FCN(input_dimension=5, output_dimension=output_dimension,
                                                                   n_hidden_layers=int(
                                                                       config['Network']['n_hidden_layers']),
                                                                   neurons=int(config['Network']['n_neurons']),
                                                                   regularization_param=0.,
                                                                   regularization_exp=2.,
                                                                   retrain_seed=3, activation=self.activation,
                                                                   n_1=int(config['Network']['n_1']),
                                                                   n_hidden_layers_after=int(
                                                                       config['Network']['n_hidden_layers_after']),
                                                                   n_neurons_after=int(
                                                                       config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'Lame_FCN_ALL_PARAMS_WAVELET_FCN_V1':
            self.domain_extrema = torch.tensor(
                [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                 [float(config['domain']['xmin']), float(config['domain']['xmax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                 ])  # Space dimension
            self.approximate_solution = Lame_FCN_all_params_Wavelet_FCN_v1(input_dimension=5, output_dimension=output_dimension,
                                                                   n_hidden_layers=int(
                                                                       config['Network']['n_hidden_layers']),
                                                                   neurons=int(config['Network']['n_neurons']),
                                                                   regularization_param=0.,
                                                                   regularization_exp=2.,
                                                                   retrain_seed=3, activation=self.activation,
                                                                   n_1=int(config['Network']['n_1']),
                                                                   n_hidden_layers_after=int(
                                                                       config['Network']['n_hidden_layers_after']),
                                                                   n_neurons_after=int(
                                                                       config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'Lame_FCN':
            self.domain_extrema = torch.tensor(
                [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                 [float(config['domain']['xmin']), float(config['domain']['xmax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                 [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                 ])  # Space dimension
            self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=output_dimension,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation)
        else:
            print("correct")
            self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=output_dimension,
                                                      n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                      neurons=int(config['Network']['n_neurons']),
                                                      regularization_param=0.,
                                                      regularization_exp=2.,
                                                      retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)
        if 'accoustic' in config:
            if config['accoustic']['accoustic_on'] == 'True':
                if config['Network']['conditioning'] == 'False':
                    self.source_function = initial_conditions.initial_condition_simple_gaussian_accoustic
                    #self.source_function = initial_conditions.initial_condition_explosion
                    print("using accoustic initial condition")
                else:
                    self.source_function = initial_conditions.initial_condition_simple_gaussian_accoustic_conditioned
                    #self.source_function = initial_conditions.initial_condition_explosion
                    print("using conditioned accoustic initial condition")
            else:
                if config['initial_condition']['source_function'] == 'explosion':
                    self.source_function = initial_conditions.initial_condition_explosion
                elif config['initial_condition']['source_function'] == 'explosion_conditioned':
                    self.source_function = initial_conditions.initial_condition_explosion_conditioned
                elif config['initial_condition']['source_function'] == 'explosion_two_sources':
                    self.source_function = initial_conditions.initial_condition_explosion_two_sources
                elif config['initial_condition']['source_function'] == 'gaussian':
                    self.source_function = initial_conditions.initial_condition_gaussian
                elif config['initial_condition']['source_function'] == 'donut':
                    self.source_function = initial_conditions.initial_condition_donut
                else:
                    print(config['initial_condition']['source_function'], 'explosion',
                          'explosion' == config['initial_condition']['source_function'])
                    raise Exception(
                        "Source function {} is not implemented".format(config['initial_condition']['source_function']))
        else:
            if config['initial_condition']['source_function'] == 'explosion':
                self.source_function = initial_conditions.initial_condition_explosion
            elif config['initial_condition']['source_function'] == 'explosion_conditioned':
                self.source_function = initial_conditions.initial_condition_explosion_conditioned
            elif config['initial_condition']['source_function'] == 'explosion_two_sources':
                self.source_function = initial_conditions.initial_condition_explosion_two_sources
            elif config['initial_condition']['source_function'] == 'gaussian':
                self.source_function = initial_conditions.initial_condition_gaussian
            elif config['initial_condition']['source_function'] == 'donut':
                self.source_function = initial_conditions.initial_condition_donut
            else:
                print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
                raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']
        if 'accoustic' in config:
            self.velocity = config['accoustic']['velocity']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']

    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        #print("start")
        #u0x = u0x * 0.0
        #tmp = u0y
        #u0y = u0x
        #u0x = tmp



        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def pinn_model_eval_accoustic(self, input_tensor):
        U_perturbation = self.approximate_solution(input_tensor)
        u0 = self.source_function(input_tensor,self.sigma_quake)

        U = torch.zeros_like(U_perturbation)

        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0 * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)

        return U

    def convert(self, tens):
        #assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.config['accoustic']['accoustic_on'] =='True':
            if self.parameter_model == 'mixture':
                velocity_mixture = mixture_model.generate_acoustic_mixture()
                velocity = mixture_model.compute_acoustic_param(input_s[:, 1], input_s[:, 2],velocity_mixture)
                plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=velocity.detach().numpy())
                plt.colorbar()
                plt.show()
                print("minimum value",torch.min(velocity))
            elif self.parameter_model == 'constant':
                velocity = torch.full((self.n_collocation_points,), float(1.0))

            elif self.parameter_model == 'layered':
                raise NotImplementedError

            elif self.parameter_model == 'layered_sine':
                velocity, _ = mixture_model.extract_values_from_2d_maps(input_s)

            self.velocity = velocity.to(device)

        else:
            if self.parameter_model == 'mixture':
                mu_mixture = mixture_model.generate_mixture()
                lambda_mixture = mixture_model.generate_mixture()
                #mu_mixture = mu_mixture.to(device)
                #lambda_mixture = lambda_mixture.to(device)
                lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
                mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)

            elif self.parameter_model == 'constant':
                lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
                mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))


            elif self.parameter_model == 'layered':
                lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)

            elif self.parameter_model == 'layered_sine':
                lambda_m, mu_m = mixture_model.extract_values_from_2d_maps(input_s)
                plt.scatter(input_s[:, 1], input_s[:, 2], c=lambda_m)
                plt.show()
            elif self.parameter_model == 'layered_sine_simpler':
                lambda_m, mu_m = mixture_model.extract_values_from_2d_maps(input_s)
                plt.scatter(input_s[:, 1], input_s[:, 2], c=lambda_m)
                plt.show()

            elif self.parameter_model == 'layered_sine_simpler_3':
                lambda_m, mu_m = mixture_model.extract_values_from_2d_maps(input_s)
                #plt.scatter(input_s[:, 1], input_s[:, 2], c=lambda_m)
                #plt.show()
            else:
                raise Exception("{} not implemented".format(self.parameter_model))
            self.lambda_m = lambda_m.to(device)
            self.mu_m = mu_m.to(device)

            if (self.config['Network']['nn_type'] == 'Lame_FCN') or self.config['Network']['nn_type'] == 'Lame_FCN_ALL_PARAMS_WAVELET_FCN' or self.config['Network']['nn_type'] =='Lame_FCN_ALL_PARAMS_WAVELET_FCN_V1':
                input_s[:, 3] = self.lambda_m
                input_s[:, 4] = self.mu_m

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)
        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]
        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )
        loss_solid = torch.mean(abs(residual_solid) ** 2)
        return loss_solid

    def compute_accoustic_loss(self,input_s):

        U = self.pinn_model_eval_accoustic(input_s)
        gradient = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        dt = gradient[:, 0]
        dx = gradient[:, 1]
        dy = gradient[:, 2]

        dt_gradient = torch.autograd.grad(dt.sum(), input_s, create_graph=True)[0]
        dt2 = dt_gradient[:, 0]

        dx_gradient = torch.autograd.grad(dx.sum(), input_s, create_graph=True)[0]
        dx2 = dx_gradient[:, 1]

        dy_gradient = torch.autograd.grad(dy.sum(), input_s, create_graph=True)[0]
        dy2 = dy_gradient[:, 2]

        laplacian = dx2 + dy2

        residual = laplacian - (1.0 / (torch.pow(self.velocity, 2))) * dt2

        residual = residual.reshape(-1, )

        loss = torch.mean(abs(residual) ** 2)

        return loss

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_loss_accoustic(self, inp_train_s):
        loss = self.compute_accoustic_loss(inp_train_s)
        loss = torch.log10(loss)
        if self.wandb_on:
            wandb.log({"Solid loss": loss.item()})
        return loss

    def compute_test_loss(self, test_input, mu_quake,sigma_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(inputs)[:, 0]
            uy = self.pinn_model_eval(inputs)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0
        count_after_decimal_x = str(mu_quake[0])[::-1].find('.')
        count_after_decimal_y = str(mu_quake[1])[::-1].find('.')
        if count_after_decimal_x > 3:
            mu_quake_str_x = str(round(float(mu_quake[0]), 3))
        else:
            mu_quake_str_x = str(float(mu_quake[0]))

        if count_after_decimal_y > 3:
            mu_quake_str_y = str(round(float(mu_quake[1]), 3))
        else:
            mu_quake_str_y = str(float(mu_quake[1]))

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_x.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_y.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_u.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def compute_test_loss_accoustic(self, test_input, mu_quake, sigma_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256

        res_list = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:
            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            u = self.pinn_model_eval_accoustic(inputs)
            u_out = u.detach()

            np_u_out = u_out.cpu().numpy()

            B_u = np.reshape(np_u_out, (-1, int(np.sqrt(np_u_out.shape[0]))))
            res_list.append(B_u)

        res_u = np.dstack(res_list)
        res = np.rollaxis(res_u, -1)

        #s = 5 * np.mean(np.abs(res))

        f, axarr = plt.subplots(1, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0
        count_after_decimal_x = str(mu_quake[0])[::-1].find('.')
        count_after_decimal_y = str(mu_quake[1])[::-1].find('.')
        if count_after_decimal_x > 3:
            mu_quake_str_x = str(round(float(mu_quake[0]), 3))
        else:
            mu_quake_str_x = str(float(mu_quake[0]))

        if count_after_decimal_y > 3:
            mu_quake_str_y = str(round(float(mu_quake[1]), 3))
        else:
            mu_quake_str_y = str(float(mu_quake[1]))
        if self.parameter_model == 'mixture':
            file_name = 'pre_computed_test_devito/accoustic/mixture_slower/mu=[{}, {}]/sigma={}/res.pkl'.format(mu_quake_str_x, mu_quake_str_y, sigma_quake)
        elif self.parameter_model =='layered_sine':
            file_name = 'pre_computed_test_devito/accoustic/layered_sine_slower/mu=[{}, {}]/sigma={}/res.pkl'.format(mu_quake_str_x, mu_quake_str_y, sigma_quake)

        else:
            file_name = 'pre_computed_test_devito/accoustic/{}/mu=[{}, {}]/sigma={}/res.pkl'.format(
                self.parameter_model, mu_quake_str_x, mu_quake_str_y, sigma_quake)


        with open(file_name, 'rb') as f_:
            res_list_devito = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito[0])))

        for h in range(0, len(res_list)):
            diffu = ((res[h, :, :]) - (res_list_devito[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu ** 2))

            if h == 0 or h == int(len(res_list) / 4) or h == int(len(res_list) / 3) or h == int(
                    len(res_list) / 2) or h == len(res_list) - 2:

                im1u = axarr[0].imshow(res[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[1].imshow(res_list_devito[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2].imshow(diffu, 'bwr', vmin=-s, vmax=s)

                axarr[0].set_title("PINN", fontsize=25, pad=20)
                axarr[1].set_title("Devito", fontsize=25, pad=20)
                axarr[2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3u, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})

        test_loss = test_loss / len(res_list)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y

        if (self.config['Network']['nn_type'] == 'Lame_FCN') or self.config['Network']['nn_type'] == 'Lame_FCN_ALL_PARAMS_WAVELET_FCN' or self.config['Network']['nn_type'] =='Lame_FCN_ALL_PARAMS_WAVELET_FCN_V1':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            lambda_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], mu_mixture)
            inputs[:, 3] = lambda_m
            inputs[:, 4] = mu_m
            plt.scatter(inputs[:, 1], inputs[:, 2],c=inputs[:, 3])
            plt.show()
            plt.scatter(inputs[:, 1], inputs[:, 2], c=inputs[:, 4])
            plt.show()


        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)

        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True' and epoch%10 == 0:
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake,self.config['parameters']['sigma_quake'])
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

    def fit_accoustic(self,num_epochs,optimizer,verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)

        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True' and epoch % 10 == 0:
                test_input = self.get_test_loss_input(256, 0.1, test_mu_quake)
                test_loss = self.compute_test_loss_accoustic(test_input, test_mu_quake, self.config['parameters']['sigma_quake'])
                wandb.log({"Test Loss": np.log10(test_loss.item())})

            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss_accoustic(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Hard_data_PINNs(Pinns):


    def pinn_model_eval(self, input_tensor, fd_contributions):
        U_perturbation = self.approximate_solution(input_tensor)

        # Combine with initial conditions and NN output
        #u0x, u0y = self.source_function(input_tensor, self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
       # U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 * input_tensor[:, 0] / self.t1) ** 2 + u0x * torch.exp(
            #-0.5 * (1.5 * input_tensor[:, 0] / self.t1) ** 2) + fd_contributions[:, 0]
       # U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 * input_tensor[:, 0] / self.t1) ** 2 + u0y * torch.exp(
            #-0.5 * (1.5 * input_tensor[:, 0] / self.t1) ** 2) + fd_contributions[:, 1]

        U[:, 0] =  fd_contributions[:, 0]
        U[:, 1] =  fd_contributions[:, 1]
        return U

    def compute_loss(self, inp_train_s,fd_contributions):
        loss_solid = self.compute_solid_loss(inp_train_s, fd_contributions)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_solid_loss(self, input_s, fd_contributions):
        U = self.pinn_model_eval(input_s, fd_contributions)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)
        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]
        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )
        loss_solid = torch.mean(abs(residual_solid) ** 2)
        return loss_solid

    def compute_test_loss(self, test_input, mu_quake,sigma_quake,fd_contributions):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(inputs,fd_contributions)[:, 0]
            uy = self.pinn_model_eval(inputs,fd_contributions)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0
        count_after_decimal_x = str(mu_quake[0])[::-1].find('.')
        count_after_decimal_y = str(mu_quake[1])[::-1].find('.')
        if count_after_decimal_x > 3:
            mu_quake_str_x = str(round(float(mu_quake[0]), 3))
        else:
            mu_quake_str_x = str(float(mu_quake[0]))

        if count_after_decimal_y > 3:
            mu_quake_str_y = str(round(float(mu_quake[1]), 3))
        else:
            mu_quake_str_y = str(float(mu_quake[1]))

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_x.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_y.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/sigma={}/res_u.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y,sigma_quake)

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def fit(self, num_epochs, optimizer, verbose=False):

        file_name_y = 'pre_computed_test_devito/constant/mu=[0.0, 0.0]/sigma=0.1/res_y.pkl'
        file_name_x = 'pre_computed_test_devito/constant/mu=[0.0, 0.0]/sigma=0.1/res_x.pkl'

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)

        devito_array_x = torch.FloatTensor(res_list_devito_x)
        devito_array_y = torch.FloatTensor(res_list_devito_y)

        original_size = 101 * 256 * 256  # Total number of points in the full grid
        reduction_factor = 20000
        new_sample_size = original_size // reduction_factor
        print("New sample size:", new_sample_size)  # Output should give you an idea of the number of points to sample

        # Flatten the FD data for easier sampling
        fd_values_x_flat = devito_array_x.view(-1)  # Flatten the tensor
        fd_values_y_flat = devito_array_y.view(-1)  # Flatten the tensor

        # Generate random indices
        indices = torch.randperm(fd_values_x_flat.size(0))[:new_sample_size]
        print("indices= ",indices)

        # Sample the FD values and corresponding grid coordinates
        sampled_fd_values_x = fd_values_x_flat[indices]
        sampled_fd_values_y = fd_values_y_flat[indices]
        sampled_fd_values = torch.stack((sampled_fd_values_x, sampled_fd_values_y),
                                        dim=1)  # Combine x and y into a single tensor

        # Sample the corresponding grid coordinates
        # Compute indices for the full grid
        t_indices = indices // (256 * 256)  # Time indices
        x_indices = (indices % (256 * 256)) // 256  # Spatial x indices
        y_indices = (indices % (256 * 256)) % 256  # Spatial y indices

        # Creating vectors for time and spatial coordinates
        time_points = torch.linspace(0, 1, 101)  # Generates 101 points from 0 to 1 for time dimension
        x = torch.linspace(-1, 1, 256)  # Generates 256 points from -1 to 1 for the x spatial dimension
        y = torch.linspace(-1, 1, 256)  # Generates 256 points from -1 to 1 for the y spatial dimension

        # Sample grid coordinates using the indices
        sampled_t = time_points[t_indices]
        sampled_x = x[x_indices]
        sampled_y = y[y_indices]

        #sampled_fd_values = sampled_fd_values.to(device)
        #sampled_t = sampled_t.to(device)
        #sampled_x = sampled_x.to(device)
        ##sampled_y = sampled_y.to(device)


        # Reshape for broadcasting
        sampled_t = sampled_t.view(1, -1)
        sampled_x = sampled_x.view(1, -1)
        sampled_y = sampled_y.view(1, -1)


        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        print(inp_train_s.shape,inp_train_s)

        # Calculate distances
        distances = (inp_train_s[:, 0:1] - sampled_t) ** 2 + (inp_train_s[:, 1:2] - sampled_x) ** 2 + (inp_train_s[:, 2:3] - sampled_y) ** 2

        # Compute Gaussian activation
        activations = torch.exp(-50 * distances)  # Shape: [num_input_points, new_sample_size]

        # Weighted sum of FD values based on activations
        fd_contributions = torch.matmul(activations, sampled_fd_values)  # Result shape: [num_sobol_points, 2]

        # Normalize contributions if necessary
        norm_activations = activations.sum(dim=1, keepdim=True)  # Sum across all FD points
        fd_contributions /= norm_activations

        fd_contributions = fd_contributions.to(device)


        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)

        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True' and epoch%10 == 0:
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake,self.config['parameters']['sigma_quake'],fd_contributions)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s, fd_contributions)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history




class Pinns_with_helper:

    def __init__(self, n_collocation_points,wandb_on,config):
        if config['Network']['activation'] == 'tanh':
             self.activation = nn.Tanh()

        else:
            print("unknown activation function", config['Network'].activation)
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)


        if config['initial_condition']['source_function'] == 'explosion':
            self.source_function = initial_conditions.initial_condition_explosion
        elif config['initial_condition']['source_function'] == 'explosion_conditioned':
            self.source_function = initial_conditions.initial_condition_explosion_conditioned
        elif config['initial_condition']['source_function'] == 'explosion_two_sources':
            self.source_function = initial_conditions.initial_condition_explosion_two_sources
        elif config['initial_condition']['source_function'] == 'gaussian':
            self.source_function = initial_conditions.initial_condition_gaussian
        elif config['initial_condition']['source_function'] == 'donut':
            self.source_function = initial_conditions.initial_condition_donut
        else:
            print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
            raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']


    def pinn_model_eval(self, helper_output,input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        #print(U_perturbation[:, 0].shape,input_tensor[:, 0].shape,helper_output[:,0].shape)
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2) + helper_output[:,0]
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2) + helper_output[:,1]
        return U

    def convert(self, tens):
        #assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, helper_output,input_s):

        U = self.pinn_model_eval(helper_output,input_s)

        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)


        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]

        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]

        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]

        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]


        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)

        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]

        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]


        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)

        residual_solid = self.rho_solid * dt2_combined - div_stress

        residual_solid = residual_solid.reshape(-1, )


        loss_solid = torch.mean(abs(residual_solid) ** 2)


        return loss_solid

    def get_solid_residual(self, helper_output,input_s):
        U = self.pinn_model_eval(helper_output,input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, helper_output,inp_train_s):
        loss_solid = self.compute_solid_loss(helper_output,inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_test_loss(self, helper_network,test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)
            helper_output = helper_network.pinn_model_eval(inputs)

            ux = self.pinn_model_eval(helper_output,inputs)[:, 0]
            uy = self.pinn_model_eval(helper_output,inputs)[:, 1]

            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        #print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y


        return inputs

    def fit(self, helper_network,num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        helper_network.approximate_solution = helper_network.approximate_solution.to(device)
        helper_output = helper_network.pinn_model_eval(training_set_s)
        helper_output = helper_output.to(device)
        #print("training set shape = ",inp_train_s.shape)



        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)
        test_input = self.get_test_loss_input(256, 0.1, test_mu_quake)
        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':

                test_loss = self.compute_test_loss(helper_network,test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(helper_output,plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(helper_output,training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Pinns_memory_recuced(Pinns):

    def __init__(self, n_collocation_points,wandb_on,config):
        if config['Network']['activation'] == 'tanh':
             self.activation = nn.Tanh()

        else:
            print("unknown activation function", config['Network'].activation)
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)


        if config['initial_condition']['source_function'] == 'explosion':
            self.source_function = initial_conditions.initial_condition_explosion
        elif config['initial_condition']['source_function'] == 'explosion_conditioned':
            self.source_function = initial_conditions.initial_condition_explosion_conditioned
        elif config['initial_condition']['source_function'] == 'explosion_two_sources':
            self.source_function = initial_conditions.initial_condition_explosion_two_sources
        elif config['initial_condition']['source_function'] == 'gaussian':
            self.source_function = initial_conditions.initial_condition_gaussian
        elif config['initial_condition']['source_function'] == 'donut':
            self.source_function = initial_conditions.initial_condition_donut
        else:
            print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
            raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']


    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def convert(self, tens):
        #assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, input_s):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("1: ",t,r,a)
        U = self.pinn_model_eval(input_s)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("2: ", t, r, a)
        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("3: ", t, r, a)
        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("4: ", t, r, a)
        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("5: ", t, r, a)
        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("6: ", t, r, a)

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)
        del gradient_x
        del gradient_y
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("7: ", t, r, a)
        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("8: ", t, r, a)
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("9: ", t, r, a)
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        del eps
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("10: ", t, r, a)
        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("11: ", t, r, a)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("12: ", t, r, a)
        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("13: ", t, r, a)
        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("14: ", t, r, a)
        del stress_tensor_11
        del stress_tensor_00
        del stress_tensor_off_diag
        del off_diag_grad
        torch.cuda.empty_cache()
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("15: ", t, r, a)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("16: ", t, r, a)
        del div_stress
        del dt2_combined
        torch.cuda.empty_cache()
        residual_solid = residual_solid.reshape(-1, )
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("17: ", t, r, a)

        loss_solid = torch.mean(abs(residual_solid) ** 2)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("18: ", t, r, a)

        del residual_solid
        torch.cuda.empty_cache()
        print("del del")
        return loss_solid

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_test_loss(self, test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(inputs)[:, 0]
            uy = self.pinn_model_eval(inputs)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y


        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)


        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Global_NSources_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        # Call the parent class's initializer
        super().__init__(n_collocation_points, wandb_on, config)

        # Modify the existing member variables
        self.domain_extrema = torch.tensor([
            [float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
            [float(config['domain']['xmin']), float(config['domain']['xmax'])],
            [float(config['domain']['ymin']), float(config['domain']['ymax'])],
            [-1.0, 1.0],
            [-1.0, 1.0]
        ])  # Space dimension

        self.approximate_solution = NeuralNet(
            input_dimension=5,
            output_dimension=2,
            n_hidden_layers=int(config['Network']['n_hidden_layers']),
            neurons=int(config['Network']['n_neurons']),
            regularization_param=0.,
            regularization_exp=2.,
            retrain_seed=3,
            activation=self.activation
        )


        self.n_sources = int(config["initial_condition"]["n_sources"])
        print(self.n_sources)

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        n_different_sources = int(self.n_sources)

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, n_different_sources)
        source_y = np.random.uniform(-0.5, 0.5, n_different_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(n_different_sources), int(self.n_collocation_points / n_different_sources))
        #print(input_s[:, 3].shape, torch.tensor(source_x[source_idx], dtype=torch.float32).shape)
        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)


        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape,sys.getsizeof((lambda_m)),sys.getsizeof(mu_m))
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self,numpoints_sqrt,time,mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

class Relative_Distance_NSources_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)


        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                                               ,[-1.0, 1.0],[-1.0, 1.0]
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        self.n_sources = int(config["initial_condition"]["n_sources"])
        #TODO: Hard coded
        self.source_function = initial_conditions.initial_condition_explosion_conditioned_relative

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, self.n_sources)
        source_y = np.random.uniform(-0.5, 0.5, self.n_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(self.n_sources), int(self.n_collocation_points /self.n_sources))
        input_s[:, 1] = input_s[:, 1] - torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 2] = input_s[:, 2] -  torch.tensor(source_y[source_idx], dtype=torch.float32)

        #Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        #Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)
        #print(input_s)

        return input_s

    def get_test_loss_input(self,numpoints_sqrt,time,mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]
        inputs[:, 1] = inputs[:, 1] - inputs[:, 3]
        inputs[:, 2] = inputs[:, 2] - inputs[:, 4]

        return inputs

class Relative_Distance_FullDomain_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                                               ,[float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],[float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)

        # TODO: Hard coded
        self.source_function = initial_conditions.initial_condition_explosion_conditioned_relative

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)
        input_s[:, 1] = input_s[:, 1] - input_s[:,3]
        input_s[:, 2] = input_s[:, 2] - input_s[:,4]
        #TODO: renormalize

        #print(input_s)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]
        inputs[:, 1] = inputs[:, 1] - inputs[:, 3]
        inputs[:, 2] = inputs[:, 2] - inputs[:, 4]

        return inputs

class Global_NSources_Conditioned_Lame_Pinns(Pinns):
    def __init__(self, n_collocation_points, wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                            [-1.0, 1.0], [-1.0, 1.0],[-1.0, 1.0],[-1.0, 1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=7, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3,activation=self.activation)

        self.n_sources = int(config["initial_condition"]["n_sources"])

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, self.n_sources)
        source_y = np.random.uniform(-0.5, 0.5, self.n_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(self.n_sources), int(self.n_collocation_points / self.n_sources))
        input_s[:, 1] = input_s[:, 1]
        input_s[:, 2] = input_s[:, 2]
        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = input_s[:,5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:,6] * eps[0, 0]
        stress_tensor_off_diag = 2.0 * input_s[:,6] * eps[0, 1]
        stress_tensor_11 = input_s[:,5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:,6] * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        loss_solid = torch.mean(abs(residual_solid) ** 2)

        return loss_solid

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = input_s[:, 5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:, 6] * eps[0, 0]
        stress_tensor_off_diag = 2.0 * input_s[:, 6] * eps[0, 1]
        stress_tensor_11 = input_s[:, 5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:, 6] * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        training_set_s[:, 5] = self.lambda_m
        training_set_s[:, 6] = self.mu_m
        #print(training_set_s)
        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([-0.25, 0.25])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})

            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            lambda_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(inputs[:,1], inputs[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        inputs[:, 5] = lambda_m
        inputs[:, 6] = mu_m

        return inputs

class Global_FullDomain_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)
        if 'accoustic' in config:
            if config['accoustic']['accoustic_on'] == 'True':
                output_dimension=1
            else:
                output_dimension = 2
        else:
            output_dimension=2

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        if config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
            print("correct")
            self.approximate_solution = FCN_all_params_Wavelet_FCN(input_dimension=5, output_dimension=output_dimension,
                                                                   n_hidden_layers=int(
                                                                       config['Network']['n_hidden_layers']),
                                                                   neurons=int(config['Network']['n_neurons']),
                                                                   regularization_param=0.,
                                                                   regularization_exp=2.,
                                                                   retrain_seed=3, activation=self.activation,
                                                                   n_1=int(config['Network']['n_1']),
                                                                   n_hidden_layers_after=int(
                                                                       config['Network']['n_hidden_layers_after']),
                                                                   n_neurons_after=int(
                                                                       config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE_FCN':
            self.approximate_solution = FCN_all_params_Planewave_FCN(input_dimension=5, output_dimension=output_dimension,
                                                                     n_hidden_layers=int(
                                                                         config['Network']['n_hidden_layers']),
                                                                     neurons=int(config['Network']['n_neurons']),
                                                                     regularization_param=0.,
                                                                     regularization_exp=2.,
                                                                     retrain_seed=3, activation=self.activation,
                                                                     n_1=int(config['Network']['n_1']),
                                                                     n_hidden_layers_after=int(
                                                                         config['Network']['n_hidden_layers_after']),
                                                                     n_neurons_after=int(
                                                                         config['Network']['n_neurons_after']))
        else:
            self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=output_dimension,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation)

        self.n_epochs = int(config['optimizer']['n_epochs'])
        self.curriculum = config['Network']['curriculum']

    def assemble_datasets(self,epoch):
        input_s1 = self.add_solid_points(epoch)

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1


    def add_solid_points(self,current_epoch):

        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)

        #Curriculum
        if self.config['accoustic']['accoustic_on'] =='True':
            if self.parameter_model == 'mixture':
                velocity_mixture = mixture_model.generate_acoustic_mixture()
                velocity = mixture_model.compute_acoustic_param(input_s[:, 1], input_s[:, 2],velocity_mixture)
                plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=velocity.detach().numpy())
                plt.colorbar()
                plt.show()
                print("minimum value",torch.min(velocity))
            elif self.parameter_model == 'constant':
                velocity = torch.full((self.n_collocation_points,), float(1.0))

            elif self.parameter_model == 'layered':
                raise NotImplementedError
            self.velocity = velocity.to(device)
        else:
            if self.parameter_model == 'mixture':
                mu_mixture = mixture_model.generate_mixture()
                lambda_mixture = mixture_model.generate_mixture()
                #mu_mixture = mu_mixture.to(device)
                #lambda_mixture = lambda_mixture.to(device)
                lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
                mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            elif self.parameter_model == 'constant':
                lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
                mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
                #print(lambda_m.shape, mu_m.shape)
            elif self.parameter_model == 'layered':
                lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            else:
                raise Exception("{} not implemented".format(self.parameter_model))
            self.lambda_m = lambda_m.to(device)
            self.mu_m = mu_m.to(device)

        if self.curriculum == 'True':
            min_time = self.domain_extrema[0, 0]
            original_min = self.domain_extrema[0, 0]
            original_max = self.domain_extrema[0, 1]

            max_time = min(1.0,((current_epoch + 1) / (self.n_epochs-50)) * original_max)

            # Rescale the time domain in-place
            input_s[:, 0] = min_time + (max_time - min_time) * (input_s[:, 0] - original_min) / (
                    original_max - original_min)
            #print("min max time: ",min_time,max_time,torch.isnan(input_s).any(),torch.isinf(input_s).any())
            #input_s = input_s.to(device)
            #input_s.requires_grad = True

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        if self.curriculum == 'False':
            training_set_s = self.assemble_datasets(0)
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

        self.approximate_solution = self.approximate_solution.to(device)

        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            print(epoch,"/",num_epochs)
            if self.curriculum == 'True':
                training_set_s = self.assemble_datasets(epoch)
                inp_train_s = next(iter(training_set_s))[0]
                training_set_s = inp_train_s.to(device)
                training_set_s.requires_grad = True
            if self.test_on == 'True' and epoch%10 == 0:
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()

                # Gradient clipping
                #nn.utils.clip_grad_norm_(self.approximate_solution.parameters(), 1.0)
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

    def fit_accoustic(self,num_epochs,optimizer,verbose=False):
        if self.curriculum == 'False':
            training_set_s = self.assemble_datasets(0)
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

        self.approximate_solution = self.approximate_solution.to(device)

        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            print(epoch, "/", num_epochs)
            if self.test_on == 'True' and epoch % 10 == 0:
                test_input = self.get_test_loss_input(256, 0.1, test_mu_quake)
                test_loss = self.compute_test_loss_accoustic(test_input, test_mu_quake, self.config['parameters']['sigma_quake'])
                wandb.log({"Test Loss": np.log10(test_loss.item())})

            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss_accoustic(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history


class Global_FullDomain_Conditioned_Pinns_batched(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        if config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
            print("correct")
            self.approximate_solution = FCN_all_params_Wavelet_FCN(input_dimension=5, output_dimension=2,
                                                                   n_hidden_layers=int(
                                                                       config['Network']['n_hidden_layers']),
                                                                   neurons=int(config['Network']['n_neurons']),
                                                                   regularization_param=0.,
                                                                   regularization_exp=2.,
                                                                   retrain_seed=3, activation=self.activation,
                                                                   n_1=int(config['Network']['n_1']),
                                                                   n_hidden_layers_after=int(
                                                                       config['Network']['n_hidden_layers_after']),
                                                                   n_neurons_after=int(
                                                                       config['Network']['n_neurons_after']))
        elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE_FCN':
            self.approximate_solution = FCN_all_params_Planewave_FCN(input_dimension=5, output_dimension=2,
                                                                     n_hidden_layers=int(
                                                                         config['Network']['n_hidden_layers']),
                                                                     neurons=int(config['Network']['n_neurons']),
                                                                     regularization_param=0.,
                                                                     regularization_exp=2.,
                                                                     retrain_seed=3, activation=self.activation,
                                                                     n_1=int(config['Network']['n_1']),
                                                                     n_hidden_layers_after=int(
                                                                         config['Network']['n_hidden_layers_after']),
                                                                     n_neurons_after=int(
                                                                         config['Network']['n_neurons_after']))
        else:
            self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3, activation=self.activation)

        self.n_epochs = int(config['optimizer']['n_epochs'])
        self.curriculum = config['Network']['curriculum']

    def assemble_datasets(self, epoch):
        input_s1, lambda_m, mu_m = self.add_solid_points(epoch)
        # Create a combined dataset
        combined_dataset = torch.utils.data.TensorDataset(input_s1, lambda_m, mu_m)
        # Return a DataLoader that iterates over combined datasets
        return DataLoader(combined_dataset, batch_size=int(self.n_collocation_points / 4), shuffle=True, num_workers=0)

    def add_solid_points(self,current_epoch):

        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)

        #Curriculum

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        if self.curriculum == 'True':
            min_time = self.domain_extrema[0, 0]
            original_min = self.domain_extrema[0, 0]
            original_max = self.domain_extrema[0, 1]

            max_time = min(1.0,((current_epoch + 1) / (self.n_epochs-50)) * original_max)

            # Rescale the time domain in-place
            input_s[:, 0] = min_time + (max_time - min_time) * (input_s[:, 0] - original_min) / (
                    original_max - original_min)
            #print("min max time: ",min_time,max_time,torch.isnan(input_s).any(),torch.isinf(input_s).any())
            #input_s = input_s.to(device)
            #input_s.requires_grad = True
        lambda_m = lambda_m.to(device)
        mu_m = mu_m.to(device)
        return input_s,lambda_m,mu_m

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def compute_solid_loss(self, input_s, batch_lambda, batch_mu):

        U = self.pinn_model_eval(input_s)

        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]


        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])),
                                 torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)

        stress_tensor_00 = batch_lambda * (eps[0, 0] + eps[1, 1]) + 2.0 * batch_mu * eps[0, 0]

        stress_tensor_off_diag = 2.0 * batch_mu * eps[0, 1]

        stress_tensor_11 = batch_lambda * (eps[0, 0] + eps[1, 1]) + 2.0 * batch_mu * eps[1, 1]

        # stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
        # torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        # print(div_stress.shape)

        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]

        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)

        residual_solid = self.rho_solid * dt2_combined - div_stress

        residual_solid = residual_solid.reshape(-1, )

        loss_solid = torch.mean(abs(residual_solid) ** 2)

        return loss_solid

    def compute_loss(self, inp_train_s, batch_lambda, batch_mu):
        loss_solid = self.compute_solid_loss(inp_train_s, batch_lambda, batch_mu)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):

        self.approximate_solution = self.approximate_solution.to(device)

        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            training_loader = self.assemble_datasets(epoch)

            for batch_inp, batch_lambda, batch_mu in training_loader:
                batch_inp = batch_inp.to(device)
                batch_lambda = batch_lambda.to(device)
                batch_mu = batch_mu.to(device)
                batch_inp.requires_grad = True

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(batch_inp, batch_lambda, batch_mu)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.approximate_solution.parameters(), 1.0)
                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)
                optimizer.step(closure=closure)
            print(epoch, "/", num_epochs)
            if self.test_on == 'True' and epoch % 10 == 0:
                test_input = self.get_test_loss_input(256, 0.1, test_mu_quake)
                test_loss = self.compute_test_loss(test_input, test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
        print('Final Loss: ', history[-1])

        return history

class PlaneWave_Pinns(Global_FullDomain_Conditioned_Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.approximate_solution = PlaneWave_NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3)

class Simple_PlaneWave_Pinns(PlaneWave_Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.approximate_solution = SimplePlaneWave_NeuralNet(input_dimension=5, output_dimension=2,
                                                        neurons=int(config['Network']['n_neurons']),
                                                        regularization_param=0.,
                                                        regularization_exp=2.,
                                                        retrain_seed=3)

    def add_solid_points(self, current_epoch):

        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        input_s[:, 3] = 0.0
        input_s[:, 4] = 0.0

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            # mu_mixture = mu_mixture.to(device)
            # lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            # print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:, 1], input_s[:, 2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        if self.curriculum == 'True':
            min_time = self.domain_extrema[0, 0]
            original_min = self.domain_extrema[0, 0]
            original_max = self.domain_extrema[0, 1]

            max_time = min(1.0, ((current_epoch + 1) / (self.n_epochs - 50)) * original_max)

            # Rescale the time domain in-place
            input_s[:, 0] = min_time + (max_time - min_time) * (input_s[:, 0] - original_min) / (
                    original_max - original_min)
            # print("min max time: ",min_time,max_time,torch.isnan(input_s).any(),torch.isinf(input_s).any())
            # input_s = input_s.to(device)
            # input_s.requires_grad = True

        return input_s

    def fit(self, num_epochs, optimizer, verbose=False):
        if self.curriculum == 'False':
            training_set_s = self.assemble_datasets(0)
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

        self.approximate_solution = self.approximate_solution.to(device)

        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            print(epoch,"/",num_epochs)
            if self.curriculum == 'True':
                training_set_s = self.assemble_datasets(epoch)
                inp_train_s = next(iter(training_set_s))[0]
                training_set_s = inp_train_s.to(device)
                training_set_s.requires_grad = True
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                print("test_input =",test_input)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                #print("omega", self.approximate_solution.activation.omega.grad, "\n")
                #print("alpha", self.approximate_solution.activation.alpha.grad, "\n")
                #print("k", self.approximate_solution.activation.k.grad, "\n")
                #print("A", self.approximate_solution.activation.A.grad, "\n")
                #print("l", self.approximate_solution.activation.l.grad, "\n")
                #print("v", self.approximate_solution.activation.v.grad, "\n")
                #print("theta_1", self.approximate_solution.activation.theta_1.grad, "\n")
                #print("theta_2", self.approximate_solution.activation.theta_2.grad, "\n")
                #print("theta_3", self.approximate_solution.activation.theta_3.grad, "\n")
                #print("theta_4", self.approximate_solution.activation.theta_4.grad, "\n")
                #print("M0", self.approximate_solution.activation.M0.grad, "\n")
                #print("T", self.approximate_solution.activation.T.grad, "\n")
                #print("offsett", self.approximate_solution.activation.offset.grad, "\n")
                #print("output:",self.approximate_solution.output_layer.weight)

                # Gradient clipping
                #nn.utils.clip_grad_norm_(self.approximate_solution.parameters(), 1.0)
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Simple_Bessel_Pinns(Simple_PlaneWave_Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.approximate_solution = SimpleBesselWave_NeuralNet(input_dimension=5, output_dimension=2,
                                                        neurons=int(config['Network']['n_neurons']),
                                                        regularization_param=0.,
                                                        regularization_exp=2.,
                                                        retrain_seed=3)

class Simple_FarField_Pinns(Simple_Bessel_Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.approximate_solution = SimpleFarFieldWave_NeuralNet(input_dimension=5, output_dimension=2,
                                                               neurons=int(config['Network']['n_neurons']),
                                                               regularization_param=0.,
                                                               regularization_exp=2.,
                                                               retrain_seed=3)

class Advanced_Bessel_Pinns(Simple_Bessel_Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.approximate_solution = Advanced_Bessel_NeuralNet(input_dimension=5, output_dimension=2,
                                                               neurons=int(config['Network']['n_neurons']),
                                                               regularization_param=0.,
                                                               regularization_exp=2.,
                                                               retrain_seed=3)




class Global_FullDomain_Conditioned_Pinns_reduced_computation(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

        self.n_epochs = int(config['optimizer']['n_epochs'])
        self.curriculum = config['Network']['curriculum']

    def assemble_datasets(self,epoch):
        input_s1 = self.add_solid_points(epoch)

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def pinn_model_eval(self, t,x,y,sx,sy):
        #assert t.requires_grad, "t does not require grad after stacking and insied the eval!"

        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(torch.stack((t,x,y,sx,sy),dim=1))
        #print(torch.stack((t,x,y,sx,sy),dim=1))
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(torch.stack((t,x,y,sx,sy),dim=1),self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *t / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * t/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *t / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * t/self.t1)**2)
        return U


    def add_solid_points(self,current_epoch):

        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)

        #Curriculum

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        if self.curriculum == 'True':
            min_time = self.domain_extrema[0, 0]
            original_min = self.domain_extrema[0, 0]
            original_max = self.domain_extrema[0, 1]

            max_time = ((current_epoch + 1) / self.n_epochs) * original_max

            # Rescale the time domain in-place
            input_s[:, 0] = min_time + (max_time - min_time) * (input_s[:, 0] - original_min) / (
                    original_max - original_min)
            print("min max time: ",min_time,max_time,torch.isnan(input_s).any(),torch.isinf(input_s).any())
            #input_s = input_s.to(device)
            #input_s.requires_grad = True

        return input_s

    def compute_solid_loss(self, t,x,y,sx,sy):
        #assert t.requires_grad, "t does not require grad after stacking and insied the eval!"
        #print(t.requires_grad)

        U = self.pinn_model_eval(t,x,y,sx,sy)
        #assert t.requires_grad, "t does not require grad after stacking and insied the eval!"
        #print(t.requires_grad)

        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        dux_dt = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), t, create_graph=True)[0]
        dux_dx = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), x, create_graph=True)[0]
        dux_dy = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), y, create_graph=True)[0]

        duy_dt = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), t, create_graph=True)[0]
        duy_dx = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), x, create_graph=True)[0]
        duy_dy = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), y, create_graph=True)[0]

        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dux_dt.sum(), t, create_graph=True)[0]

        dt2_y = torch.autograd.grad(duy_dt.sum(), t, create_graph=True)[0]

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * dux_dx, dux_dy + duy_dx)), torch.stack((dux_dy + duy_dx, 2.0 * duy_dy))), dim=1)


        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]

        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]

        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]

        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad_x = torch.autograd.grad(stress_tensor_off_diag.sum(), x, create_graph=True)[0]
        off_diag_grad_y = torch.autograd.grad(stress_tensor_off_diag.sum(), y, create_graph=True)[0]


        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, t.size(0), dtype=torch.float32, device=t.device)
        print(div_stress.shape)

        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), x, create_graph=True)[0] + \
                           off_diag_grad_y

        div_stress[1, :] = off_diag_grad_x + \
                           torch.autograd.grad(stress_tensor_11.sum(), y, create_graph=True)[0]


        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)

        residual_solid = self.rho_solid * dt2_combined - div_stress

        residual_solid = residual_solid.reshape(-1, )


        loss_solid = torch.mean(abs(residual_solid) ** 2)


        return loss_solid

    def compute_loss(self, t,x,y,sx,sy):
        loss_solid = self.compute_solid_loss(t,x,y,sx,sy)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def compute_test_loss(self, test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(t=inputs[:, 0],x=inputs[:,1],y=inputs[:,2],sx=inputs[:,3],sy=inputs[:,4])[:, 0]
            uy = self.pinn_model_eval(t=inputs[:, 0],x=inputs[:,1],y=inputs[:,2],sx=inputs[:,3],sy=inputs[:,4])[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0
        count_after_decimal_x = str(mu_quake[0])[::-1].find('.')
        count_after_decimal_y = str(mu_quake[1])[::-1].find('.')
        if count_after_decimal_x > 3:
            mu_quake_str_x = str(round(float(mu_quake[0]), 3))
        else:
            mu_quake_str_x = str(float(mu_quake[0]))

        if count_after_decimal_y > 3:
            mu_quake_str_y = str(round(float(mu_quake[1]), 3))
        else:
            mu_quake_str_y = str(float(mu_quake[1]))

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def fit(self, num_epochs, optimizer, verbose=False):
        if self.curriculum == 'False':
            training_set_s = self.assemble_datasets(0)
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            t = training_set_s[:, 0]
            x = training_set_s[:, 1]
            y = training_set_s[:, 2]
            sx = training_set_s[:,3]
            sy = training_set_s[:,4]
            t.requires_grad = True
            x.requires_grad = True
            y.requires_grad = True
            #training_set_s = torch.stack((t, x, y, sx, sy), dim=1)
            #assert training_set_s[:, 0].requires_grad, "t does not require grad after stacking!"

        # training_set_s[:, 0].requires_grad = True
            #training_set_s[:, 1].requires_grad = True
            #training_set_s[:, 2].requires_grad = True
            #training_set_s.requires_grad = True

        self.approximate_solution = self.approximate_solution.to(device)




        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            print(epoch)
            if self.curriculum == 'True':
                training_set_s = self.assemble_datasets(epoch)
                inp_train_s = next(iter(training_set_s))[0]
                training_set_s = inp_train_s.to(device)
                t = training_set_s[:, 0]
                x = training_set_s[:, 1]
                y = training_set_s[:, 2]
                sx = training_set_s[:, 3]
                sy = training_set_s[:, 4]
                t.requires_grad = True
                x.requires_grad = True
                y.requires_grad = True
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(t,x,y,sx,sy)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Global_FullDomain_Conditioned_Pinns_Scramble_Resample(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0],scramble=True)

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)

        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            training_set_s = self.assemble_datasets()
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Global_FullDomain_Conditioned_Pinns_FD_Ansatz(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs






