import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch.nn as nn
import wandb
import mixture_model
import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import os
import initial_conditions
import numpy as np
import torch
import FD_devito
from devito import *
import pickle
from Marmousi import plot_marmousi


def compute_test_loss(parameter_model,domain_extrema,lambda_solid_c,mu_solid_c, rho_solid, mu_quake,sigma_quake):
    # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
    numpoints_sqrt = 256
    dt=1.095e-3

    pinn_domain_extent_x = [domain_extrema[1][0], domain_extrema[1][1]]
    pinn_domain_extent_y = [domain_extrema[2][0], domain_extrema[2][1]]
    pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
    pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

    enlarge_factor = 3
    devito_length_x = enlarge_factor * pinn_length_x
    devito_length_y = enlarge_factor * pinn_length_y
    extent = (devito_length_x, devito_length_y)
    devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]

    mu_quake_devito = [devito_center[0] + mu_quake[0],
                       devito_center[1] + mu_quake[1]]

    # increase Devito domain to avoid Boundary issues (reflections)
    shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
    y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
    grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
    print("grid = ",grid)
    exit()
    spacing = (extent[0] / shape[0], extent[0] / shape[0])
    X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                       np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))
    print(numpoints_sqrt * spacing[0])

    if parameter_model == "mixture":
        # Generate mixtures for mu and lambda
        mu_mixture = FD_devito.generate_mixture().numpy()
        lambda_mixture = FD_devito.generate_mixture().numpy()
        lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
        mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)
    elif parameter_model == "constant":
        lambda_solid = np.full(X.shape, float(lambda_solid_c))
        mu_solid = np.full(X.shape, float(mu_solid_c))
    elif parameter_model == 'layered':
        lambda_solid,mu_solid = mixture_model.compute_lambda_mu_layers(torch.tensor(X),torch.tensor(Y),5)
        lambda_solid = lambda_solid.numpy()
        mu_solid = mu_solid.numpy()

    elif parameter_model == 'layered_sine':
        lambda_solid, mu_solid = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
    elif parameter_model == 'layered_sine_simpler':
        lambda_solid, mu_solid = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
    elif parameter_model =='layered_sine_simpler_3':
        lambda_solid, mu_solid = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))

    elif parameter_model == 'marmousi':
        vp = plot_marmousi.load_segy('Marmousi/vp_marmousi-ii.segy')  # P-wave velocity
        vs = plot_marmousi.load_segy('Marmousi/vs_marmousi-ii.segy')  # S-wave velocity
        density = plot_marmousi.load_segy('Marmousi/density_marmousi-ii.segy')  # Density

        # Calculate Lame parameters
        mu = density * vs ** 2  # Shear modulus or the second Lame parameter
        lambda_ = density * vp ** 2 - 2 * mu  # First Lame parameter

        # Plot results
        plot_marmousi.plot_data(mu, 'Second Lame Parameter (μ) - Shear Modulus')
        plot_marmousi.plot_data(lambda_, 'First Lame Parameter (λ)')

        # Example usage (you should replace these with actual values)
        #X, Y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))

        mu_solid, lambda_solid = plot_marmousi.get_params_at_coordinates(X, Y, mu, lambda_)

        #print("Mu Values at given Coordinates: ", mu_vals)
        #print("Lambda Values at given Coordinates: ", lambda_vals)
        #plot_marmousi.plot_data(mu_vals.T, 'Second Lame Parameter (μ) - Shear Modulus normalized')
        #plot_marmousi.plot_data(lambda_vals.T, 'First Lame Parameter (λ) normalized')

    else:
        raise Exception("model type {} not implemented".format(parameter_model))

    # Create a larger array (7x7)
    X_large, Y_large = numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor
    lambda_large = np.full((X_large, Y_large), 0.0)
    # Calculate the position to place the smaller array at the center of the larger array
    x_offset = (X_large - numpoints_sqrt) // 2
    y_offset = (Y_large - numpoints_sqrt) // 2

    # Place the smaller array in the center of the larger array
    lambda_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = lambda_solid
    lambda_large[0:x_offset, :] = lambda_large[x_offset, :]
    lambda_large[x_offset + numpoints_sqrt:, :] = lambda_large[x_offset + (numpoints_sqrt - 1), :]
    lambda_large[:, 0:y_offset] = lambda_large[:, y_offset][:, np.newaxis]
    lambda_large[:, y_offset + numpoints_sqrt:] = lambda_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]

    lambda_solid = lambda_large

    Xp, Yp = np.meshgrid(
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))
    plt.scatter(Xp, Yp, c=lambda_solid)
    plt.show()
    mu_large = np.full((X_large, Y_large), 0.0)
    mu_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = mu_solid
    mu_large[0:x_offset, :] = mu_large[x_offset, :]
    mu_large[x_offset + numpoints_sqrt:, :] = mu_large[x_offset + (numpoints_sqrt - 1), :]
    mu_large[:, 0:y_offset] = mu_large[:, y_offset][:, np.newaxis]
    mu_large[:, y_offset + numpoints_sqrt:] = mu_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]
    mu_solid = mu_large

    # Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
    ux_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)
    uy_devito = TimeFunction(name='uy_devito', grid=grid, space_order=4, time_order=2)

    lambda_ = Function(name='lambda_f', grid=grid, space_order=4)
    mu_ = Function(name='mu_f', grid=grid, space_order=4)
    initialize_function(lambda_, lambda_solid, nbl=0)
    initialize_function(mu_, mu_solid, nbl=0)
    # print(mu_.data, mu_.data.shape)
    # plt.scatter(Xp, Yp, c=lambda_.data)
    # plt.show()
    # plt.scatter(Xp, Yp, c=mu_.data)
    # plt.show()

    # The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
    u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
        sigma=sigma_quake, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
        dx=spacing[0], dy=spacing[1])
    #tmp = u0x_devito
    #u0x_devito = u0y_devito
    #u0y_devito = tmp

    # Initialize the VectorTimeFunction with the initial values
    # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
    # and because 2nd order time discretization is neeed
    ux_devito.data[0] = u0x_devito
    uy_devito.data[0] = u0y_devito
    ux_devito.data[1] = u0x_devito
    uy_devito.data[1] = u0y_devito

    # FD_devito.plot_field(ux_devito.data[0])
    # FD_devito.plot_field(uy_devito.data[0])
    # Plot from initial field look good
    # print(ux_devito.shape, uy_devito.shape)

    print("start stress calculation")
    # Divergence of stress tensor for ux
    div_stress_ux = (
                            lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy

    # Divergence of stress tensor for uy
    div_stress_uy = (
                            lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx

    print("stress calculated")
    # Elastic wave equation for ux and uy
    # print("model damp = ",model.damp,model.damp.data)

    pde_x = rho_solid * ux_devito.dt2 - div_stress_ux
    pde_y = rho_solid * uy_devito.dt2 - div_stress_uy

    # BOundary conditions:
    x, y = grid.dimensions
    t = grid.stepping_dim
    ny, nx = shape[0], shape[1]
    bc = [Eq(ux_devito[t + 1, x, 0], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(ux_devito[t + 1, 0, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(ux_devito[t + 1, x, 1], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(ux_devito[t + 1, 1, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 2, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 0], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(uy_devito[t + 1, 0, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 1], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(uy_devito[t + 1, 1, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 2, y], 0.)]

    # Formulating stencil to solve for u forward
    print(type(ux_devito.forward), type(ux_devito), type(pde_x), ux_devito)
    stencil_x = Eq(ux_devito.forward, solve(pde_x, ux_devito.forward))
    stencil_y = Eq(uy_devito.forward, solve(pde_y, uy_devito.forward))
    op = Operator([stencil_x] + [stencil_y] + bc)

    print("start")
    res_list_devito_x = []
    res_list_devito_y = []
    res_list_devito_u = []
    time_list = np.linspace(0, 1, 102).tolist()
    for i in time_list:
        res_list_devito_x.append(np.transpose(
            ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
        res_list_devito_y.append(np.transpose(
            uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
        res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                                      y_offset:y_offset + numpoints_sqrt]).copy() ** 2 + np.transpose(
                                                      ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                                      y_offset:y_offset + numpoints_sqrt]).copy() ** 2))

        op(time_M=10, dt=dt)
        if (i == 0) or (i> 0.245 and i < 0.255) or (i> 0.495 and i < 0.51) or (i> 0.74 and i < 0.75) or (i >0.981):
            plt.imshow(np.transpose(ux_devito.data[0][x_offset:x_offset + 201, y_offset:y_offset + 201]),cmap='bwr', vmin=-0.1,
                       vmax=0.1)
            plt.title("ux time={}".format(i))
            plt.show()
            plt.imshow(np.transpose(uy_devito.data[0][x_offset:x_offset + 201, y_offset:y_offset + 201]),cmap='bwr', vmin=-0.1,
                       vmax=0.1)
            plt.title("uy time={}".format(i))
            plt.show()
        else:
            print(i)

    path = "../pre_computed_test_devito/{}/tmp/mu={}/sigma={}/".format(parameter_model, mu_quake,sigma_quake)
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)

    file_name_x = '../pre_computed_test_devito/{}/tmp/mu={}/sigma={}/res_x.pkl'.format(parameter_model, mu_quake,sigma_quake)
    file_name_y = '../pre_computed_test_devito/{}/tmp/mu={}/sigma={}/res_y.pkl'.format(parameter_model, mu_quake,sigma_quake)
    file_name_u = '../pre_computed_test_devito/{}/tmp/mu={}/sigma={}/res_u.pkl'.format(parameter_model, mu_quake,sigma_quake)
    with open(file_name_x, 'wb+') as file:
        pickle.dump(res_list_devito_x, file)
    print(f'Object successfully saved to "{file_name_x}"')

    with open(file_name_y, 'wb+') as file:
        pickle.dump(res_list_devito_y, file)
    print(f'Object successfully saved to "{file_name_y}"')

    with open(file_name_u, 'wb+') as file:
        pickle.dump(res_list_devito_u, file)
    print(f'Object successfully saved to "{file_name_u}"')

    with open(file_name_x, 'rb') as f:
        res_list_devito_x = pickle.load(f)
    with open(file_name_y, 'rb') as f:
        res_list_devito_y = pickle.load(f)
    with open(file_name_u, 'rb') as f:
        res_list_devito_u = pickle.load(f)

def compute_test_loss_accoustic(parameter_model,domain_extrema,velocity, mu_quake,sigma_quake):
    # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
    numpoints_sqrt = 256
    dt = 1.1e-3

    pinn_domain_extent_x = [domain_extrema[1][0], domain_extrema[1][1]]
    pinn_domain_extent_y = [domain_extrema[2][0], domain_extrema[2][1]]
    pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
    pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

    enlarge_factor = 3
    devito_length_x = enlarge_factor * pinn_length_x
    devito_length_y = enlarge_factor * pinn_length_y
    extent = (devito_length_x, devito_length_y)
    devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]

    mu_quake_devito = [devito_center[0] + mu_quake[0],
                       devito_center[1] + mu_quake[1]]

    # increase Devito domain to avoid Boundary issues (reflections)
    shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
    y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
    grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
    spacing = (extent[0] / shape[0], extent[0] / shape[0])
    X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                       np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))
    print(numpoints_sqrt * spacing[0])

    if parameter_model == "mixture":
        # Generate mixtures for mu and lambda
        velocity_mixture = FD_devito.generate_acoustic_mixture_np()
        velocity = FD_devito.compute_param_np(X, Y, velocity_mixture)
    elif parameter_model == "constant":
        velocity = np.full(X.shape, velocity)
    elif parameter_model == 'layered':
        velocity, _ = mixture_model.compute_lambda_mu_layers_accoustic_slow(torch.tensor(X), torch.tensor(Y), 5)
        velocity = velocity.numpy()
    elif parameter_model == 'layered_sine':
        velocity, _ = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
    else:
        raise Exception("model type {} not implemented".format(parameter_model))

    # Create a larger array (7x7)
    X_large, Y_large = numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor
    velocity_large = np.full((X_large, Y_large), 0.0)
    # Calculate the position to place the smaller array at the center of the larger array
    x_offset = (X_large - numpoints_sqrt) // 2
    y_offset = (Y_large - numpoints_sqrt) // 2

    # Place the smaller array in the center of the larger array
    velocity_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = velocity
    velocity_large[0:x_offset, :] = velocity_large[x_offset, :]
    velocity_large[x_offset + numpoints_sqrt:, :] = velocity_large[x_offset + (numpoints_sqrt - 1), :]
    velocity_large[:, 0:y_offset] = velocity_large[:, y_offset][:, np.newaxis]
    velocity_large[:, y_offset + numpoints_sqrt:] = velocity_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]

    velocity = velocity_large

    Xp, Yp = np.meshgrid(
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))
    plt.scatter(Xp, Yp, c=velocity)
    plt.colorbar()
    plt.show()

    # Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
    u_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)

    velocity_ = Function(name='velocity_f', grid=grid, space_order=4)

    initialize_function(velocity_, velocity, nbl=0)

    #initial_condition_simple_gaussian
    # The initial condition
    u0_devito = FD_devito.initial_condition_simple_gaussian(
        sigma=sigma_quake, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
        dx=spacing[0], dy=spacing[1])


    # Initialize the VectorTimeFunction with the initial values
    # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
    # and because 2nd order time discretization is neeed
    u_devito.data[0] = u0_devito
    u_devito.data[1] = u0_devito

    print("start laplacian calculation")
    laplacian = u_devito.dx2 + u_devito.dy2
    print("laplacian calculated")

    #Accoustic wave equation
    pde = laplacian - (1.0/velocity_**2) * u_devito.dt2


    # BOundary conditions:
    x, y = grid.dimensions
    t = grid.stepping_dim
    ny, nx = shape[0], shape[1]
    bc = [Eq(u_devito[t + 1, x, 0], 0.)]
    bc += [Eq(u_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(u_devito[t + 1, 0, y], 0.)]
    bc += [Eq(u_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(u_devito[t + 1, x, 1], 0.)]
    bc += [Eq(u_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(u_devito[t + 1, 1, y], 0.)]
    bc += [Eq(u_devito[t + 1, nx - 2, y], 0.)]


    # Formulating stencil to solve for u forward
    print(type(u_devito.forward), type(u_devito), type(pde), u_devito)
    stencil_x = Eq(u_devito.forward, solve(pde, u_devito.forward))
    op = Operator([stencil_x]  + bc)

    print("start")
    res_list_devito = []
    time_list = np.linspace(0, 1, 101).tolist()

    for i in time_list:
        res_list_devito.append(np.transpose(
            u_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
        op(time_M=10, dt=dt)
        if i==0.0 or i == 0.1 or i ==0.2 or i ==0.3 or i ==0.4 or i==0.5 or i==0.6 or i==0.7 or i==0.8 or i==0.9 or i==1.0:
            plt.imshow(np.transpose(u_devito.data[0][x_offset:x_offset + 201, y_offset:y_offset + 201]),cmap='bwr', vmin=-0.1,
                       vmax=0.1)
            plt.title("ux time={}".format(i))
            plt.show()

    path = "../pre_computed_test_devito/accoustic/{}_slower/mu={}/sigma={}/".format(parameter_model, mu_quake, sigma_quake)
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)

    file_name = '../pre_computed_test_devito/accoustic/{}_slower/mu={}/sigma={}/res.pkl'.format(parameter_model, mu_quake,
                                                                                   sigma_quake)
    with open(file_name, 'wb+') as file:
        pickle.dump(res_list_devito, file)
    print(f'Object successfully saved to "{file_name}"')

    with open(file_name, 'rb') as f:
        res_list_devito = pickle.load(f)


domain_extrema = [[0.0, 1.0],[-1.0, 1.0], [-1.0, 1.0]]
compute_test_loss('layered_sine_simpler_3',domain_extrema,20.0,30.0,100.0,[0.0,0.0],0.1)
#compute_test_loss_accoustic('layered_sine',domain_extrema,1.0,[0.0,0.0],0.06)

