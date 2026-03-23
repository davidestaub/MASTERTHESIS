import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import os
import numpy as np
import devito
from devito import *
import matplotlib.pyplot as plt
import torch
from devito.builtins import initialize_function
import FD_devito
import time as timer


import sys

# Check if at least one command-line argument is provided
if len(sys.argv) > 1:
    num_sources = sys.argv[1]  # Take the first argument after the script name
else:
    print("Please provide a command-line argument for the number of sources.")

plt.rcParams['text.usetex'] = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

resolution = 128
res_lists_devito_u = []
for numpoints_sqrt in [resolution]:
    for n_different_sources in [num_sources]:



        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        dt = 1.095e-3


        enlarge_factor = 3
        pinn_domain_extent_x = [-1.0, 1.0]
        pinn_domain_extent_y = [-1.0, 1.0]

        FD_prep_0 = timer.time()

        pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
        pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

        devito_length_x = enlarge_factor * pinn_length_x
        devito_length_y = enlarge_factor * pinn_length_y
        extent = (devito_length_x, devito_length_y)
        devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]
        shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
        y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
        grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
        spacing = (extent[0] / shape[0], extent[0] / shape[0])
        X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                           np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))

        FD_prep_1 = timer.time()

        # Generate mixtures for mu and lambda
        mu_mixture = FD_devito.generate_mixture().numpy()
        lambda_mixture = FD_devito.generate_mixture().numpy()
        lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
        mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)

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
        lambda_large[:, y_offset + numpoints_sqrt:] = lambda_large[:, y_offset + (numpoints_sqrt - 1)][:,
                                                      np.newaxis]

        lambda_solid = lambda_large

        Xp, Yp = np.meshgrid(
            np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
            np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))

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

        rho_solid = 100.0

        div_stress_ux = (
                                    lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy
        div_stress_uy = (
                                    lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx
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
        stencil_x = Eq(ux_devito.forward, solve(pde_x, ux_devito.forward))
        stencil_y = Eq(uy_devito.forward, solve(pde_y, uy_devito.forward))
        op = Operator([stencil_x] + [stencil_y] + bc)

        n = 102
        dt = 1.096e-3
        time_list = np.linspace(0, 1, n).tolist()
        # Determine the grid size
        #grid_side = int(np.sqrt(n_different_sources))  # Calculate side length of the square grid
        grid_side = n_different_sources  # Calculate side length of the square grid
        #actual_n_sources = grid_side ** 2  # Actual number of sources used

        # Generate grid points
        x_grid = np.linspace(-1, -1, grid_side)
        y_grid = np.linspace(-1, 1, grid_side)

        t0 = timer.time()
        for mu_quake_x in np.linspace(-0.3, 0.3, grid_side):
            for mu_quake_y in np.linspace(-0.3, 0.3, grid_side):

                mu_quake_devito = [devito_center[0] + mu_quake_x, devito_center[1] + mu_quake_y]

                u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
                    sigma=float(0.1), mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
                    dx=spacing[0], dy=spacing[1])

                ux_devito.data[0] = u0x_devito
                uy_devito.data[0] = u0y_devito
                ux_devito.data[1] = u0x_devito
                uy_devito.data[1] = u0y_devito

                index = 0
                FD_prep_1 = timer.time()

                res_list_devito_x = []
                res_list_devito_y = []
                res_list_devito_u = []


                for i in time_list:
                    res_list_devito_x.append(np.transpose(ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
                    res_list_devito_y.append(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
                    res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,y_offset:y_offset + numpoints_sqrt]).copy() ** 2 + np.transpose(ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,y_offset:y_offset + numpoints_sqrt]).copy() ** 2))
                    op(time_M=10, dt=dt)
                res_lists_devito_u.append(res_list_devito_u)

        t1 = timer.time()

         # Parameters
        n_receivers = resolution-1
        receiver_x_positions = np.linspace(0, resolution-1, n_receivers).astype(int)  # Indices for the receivers along x-axis
        y_receiver_index = 0  # Assuming the y-line is at the middle of the grid

        # Initialize an array to store global maximum values for each receiver
        global_max_pressures = np.zeros(n_receivers)

        # Process each list of results
        for res_list_devito_u in res_lists_devito_u:
            # Initialize an array to store maximum values for each receiver from the current source
            max_pressures = np.zeros(n_receivers)

            # Loop over each receiver's x position
            for i, x_idx in enumerate(receiver_x_positions):
                # Extract the values at the receiver location over all time steps
                receiver_values = np.array([res_list_devito_u[t][x_idx,63] for t in range(102)])
                # Compute the maximum value over all time steps for this receiver
                max_pressures[i] = np.max(np.abs(receiver_values))

            # Update the global maximum pressures if current values are higher
            global_max_pressures = np.maximum(global_max_pressures, max_pressures)
            #print(global_max_pressures)
        I = np.sum(global_max_pressures)
        print(I)

        # Plotting
        #plt.figure(figsize=(10, 6))
        #plt.plot(receiver_x_positions, global_max_pressures, '-o', markersize=4, label='Global Max Pressure')
        #plt.title("Global Maximum Pressure at Different Receiver Locations")
        #plt.xlabel("Receiver Location X (index)")
        #plt.ylabel("Maximum Pressure [Pa]")
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        with open('output_devito_amplitude.txt', 'a') as file:
            # Create the string to be appended to the file
            line =f"{I},{n_different_sources},{t1 - t0}\n"
            #line = f"Took {t1 - t0} seconds for {n_different_sources} sources and {numpoints_sqrt} points\n"
            # Append the string to the file
            file.write(line)
