import torch
import numpy as np
import matplotlib.pyplot as plt
#The package you need to install
from devito import *
#My code for lame parameter model generation
import mixture_model
#My code with helper functions
import FD_devito

#Set spatial grid resolution
numpoints_sqrt = 512

#Set time step
dt = 1.096e-3

#Set Velocity model (constant,mixture,layered,layered_sine)
model_type = 'constant'

domain_length_x = 2
domain_length_y = 2
extent = (domain_length_x, domain_length_y)

mu_quake_devito = [1.0,1.0]
shape = (numpoints_sqrt, numpoints_sqrt)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
spacing = (extent[0] / shape[0], extent[0] / shape[0])
X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                   np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))


# These are parameters models I used for my thesis, you can find them in mixture_model.py
# Feel free to not use them or change them for your needs.
if model_type == "mixture":
    # Generate mixtures for mu and lambda
    velocity_mixture = FD_devito.generate_acoustic_mixture_np()
    velocity = FD_devito.compute_param_np(X, Y, velocity_mixture)
elif model_type == "constant":
    velocity = np.full(X.shape, 1.0)
elif model_type == 'layered':
    velocity, _ = mixture_model.compute_lambda_mu_layers_accoustic_slow(torch.tensor(X), torch.tensor(Y), 5)
    velocity = velocity.numpy()
elif model_type == 'layered_sine':
    velocity, _ = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
else:
    raise Exception("model type {} not implemented".format(model_type))

Xp, Yp = np.meshgrid(
    np.linspace(0, (numpoints_sqrt ) * spacing[0], numpoints_sqrt),
    np.linspace(0, (numpoints_sqrt) * spacing[1], numpoints_sqrt))

# Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
u_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)

velocity_ = Function(name='velocity_f', grid=grid, space_order=4)

initialize_function(velocity_, velocity, nbl=0)

# The initial condition
u0_devito = FD_devito.initial_condition_simple_gaussian(
    sigma=0.1, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
    dx=spacing[0], dy=spacing[1])

# Initialize the VectorTimeFunction with the initial values
# Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
# and because 2nd order time discretization is neeed
u_devito.data[0] = u0_devito
u_devito.data[1] = u0_devito

laplacian = u_devito.dx2 + u_devito.dy2

# Accoustic wave equation
pde = laplacian - (1.0 / velocity_ ** 2) * u_devito.dt2

# Boundary conditions:
#I did not use physically meaningful boundary conditions so you might want to change theese
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
stencil_x = Eq(u_devito.forward, solve(pde, u_devito.forward))
op = Operator([stencil_x] + bc)

res_list_devito = []

time_list = np.linspace(0, 1, 200).tolist()

#This is not really optimal, I did it like this because I needed it to compare to my PINNs
#Actually you dont even need a loop and can just specify the final time, check the examples online and on Slack
for i in time_list:
    res_list_devito.append(np.transpose(
        u_devito.data[1][0:numpoints_sqrt, 0:numpoints_sqrt]).copy())
    op(time_M=10, dt=dt)

s = 12 * np.mean(np.abs(res_list_devito[0]))
for h in range(0, len(res_list_devito)):
    data = res_list_devito[h]
    fig,ax = plt.subplots()
    ax.imshow(data, cmap='bwr', vmin=-s, vmax=s)
    plt.show()