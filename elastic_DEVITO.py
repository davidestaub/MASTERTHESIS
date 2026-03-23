from sympy import pprint
#Only need torch because of how I handled the Lamé parameter model generation, If you use your own models you dont need torch
import torch
import numpy as np
import matplotlib.pyplot as plt
#The package you need to install
from devito import *
#My code for lame parameter model generation
import mixture_model
#My code with helper functions
import FD_devito



#Number of points along x and y axis, total number of points will be numpoints_sqrt**2
numpoints_sqrt = 512
dt = 1.096e-3

#Set Velocity model (constant,mixture,layered,layered_sine)
model_type ="layered_sine"

domain_length_x = 2
domain_length_y = 2
extent = (domain_length_x, domain_length_y)

devito_center = [domain_length_x  / 2.0, domain_length_y  / 2.0]

shape = (numpoints_sqrt , numpoints_sqrt )
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
spacing = (extent[0] / shape[0], extent[0] / shape[0])
X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                   np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))

# You can also remove this and set your own Lamé Parameter models
if model_type == "mixture":
    # Generate mixtures for mu and lambda
    mu_mixture = FD_devito.generate_mixture().numpy()
    lambda_mixture = FD_devito.generate_mixture().numpy()
    lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
    mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)
elif model_type == "constant":
    lambda_solid = np.full(X.shape, 20.0)
    mu_solid = np.full(X.shape, 30.0)
elif model_type == 'layered':
    lambda_solid, mu_solid = mixture_model.compute_lambda_mu_layers_accoustic_slow(torch.tensor(X), torch.tensor(Y), 5)
    lambda_solid = lambda_solid.numpy()
    mu_solid = mu_solid.numpy()
elif model_type == 'layered_sine' or model_type == 'layered_sine_simpler_3':
    lambda_solid, mu_solid = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
else:
     raise Exception("model type {} not implemented".format(model_type))

Xp, Yp = np.meshgrid(np.linspace(0, (numpoints_sqrt) * spacing[0], numpoints_sqrt),np.linspace(0, (numpoints_sqrt) * spacing[1], numpoints_sqrt))


# Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
# You can change the spatial and temporal order but be careful as the scheme will be chosen automatically and sometimes this produces weird behaviours.
#Example: 2nd-order-accurate derivatives are forward by default which make the cross derivatives weird and makes 2nd-order elastic unstable. if you want 2nd order accurate use dxc for the cross derivatives
ux_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)
uy_devito = TimeFunction(name='uy_devito', grid=grid, space_order=4, time_order=2)

lambda_ = Function(name='lambda_f', grid=grid, space_order=4)
mu_ = Function(name='mu_f', grid=grid, space_order=4)
initialize_function(lambda_, lambda_solid, nbl=0)
initialize_function(mu_, mu_solid, nbl=0)

#Change to any density you want
rho_solid = 100.0
#Change to any center you want
mu_quake_devito = [devito_center[0], devito_center[1]]
#Initializing the initial conditions
u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
                sigma=0.1, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
                dx=spacing[0], dy=spacing[1])

# Initialize the VectorTimeFunction with the initial values
# Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
# and because 2nd order time discretization is neeed
ux_devito.data[0] = u0x_devito
uy_devito.data[0] = u0y_devito
ux_devito.data[1] = u0x_devito
uy_devito.data[1] = u0y_devito

# Divergence of stress tensor for ux
#I separate it for visibility
div_stress_ux = (lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy

# Divergence of stress tensor for uy
div_stress_uy = (lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx

# Elastic wave equation for ux and uy
pde_x = rho_solid * ux_devito.dt2 - div_stress_ux
pde_y = rho_solid * uy_devito.dt2 - div_stress_uy

# Boundary conditions:
#I did not use physically meaningful boundary conditions so you might want to change theese
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

#Like this you can print it out to see the actual stencil
pprint(stencil_x)
pprint(stencil_y)

#Formulating the operator
op = Operator([stencil_x] + [stencil_y] + bc)

res_list_devito_x = []
res_list_devito_y = []
res_list_devito_u = []

time_list = np.linspace(0, 1, 100).tolist()


#This is not really optimal, I did it like this because I needed it to compare to my PINNs
#Actually you dont even need a loop and can just specify the final time, check the examples online and on Slack
for i in time_list:

    res_list_devito_x.append(np.transpose(ux_devito.data[1]).copy())
    res_list_devito_y.append(np.transpose(uy_devito.data[1]).copy())
    res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1]).copy() ** 2 + np.transpose(ux_devito.data[1]).copy() ** 2))

    op(time_M=10, dt=dt)


#Setting plotting range, change to what you prefere.
s = 12 * np.mean(np.abs(res_list_devito_x[0]))

#You could also plot just the y and or x displacement field in the same style
for h in range(0, len(res_list_devito_u)):
    fig, ax = plt.subplots()
    ax.imshow(res_list_devito_u[h], cmap='bwr', vmin=-s, vmax=s)
    plt.show()




