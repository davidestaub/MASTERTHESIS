import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



domain_extrema = torch.tensor([[-1.0, 0.0],  # Time dimension
                                    [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension

soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])

def convert(tens):
    assert (tens.shape[1] == domain_extrema.shape[0])
    return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]

n_collocation_points = 40000
input_s = convert(soboleng.draw(int(n_collocation_points)))
sorted_indices = torch.argsort(input_s[:, 0])
input_s = input_s[sorted_indices]
input_s.requires_grad = True


lamda_solid = 1.0
mu_solid = 1.0
rho_solid = 100.0
rho_fluid = 1.0
c2 = 1.0
mu_quake = [0, 0]
sigma_quake = min(2, 1) * 0.12
radius = 0.2




def initial_condition3(input_tensor, inner_radius=0.1):
    x = input_tensor[:, 1] - mu_quake[0]
    y = input_tensor[:, 2] - mu_quake[1]

    r = torch.sqrt((x ** 2 + y ** 2) + 1e-8)  # radius from the center of the quake
    theta = torch.atan2(y, x+ 1e-8)  # angle from the positive x-axis

    # create a mask that is 1 inside the annulus and 0 elsewhere
    inside_annulus = ((r < radius) & (r > inner_radius)).float()

    # interpolate u0x from -1 on the left to +1 on the right, and gradually decrease to 0 towards the center
    u0x = inside_annulus * r / radius * torch.cos(theta)
    # interpolate u0y from +1 at the bottom to -1 at the top, and gradually decrease to 0 towards the center
    u0y = inside_annulus * r / radius * torch.sin(theta)

    return u0x, u0y

import torch.nn.functional as F

def initial_condition_explosion(input_tensor, sigma=0.1):
    x = input_tensor[:, 1] - mu_quake[0]
    y = input_tensor[:, 2] - mu_quake[1]

    # Generate 2D Gaussian distribution
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))

    # Compute gradients of the Gaussian
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)

    # Normalize gradients to get initial velocity field
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))

    return u0x, u0y







def initial_condition2(input_tensor):
    t = input_tensor[:, 0]
    x_part = torch.pow(input_tensor[:, 1] - mu_quake[0], 2)
    y_part = torch.pow(input_tensor[:, 2] - mu_quake[1], 2)

    exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part + 1e-8) / sigma_quake), 2)
    earthquake_spike = torch.exp(exponent)
    u0x = earthquake_spike  # * solid_mask
    u0y = earthquake_spike  # * solid_mask
    gradient_x = torch.autograd.grad(u0x.sum(), input_s, create_graph=True)[0]

    dx_x = gradient_x[:, 1]
    dy_x = gradient_x[:, 2]
    print(u0x.requires_grad)
    return dx_x,dy_x

def initial_condition(input_tensor):
    t = input_tensor[:, 0]
    x_part = torch.pow(input_tensor[:, 1] - mu_quake[0], 2)
    y_part = torch.pow(input_tensor[:, 2] - mu_quake[1], 2)

    exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part + 1e-8) / sigma_quake), 2)
    earthquake_spike = torch.exp(exponent)
    u0x = earthquake_spike  # * solid_mask
    u0y = earthquake_spike  # * solid_mask

    return u0x,u0y

ux1,uy1 = initial_condition(input_s)
print(ux1.shape,uy1.shape)
ux2,uy2 = initial_condition3(input_s)
print(ux2.shape,uy2.shape)

udx,udy = initial_condition_explosion(input_s)

im_init_dx = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=udx.detach().numpy(),s=5)
plt.colorbar(im_init_dx)
plt.title("donut x")
plt.plot(1,1, marker="o")
plt.show()
im_init_dy = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=udy.detach().numpy(),s=5)
plt.colorbar(im_init_dy)
plt.title("donut y")
plt.plot(1,1, marker="o")
plt.plot(0,1, marker="x")
plt.show()
exit()

im_init_x = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=ux2.detach().numpy(),s=5)
plt.colorbar(im_init_x)
plt.title("new init x")
plt.show()

im_init_x = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=uy2.detach().numpy(),s=5)
plt.colorbar(im_init_x)
plt.title("new init y")
plt.show()


def get_solid_residual(input_s):
    u_x,u_y = initial_condition_explosion(input_s)
    u_x = u_x.unsqueeze(1)
    u_y = u_y.unsqueeze(1)
    gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
    print(gradient_x.shape)
    gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
    dt_x = gradient_x[:, 0]
    dx_x = gradient_x[:, 1]
    dy_x = gradient_x[:, 2]
    dt_y = gradient_y[:, 0]
    dx_y = gradient_y[:, 1]
    dy_y = gradient_y[:, 2]

    im_gradx_x = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dx_x.detach().numpy(), s=5)
    plt.colorbar(im_gradx_x)
    plt.title("dux/dx")
    plt.show()

    im_gradx_y = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dy_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_y)
    plt.title("dux/dy")
    plt.show()

    im_gradx_t = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_t)
    plt.title("dux/dt")
    plt.show()


    im_grady_x = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dx_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_x)
    plt.title("duy/dx")
    plt.show()

    im_grady_y = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dy_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_y)
    plt.title("duy/dy")
    plt.show()

    im_grady_t = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_t)
    plt.title("duy/dt")
    plt.show()

    print(dt_x.requires_grad)
    dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
    dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]
    print("dt2_x shape = ",dt2_x.shape)

    im_grady_t2 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt2_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_t2)
    plt.title("duy/dt2")
    plt.show()

    im_gradx_t2 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt2_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_t2)
    plt.title("dux/dt2")
    plt.show()


    # Reshape the gradients into tensors of shape [batch_size, 1]
    #dx_x = dx_x.view(-1, 1)
    #dy_x = dy_x.view(-1, 1)
    #dx_y = dx_y.view(-1, 1)
    #dy_y = dy_y.view(-1, 1)
    print(" dx_x shape = ",dx_x.shape)
    diag_1 = 2.0 * dx_x
    diag_2 = 2.0 * dy_y
    off_diag = dy_x + dx_y
    # Stack your tensors to a 2x2 tensor
    # The size of b will be (n_points, 2, 2)
    eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)
    #eps = eps.squeeze()
    print("eps shape = ",eps.shape)

    im_eps00 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=eps[0,0,:].detach().numpy(),s=5)
    plt.colorbar(im_eps00)
    plt.title("eps00")
    plt.show()
    im_eps01 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[0, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_eps01)
    plt.title("eps01")
    plt.show()
    im_eps11 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[1, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_eps11)
    plt.title("eps11")
    plt.show()
    im_eps10 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[1, 0, :].detach().numpy(), s=5)
    plt.colorbar(im_eps10)
    plt.title("eps10")
    plt.show()

    stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
    stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
    stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
    stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                 torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

    print("stress tensor shape =",stress_tensor.shape)

    im_sigma00 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=stress_tensor[0,0,:].detach().numpy(),s=5)
    plt.colorbar(im_sigma00)
    plt.title("sigma00")
    plt.show()
    im_sigma01 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[0, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma01)
    plt.title("sigma01")
    plt.show()
    im_sigma11 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[1, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma11)
    plt.title("sigma11")
    plt.show()
    im_sigma10 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[1, 0, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma10)
    plt.title("sigma10")
    plt.show()

    # Compute divergence of the stress tensor
    div_stress = torch.zeros(2,input_s.size(0), dtype=torch.float32, device=input_s.device)
    div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                       torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
    div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                       torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

    print("div stress shape =",div_stress.shape)

    im_div_stress0 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=div_stress[0, :].detach().numpy(),s=5)
    plt.colorbar(im_div_stress0)
    plt.title("div_stress00")
    plt.show()
    im_div_stress1 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=div_stress[1, :].detach().numpy(), s=5)
    plt.colorbar(im_div_stress1)
    plt.title("div_stress01")
    plt.show()



    dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
    print("dt2_combined shape = ",dt2_combined.shape)
    residual_solid = rho_solid * dt2_combined - div_stress
    print("residual shape = ",residual_solid.shape)

    im_res0 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=residual_solid[0, :].detach().numpy(), s=5,vmin=torch.min(residual_solid[0, :]),vmax=torch.max(residual_solid[0, :]))
    plt.colorbar(im_res0)
    plt.title("res0")
    plt.show()
    im_res1 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=residual_solid[1, :].detach().numpy(), s=5,vmin=torch.min(residual_solid[1, :]),vmax=torch.max(residual_solid[1, :]))
    plt.colorbar(im_res1)
    plt.title("res1")
    plt.show()
    print(residual_solid)
    residual_solid = residual_solid.reshape(-1, )


    return residual_solid

residual = get_solid_residual(input_s)
print(residual)
residual_x = residual[:40000]
residual_y = residual[40000:]

im_res_x = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=residual_x.detach().numpy(),s=5)
plt.colorbar(im_res_x)
plt.show()

im_res_y = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=residual_y.detach().numpy(),s=5)
plt.colorbar(im_res_y)
plt.show()







