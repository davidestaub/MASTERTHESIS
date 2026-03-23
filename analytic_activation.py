import numpy as np
import matplotlib.pyplot as plt

import torch

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)

import torch
import os

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


n_collocation_points = 100000
n_points_per_training_set = int(n_collocation_points)

domain_extrema = torch.tensor([[0.0, 1.0],[-0.3, 0.3], [-0.3, 0.3]])
soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])


def convert(tens):
    assert (tens.shape[1] == domain_extrema.shape[0])
    return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]

input_s = convert(soboleng.draw(int(n_collocation_points)))

lamda_solid = torch.tensor(20.0)#2.0 * 1e+8
mu_solid = torch.tensor(30.0)#3.0 * 1e+8
rho_solid = torch.tensor(100.0)#1000.0
rho_fluid = torch.tensor(1.0)
mu_quake = torch.tensor([0, 0])
torch.pi = torch.acos(torch.zeros(1)).item() * 2



mean2 = torch.zeros(2)
mean2[0] = mu_quake[0]
mean2[1] = mu_quake[1]

theta = torch.tensor(0.0)
# p wave speed
alpha = torch.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid)
# s wave speed
beta = torch.sqrt(mu_solid / rho_solid)

T=torch.tensor(0.1)
#TRAINABLE
M0 = torch.tensor(0.5)


def pinn_model_eval(input_tensor,theta_1,theta_2,theta_3,theta_4,M0,T,offset):


    t = input_tensor[:, 0]
    x = input_tensor[: ,1]
    y = input_tensor[: ,2]
    sx = 0.0
    sy = 0.0

    r_abs = torch.sqrt(torch.pow((x - mu_quake[0]) ,2) + torch.pow((y - mu_quake[1]) ,2))
    r = torch.sqrt(((x - sx) ** 2 + (y - sy) ** 2) + 1e-8)  # Distance from source

    r_abs = r_abs + 1e-20
    r_hat_x = (x - mu_quake[0] ) /r_abs
    r_hat_y = (y - mu_quake[1] ) /r_abs
    phi_hat_x = -1.0 * r_hat_y
    phi_hat_y = r_hat_x
    #phi = torch.atan2(y-mu_quake[1],x - mu_quake[0])

    M0_dot_input1 = (t + 1 + offset) - r_abs / alpha
    M0_dot_input2 = (t + 1 + offset) - r_abs / beta
    phi = torch.atan2(y-mu_quake[1],x - mu_quake[0])


    #M_1 = -(M0/T) * (M0_dot_input1 + T) * torch.exp(-M0_dot_input1/T)
    #M_2 = -(M0/T) * (M0_dot_input2 + T) * torch.exp(-M0_dot_input2/T)

   # A_IP_x = 4.0 * torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_x - 2.0 * (
              #  0.0 - torch.cos(theta) * torch.sin(phi) * phi_hat_x)
    #A_IP_y = 4.0 * torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_y - 2.0 * (
               # 0.0 - torch.cos(theta) * torch.sin(phi) * phi_hat_y)

    #A_IS_x = -3.0 * torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_x + 3.0 * (
                #0.0 - torch.cos(theta) * torch.sin(phi) * phi_hat_x)
    #A_IS_y = -3.0 * torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_y + 3.0 * (
                #0.0 - torch.cos(theta) * torch.sin(phi) * phi_hat_y)

    #intermediate_field_x = (1.0 / (4.0 * torch.pi * alpha ** 2)) * A_IP_x * (1.0 / r_abs ** 2) * M_1 + (
                #1.0 / (4.0 * torch.pi * beta ** 2)) * A_IS_x * (1.0 / r_abs ** 2) * M_2
    #intermediate_field_y = (1.0 / (4.0 * torch.pi * alpha ** 2)) * A_IP_y * (1.0 / r_abs ** 2) * M_1 + (
                #1.0 / (4.0 * torch.pi * beta ** 2)) * A_IS_y * (1.0 / r_abs ** 2) * M_2

    M_dot1 = M0/(T**2) * (M0_dot_input1 - 3.0*T/2.0) * torch.exp(-(M0_dot_input1- 3.0*T/2.0)**2/T**2)
    M_dot2 = M0/(T**2) * (M0_dot_input2- 3.0*T/2.0) * torch.exp(-(M0_dot_input2- 3.0*T/2.0)**2/T**2)



    A_FP_x = torch.sin(theta_1) * torch.cos(theta_3) * r_hat_x
    A_FP_y = torch.sin(theta_1) * torch.cos(theta_3) * r_hat_y

    A_FS_x = -torch.cos(theta_2) * torch.sin(theta_4) * phi_hat_x
    A_FS_y = -torch.cos(theta_2) * torch.sin(theta_4) * phi_hat_y


    far_field_x = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_x * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_x * (1.0 / r_abs) * M_dot2
    far_field_x = far_field_x/torch.max(torch.abs(far_field_x))
    #intermediate_field_x = intermediate_field_x/torch.max(torch.abs(intermediate_field_x))

    far_field_y = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_y * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_y * (1.0 / r_abs) * M_dot2


    return far_field_x,far_field_y #+ intermediate_field_x


index = 0
while True:
    for j in np.linspace(-0.95,-0.94,3).tolist():
        # Generate a random number between 0 and 2*pi
        analytic_x = torch.zeros(input_s[:, 0].shape)
        analytic_y = torch.zeros(input_s[:,0].shape)
        for i in range(0,10):
            theta_1 = np.random.uniform(0.0,2.0*np.pi) + 1e-5
            theta_2 = np.random.uniform(0.0,2.0*np.pi)+ 1e-5
            theta_3 = np.random.uniform(0.0,2.0*np.pi)+ 1e-5
            theta_4 = np.random.uniform(0.0,2.0*np.pi)+ 1e-5
            M0 = np.random.uniform(-1.0,1.0)
            T = np.random.uniform(-1.0,1.0)
            offset = np.random.uniform(0.0,0.01)

            # Convert the numbers into PyTorch tensors
            theta_1 = torch.tensor(theta_1)
            theta_2 = torch.tensor(theta_2)
            theta_3 = torch.tensor(theta_3)
            theta_4 = torch.tensor(theta_4)
            M0 = torch.tensor(M0)
            T = torch.tensor(T)
            offset = torch.tensor(offset)

            index = index + 1
            input_s[:, 0] = torch.full(input_s[:, 0].shape, j)
            analytic_x+= pinn_model_eval(input_s, theta_1,theta_2,theta_3,theta_4,M0,T,offset)[0]
            analytic_y+=pinn_model_eval(input_s, theta_1,theta_2,theta_3,theta_4,M0,T,offset)[1]

            ta = input_s[:, 0]
            xa = input_s[:, 1]
            ya = input_s[:, 2]

            ta = ta.unsqueeze(-1)
            xa = xa.unsqueeze(-1)
            ya = ya.unsqueeze(-1)

            point2 = torch.cat((xa, ya), dim=-1)

            # Apply time decay
            ta = ta.squeeze()

        im = plt.scatter(input_s[:, 1].detach(), input_s[:, 2].detach(), c=analytic_x, cmap="jet", s=0.05)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(im)
        plt.title("analytic x time = {}".format(j))
        plt.show()

        im = plt.scatter(input_s[:, 1].detach(), input_s[:, 2].detach(), c=analytic_y, cmap="jet", s=0.05)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(im)
        plt.title("analytic y time = {}".format(j))
        plt.show()
