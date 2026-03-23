import torch
import matplotlib.pyplot as plt
import numpy as np


def initial_condition_explosion(input_tensor, sigma):
    x = input_tensor[:, 0] - mu_quake[0]
    y = input_tensor[:, 1] - mu_quake[1]
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))
    print(torch.max(torch.abs(grad_x)),torch.max(torch.abs(grad_y)))
    return u0x, u0y


if __name__=='__main__':
    # Parameters
    sigma = 0.1
    mu_quake = [0, 0]  # Assuming the explosion is centered at (0,0)

    # Create a meshgrid for the domain
    x = np.linspace(-1, 1, 10000)
    y = np.linspace(-1, 1, 10000)
    X, Y = np.meshgrid(x, y)

    # Convert to tensor
    input_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    # Compute u0x, u0y
    u0x, u0y = initial_condition_explosion(input_tensor, sigma)

    # Convert back to numpy for plotting
    u0x = u0x.numpy().reshape(X.shape)
    u0y = u0y.numpy().reshape(Y.shape)
    magnitude = np.sqrt(u0x ** 2 + u0y ** 2)

    # Setting min and max values for plots
    min_val = -1.0
    max_val = 1.0

    # Increasing font size for better visibility
    # plt.rcParams.update({'font.size': 20})

    # Creating the figure and axes with revised layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1]})

    # u0x with limited number of ticks
    im1 = axes[0].imshow(u0x, extent=[-1, 1, -1, 1], origin='lower', cmap='bwr', vmin=min_val, vmax=max_val)
    axes[0].set_title('$u_x^{0}$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_xticks(np.linspace(-1, 1, 5))
    axes[0].set_yticks(np.linspace(-1, 1, 5))

    # u0y with frame but no ticks
    im2 = axes[1].imshow(u0y, extent=[-1, 1, -1, 1], origin='lower', cmap='bwr', vmin=min_val, vmax=max_val)
    axes[1].set_title('$u_y^{0}$')
    axes[1].spines['top'].set_visible(True)
    axes[1].spines['right'].set_visible(True)
    axes[1].spines['bottom'].set_visible(True)
    axes[1].spines['left'].set_visible(True)
    axes[1].set_xticks(np.linspace(-1, 1, 5))
    axes[1].set_yticks(np.linspace(-1, 1, 5))
    # axes[1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Magnitude with frame but no ticks
    im3 = axes[2].imshow(magnitude, extent=[-1, 1, -1, 1], origin='lower', cmap='bwr', vmin=min_val, vmax=max_val)
    axes[2].set_title('$|u|^0$')
    axes[2].spines['top'].set_visible(True)
    axes[2].spines['right'].set_visible(True)
    axes[2].spines['bottom'].set_visible(True)
    axes[2].spines['left'].set_visible(True)
    axes[2].set_xticks(np.linspace(-1, 1, 5))
    axes[2].set_yticks(np.linspace(-1, 1, 5))
    # axes[2].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Adding a single colorbar on the right side
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im3, cax=cbar_ax)
    plt.show()
    plt.savefig('initial.pdf', dpi=400)