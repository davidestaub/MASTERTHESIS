import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scienceplots
plt.style.use(['science','ieee'])
plt.rcParams.update({'font.size': 20})


# Define the functions
def psi_minus(t, t1):
    return np.exp(-0.5 * (1.5 * t/t1)**2)


def psi_plus(t, t1):
    return np.tanh(2.5 *t / t1)**2



# t1 values
t1_values = [0.01, 0.1, 0.2]

# Time range
t = np.linspace(0, 1, 400)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

for ax, t1 in zip(axes, t1_values):
    ax.plot(t, psi_minus(t, t1), label=r'$\psi^{-}(t)$', color='blue')
    ax.plot(t, psi_plus(t, t1), label=r'$\psi^{+}(t)$', color='red')

    ax.set_title(f'$t_1 = {t1}$')
    ax.set_xlabel('Time ($t$)')
    ax.set_ylabel('Function Value')

    # Improve the appearance
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='black')

# Adjust layout
plt.tight_layout()

# Adding a general title
plt.suptitle('Comparison of $\psi^{-}(t)$ and $\psi^{+}(t)$ for Different $t_1$ Values', fontsize=16, y=1.05)

# Create a single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=False, ncol=2)

# Adjust layout again to make space for the new legend
plt.subplots_adjust(bottom=0.2)

# Save the figure
plt.savefig('ansatz_plot.pdf', format='pdf', dpi=400, bbox_inches='tight')

plt.show()