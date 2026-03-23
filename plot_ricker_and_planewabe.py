import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])
plt.rcParams.update({'font.size': 20})

# Ricker Wavelet Function
def ricker_wavelet(t, a=4):
    return (1 - 2 * (a ** 2) * (t ** 2)) * np.exp(-(a ** 2) * (t ** 2))

# Plane Wave Function
def plane_wave(t, x, y, A=1, kx=2, ky=2, omega=2*np.pi, phi=0):
    return np.real(A * np.exp(1j * (kx * x + ky * y - omega * t + phi * np.sqrt(x**2 + y**2))))

# Time and Space Variables
t = np.linspace(-1, 1, 400)  # Time axis
x = 0  # For 1D plane wave, x and y are set to 0
y = 0


# Ricker Wavelet Plot
plt.figure(figsize=(8.5, 5.1))
ricker = ricker_wavelet(t)
plt.plot(t, ricker, label='Ricker Wavelet',color='blue')
plt.title('Ricker Wavelet')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
#plt.show()
plt.savefig('ricker.pdf',dpi=400)

# 1D Plane Wave Plot
plt.figure(figsize=(8.5, 5.1))
plane = plane_wave(t, x, y)
plt.plot(t, plane, label='1D Plane Wave', color='red')
plt.title('1D Plane Wave')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
#plt.show()
plt.savefig('planewave.pdf',dpi=400)