import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Function definition: g(x,y) = sin(2*pi*f0*sqrt(x²+y²))
def g(xx, yy, f0):
    return np.sin(2 * np.pi * f0 * np.sqrt(xx**2 + yy**2))

# Parameters for demonstration
f0 = 5  # frequency of the signal
sampling_frequencies = [10, 20, 300]  # different sampling frequencies

# Plotting images to visualize the aliasing effect
fig, axs = plt.subplots(1, len(sampling_frequencies), figsize=(15, 5))

for i, fs in enumerate(sampling_frequencies):
    t = np.arange(0, 1, 1/fs)
    xx, yy = np.meshgrid(t, t)
    img = g(xx, yy, f0)

    axs[i].imshow(img, cmap='gray', extent=(0,1,0,1))
    axs[i].set_title(f'$f_s$ = {fs}, $f_0$ = {f0}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()
