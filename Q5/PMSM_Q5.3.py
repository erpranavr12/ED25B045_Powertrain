import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 0.02, 1000)   
f = 50                           # frequency
w = 2 * np.pi * f                # angular frequency

# Three-phase currents
ia = np.sin(w * t)
ib = np.sin(w * t - 2*np.pi/3)
ic = np.sin(w * t + 2*np.pi/3)

# Clarke Transformation (abc → αβ)
i_alpha = ia
i_beta = (ia + 2*ib) / np.sqrt(3)

# Park Transformation (αβ → dq)
theta = w * t

i_d =  i_alpha*np.cos(theta) + i_beta*np.sin(theta)
i_q = -i_alpha*np.sin(theta) + i_beta*np.cos(theta)

# Plot 1: Three-phase currents
plt.figure()
plt.plot(t, ia, label='ia')
plt.plot(t, ib, label='ib')
plt.plot(t, ic, label='ic')
plt.title("Original Three-Phase Currents (abc)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

# Plot 2: Clarke components
plt.figure()
plt.plot(t, i_alpha, label='i_alpha')
plt.plot(t, i_beta, label='i_beta')
plt.title("Clarke Transform (αβ)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

# Plot 3: Park components
plt.figure()
plt.plot(t, i_d, label='i_d')
plt.plot(t, i_q, label='i_q')
plt.title("Park Transform (dq)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

plt.show()