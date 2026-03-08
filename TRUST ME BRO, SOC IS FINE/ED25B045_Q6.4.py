import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("voltage_data.csv")
data = df["Measured_Voltage"].values
N = len(data)

A = 1
H = 1
Q = 1e-4
R = np.var(data) * 0.5

x_hat = data[0]   # x̂_0
P     = 1.0       # P_0

x_hat_k  = np.zeros(N)
K_k = np.zeros(N)
inno = np.zeros(N)

for k in range(N):

    x_hat_minus = A * x_hat
    P_minus     = A * P * A + Q

    K = P_minus * H * (H * P_minus * H + R) ** -1
    x_hat = x_hat_minus + K * (data[k] - H * x_hat_minus)
    P = (1 - K * H) * P_minus

    x_hat_k[k]  = x_hat
    K_k[k] = K
    inno[k]    = data[k] - x_hat

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Kalman Filter - Constant Voltage Estimation", fontsize=13, fontweight="bold")

axs[0].plot(data,  label="Measured z_k", alpha=0.5)
axs[0].plot(x_hat_k, label="KF Estimate x̂_k", linewidth=2)
axs[0].axhline(np.mean(data), color="orange", linestyle="--", label=f"Mean ({np.mean(data):.2f} V)")
axs[0].set_ylabel("Voltage (V)")
axs[0].set_title("Measured vs Filtered Voltage")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(K_k, color="red")
axs[1].set_ylabel("Kalman Gain K")
axs[1].set_title("Kalman Gain Over Time")
axs[1].grid(True)

axs[2].plot(inno, color="purple")
axs[2].axhline(0, linestyle="--", color="black", alpha=0.4)
axs[2].set_ylabel("Innovation z_k - Hx̂_k⁻ (V)")
axs[2].set_title("Estimation Error Over Time")
axs[2].grid(True)

plt.tight_layout()
plt.show()