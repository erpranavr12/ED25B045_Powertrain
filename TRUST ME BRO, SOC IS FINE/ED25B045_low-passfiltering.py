import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("current_data.csv")
raw_data = df["INV_DC_Bus_Current"].values

alpha = 0.15
filtered_data = np.zeros(len(raw_data))
filtered_data[0] = raw_data[0]
for i in range(1, len(raw_data)):
    filtered_data[i] = (1 - alpha) * filtered_data[i-1] + alpha * raw_data[i]

noise = raw_data - filtered_data

print(f"Noise RMS    : {np.sqrt(np.mean(noise**2)):.2f} A")
print(f"SNR          : {20 * np.log10(np.std(filtered_data) / np.sqrt(np.mean(noise**2))):.1f} dB")

plt.plot(raw_data, color="royalblue", lw=0.8, alpha=0.7, label="Raw")
plt.plot(filtered_data, color="red", lw=1.5, label="Filtered")
plt.ylabel("Current (A)")
plt.xlabel("Sample Data")
plt.title("Raw vs Filtered")
plt.legend()
plt.grid(True)
plt.show()