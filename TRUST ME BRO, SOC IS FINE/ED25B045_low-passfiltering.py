import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("current_data.csv")
raw = df["INV_DC_Bus_Current"].values

alpha = 0.15
filtered = np.zeros(len(raw))
filtered[0] = raw[0]
for i in range(1, len(raw)):
    filtered[i] = (1 - alpha) * filtered[i-1] + alpha * raw[i]

noise = raw - filtered

print(f"Raw std      : {np.std(raw):.2f} A")
print(f"Filtered std : {np.std(filtered):.2f} A")
print(f"Noise RMS    : {np.sqrt(np.mean(noise**2)):.2f} A")
print(f"Peak noise   : ±{np.max(np.abs(noise)):.2f} A")
print(f"SNR          : {20 * np.log10(np.std(filtered) / np.sqrt(np.mean(noise**2))):.1f} dB")

plt.plot(raw, color="royalblue", lw=0.8, alpha=0.7, label="Raw")
plt.plot(filtered, color="red", lw=1.5, label="Filtered")
plt.ylabel("Current (A)")
plt.xlabel("Sample Index")
plt.title("Raw vs Filtered")
plt.legend()
plt.grid(True)
plt.show()