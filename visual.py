import matplotlib.pyplot as plt
import numpy as np

block = np.array([
    [608.625, -30.1857, -61.1971, 27.2393, 56.125, -20.0952, -2.38765, 0.461815, ],
    [4.46552, -21.8574, -60.758, 10.2536, 13.1451, -7.08742, -8.53544, 4.87689, ],
    [-46.8345, 7.3706, 77.1294, -24.562, -28.9117, 9.93352, 5.41682, -5.64895, ],
    [-48.535, 12.0684, 34.0998, -14.7594, -10.2406, 6.29597, 1.83117, 1.94594, ],
    [12.125, -6.55345, -13.1961, -3.95143, -1.875, 1.74528, -2.78723, 3.13528, ],
    [-7.73474, 2.90546, 2.3798, -5.93931, -2.3778, 0.941392, 4.30371, 1.84869, ],
    [-1.03067, 0.183067, 0.416815, -2.41556, -0.877794, -3.01931, 4.12061, -0.661948, ],
    [-0.165376, 0.141607, -1.07154, -4.19291, -1.17031, -0.0977611, 0.501269, 1.67546, ],
])

# Create the heatmap
plt.figure(figsize=(6, 6))
plt.imshow(block, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Float Value")

# Add text labels to each cell
for i in range(8):
    for j in range(8):
        plt.text(j, i, f"{block[i, j]:.2f}", ha="center", va="center", color="black")

# Title and labels
plt.title("8x8 Block of Floats")
plt.xlabel("Column")
plt.ylabel("Row")

plt.show()
