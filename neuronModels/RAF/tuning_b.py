import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy

output_dir = "../../exp/neuronModels/RAF"
from neuronModels.RAF.sim import plot_boundary
import os
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    plot_boundary(-1.7, -1.7, -1.7, b=[1.5, 1, 0.75], omega=7.5, dt=0.015,
                  can_explode=True, spike_terminate=False,
                  point_labels=[1.5, 1, 0.75],
                  title=f"Changing $b$, Fixed $\omega $ = {7.5:.2f}",
                  save_title="tuning_b")
