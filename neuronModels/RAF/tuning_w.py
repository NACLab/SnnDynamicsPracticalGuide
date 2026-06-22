import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy

output_dir = "../../exp/neuronModels/RAF"

from neuronModels.RAF.sim import plot_boundary
import os
os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":
    plot_boundary(-1.5, -1.5, -1.5, b=1, omega=[10, 7.5, 5], dt=0.015, cycles=1,
                  can_explode=True, spike_terminate=False,
                  point_labels=[10, 7.5, 5],
                  title=f"Changing $\omega$, Fixed $b$ = {1:.2f}",
                  save_title="tuning_w")
