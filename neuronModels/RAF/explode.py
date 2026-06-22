import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy

output_dir = "../../exp/neuronModels/RAF"
import os
os.makedirs(output_dir, exist_ok=True)

from neuronModels.RAF.sim import plot_boundary

if __name__ == "__main__":
    plot_boundary(-0.8, b=0.33, omega=7.5, dt=0.015,
                  can_explode=True, spike_terminate=False,
                  title="Explosion Scenario")
