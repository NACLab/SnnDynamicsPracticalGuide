import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy

from neuronModels.RAF.sim import plot_boundary

output_dir = "../../exp/neuronModels/RAF"
import os
os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":
    plot_boundary(-2.1, -1.5, b=0.02, omega=0.1, dt=1,
                  current_inject=[(0, 5, 0.4), (1, 28, 0.4)],
                  title="interference")





