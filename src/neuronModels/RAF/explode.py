from .sim import plot_boundary

output_dir = "exp/neuronModels/RAF"

if __name__ == "__main__":
    plot_boundary(-0.8, b=0.33, omega=7.5, dt=0.015,
                  can_explode=True, spike_terminate=False,
                  title="Explosion Scenario")
