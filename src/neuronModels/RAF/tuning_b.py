from .sim import plot_boundary

output_dir = "../../exp/neuronModels/RAF"

if __name__ == "__main__":
    plot_boundary(-1.7, -1.7, -1.7, b=[1.5, 1, 0.75], omega=7.5, dt=0.015,
                  can_explode=True, spike_terminate=False,
                  point_labels=[1.5, 1, 0.75],
                  title=f"Changing $b$, Fixed $\omega $ = {7.5:.2f}",
                  save_title="tuning_b")
