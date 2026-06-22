from .sim import plot_boundary

output_dir = "../../exp/neuronModels/RAF"

if __name__ == "__main__":
    plot_boundary(-1.5, -1.5, -1.5, b=1, omega=[10, 7.5, 5], dt=0.015, cycles=1,
                  can_explode=True, spike_terminate=False,
                  point_labels=[10, 7.5, 5],
                  title=f"Changing $\omega$, Fixed $b$ = {1:.2f}",
                  save_title="tuning_w")
