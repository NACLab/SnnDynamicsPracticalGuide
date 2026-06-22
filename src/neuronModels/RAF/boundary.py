from .sim import plot_boundary
output_dir = "../../exp/neuronModels/RAF"

if __name__ == "__main__":
    plot_boundary(-1.75, -1.25,
                  point_labels=['Will Spike', 'Won\'t Spike'],
                  title="Boundary")





