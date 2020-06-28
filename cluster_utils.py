import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_scatter(df, grp_by, save2im):

    groups = df.groupby(grp_by)

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=3, label=name)
    ax.legend()
    plt.savefig(save2im)
    plt.show()
