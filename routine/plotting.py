import matplotlib.pyplot as plt


def ax_tick(data, x_var=None, y_var=None, **kwargs):
    ax = plt.gca()
    if x_var is not None:
        ax.set_xticks(data[x_var].unique())
    if y_var is not None:
        ax.set_yticks(data[y_var].unique())


def format_tick(x_formatter=None, y_formatter=None, **kwargs):
    ax = plt.gca()
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(x_formatter)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
