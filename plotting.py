import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec


mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["patch.linewidth"] = 2
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "18"


colors = {
    "data": (205 / 256, 223 / 256, 237 / 256),
    "refiner": "green",
    "reweighter": "orange",
}


legend_kwargs = {
    "loc": "upper center",
    "frameon": False,
    "mode": "expand",
    "ncol": 3,
}


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(b), where=b != 0)


def savefig(path):
    # Create directory if it does not exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot
    plt.savefig(path)


def get_figure(figsize=(8, 7)):
    # Create a figure and set padding
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.90)
    return fig


def get_fig_with_legend(figsize=(8, 7), height_ratios=[0.5, 3]):
    # Create a figure
    fig = get_figure(figsize=figsize)

    # Create a GridSpec with 3 rows and 1 columns
    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, figure=fig)

    # Create subplots
    legend_axis = fig.add_subplot(gs[0, 0])
    plot_axis = fig.add_subplot(gs[1, 0])

    # Adjust the position of the third subplot to restore some space
    pos2 = gs[1, 0].get_position(fig)
    pos2.y1 += 0.07  # Adjust this value as needed
    plot_axis.set_position(pos2)

    # No ticks and labels
    legend_axis.tick_params(
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    legend_axis.spines["bottom"].set_visible(False)
    plot_axis.spines["top"].set_visible(False)
    return fig, (legend_axis, plot_axis)


def get_fig_with_legend_ratio(figsize=(8, 8), height_ratios=[0.5, 3, 1]):
    fig = get_figure(figsize=figsize)

    # Create a GridSpec with 3 rows and 1 columns
    gs = gridspec.GridSpec(3, 1, height_ratios=height_ratios, figure=fig)

    # Create subplots
    legend_axis = fig.add_subplot(gs[0, 0])
    plot_axis = fig.add_subplot(gs[1, 0])
    ratio_axis = fig.add_subplot(gs[2, 0], sharex=plot_axis)

    # Adjust the position of the third subplot to restore some space
    pos2 = gs[1, 0].get_position(fig)
    pos2.y1 += 0.05  # Adjust this value as needed
    plot_axis.set_position(pos2)

    # No ticks and labels
    plot_axis.tick_params(
        labelbottom=False,
    )
    legend_axis.tick_params(
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    legend_axis.spines["bottom"].set_visible(False)
    plot_axis.spines["top"].set_visible(False)

    return fig, (legend_axis, plot_axis, ratio_axis)


def plot_raw(data=None, bins=100, transform=lambda x: x[:, 0], path=None):
    pos, neg, pos_weights, neg_weights = data
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    # Plot the data
    _, _, ___ = plot_axis.hist(
        transform(np.concatenate([pos, neg])),
        weights=np.concatenate([pos_weights, neg_weights]),
        bins=bins,
        label="Effective",
        color=colors["data"],
    )
    _, _, ___ = plot_axis.hist(
        transform(pos),
        weights=pos_weights,
        bins=bins,
        label="Positive",
        color=colors["refiner"],
        histtype="step",
    )
    _, __, ___ = plot_axis.hist(
        transform(neg),
        weights=neg_weights,
        bins=bins,
        label="Negative",
        color=colors["reweighter"],
        histtype="step",
    )

    # Set labels
    plot_axis.set_xlabel(r"$\xi$")
    plot_axis.set_ylabel(r"$\Sigma_i w_i$")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # Save the plot
    if path is not None:
        savefig(path)


def plot_n_ratio(
    data=None,
    reweighter=None,
    refiner=None,
    bins=100,
    transform=lambda x: x[:, 0],
    path=None,
):
    # Create the Figure
    fig, (legend_axis, plot_axis, ratio_axis) = get_fig_with_legend_ratio()

    # Plot the data
    hist_data, bins, __ = plot_axis.hist(
        transform(data[0]),
        weights=data[-1],
        bins=bins,
        label="Data",
        color=colors["data"],
    )
    hist_reweighter, _, __ = plot_axis.hist(
        transform(reweighter[0]),
        weights=reweighter[-1],
        bins=bins,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )
    hist_refiner, _, __ = plot_axis.hist(
        transform(refiner[0]),
        weights=refiner[-1],
        bins=bins,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )

    # Set labels
    plot_axis.set_ylabel(r"$\Sigma_i w_i$")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # Plot the ratio and error
    # Calculate ratio
    ratio_reweighter = safe_divide(hist_reweighter, hist_data)
    ratio_refiner = safe_divide(hist_refiner, hist_data)

    # Calculate w2 histograms
    hist_data_w2, bins = np.histogram(
        transform(data[0]),
        weights=data[-1] ** 2,
        bins=bins,
    )
    hist_reweighter_w2, _ = np.histogram(
        transform(reweighter[0]),
        weights=reweighter[-1] ** 2,
        bins=bins,
    )
    hist_refiner_w2, _ = np.histogram(
        transform(refiner[0]),
        weights=refiner[-1] ** 2,
        bins=bins,
    )

    # Turn w2 histograms into errors
    reweighter_err = (hist_data_w2 + hist_reweighter_w2) ** 0.5
    refiner_err = (hist_data_w2 + hist_refiner_w2) ** 0.5

    # Take relative errors
    ratio_reweighter_err = safe_divide(reweighter_err, hist_data)
    ratio_refiner_err = safe_divide(refiner_err, hist_data)

    # Plot the ratio
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ratio_axis.errorbar(
        bin_centers,
        ratio_reweighter,
        yerr=ratio_reweighter_err,
        linestyle="none",
        color=colors["reweighter"],
    )
    ratio_axis.errorbar(
        bin_centers,
        ratio_refiner,
        yerr=ratio_refiner_err,
        linestyle="none",
        color=colors["refiner"],
    )

    y_range = (0.8, 1.2)

    # Plot arrows for out-of-range values
    for ratio, color in zip(
        [ratio_reweighter, ratio_refiner], [colors["reweighter"], colors["refiner"]]
    ):
        for i, y in enumerate(ratio):
            if y > y_range[1]:
                ratio_axis.plot(
                    bin_centers[i],
                    y_range[1] - 0.02,
                    marker=(3, 0, 0),
                    color=color,
                    markersize=10,
                )
            elif y < y_range[0]:
                ratio_axis.plot(
                    bin_centers[i],
                    y_range[0] + 0.02,
                    marker=(3, 0, 180),
                    color=color,
                    markersize=10,
                )

    # Set the y-axis limit
    ratio_axis.set_ylim(y_range)

    # Plot the 1:1 line
    ratio_axis.axhline(y=1, linewidth=2, color="gray")

    # Set labels
    ratio_axis.set_xlabel(r"$\xi$")
    ratio_axis.set_ylabel("Ratio")

    # Save the plot
    if path is not None:
        savefig(path)


def plot_w(data=None, reweighter=None, refiner=None, bins=100, path=None):
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    # Plot the weights
    _, bins, __ = plot_axis.hist(
        data[-1],
        bins=bins,
        label="Data",
        color=colors["data"],
    )
    _, __, ___ = plot_axis.hist(
        reweighter[-1],
        bins=bins,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )
    _, __, ___ = plot_axis.hist(
        refiner[-1],
        bins=bins,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )

    # Set labels
    plot_axis.set_xlabel(r"$w_i$")
    plot_axis.set_ylabel(r"Counts")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # Save the plot
    if path is not None:
        savefig(path)


def plot_w2(
    data=None,
    reweighter=None,
    refiner=None,
    bins=100,
    transform=lambda x: x[:, 0],
    path=None,
):
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    # Calculate w2 histograms
    hist_data_w2, bins = np.histogram(
        transform(data[0]),
        weights=data[-1] ** 2,
        bins=bins,
    )
    hist_reweighter_w2, _ = np.histogram(
        transform(reweighter[0]),
        weights=reweighter[-1] ** 2,
        bins=bins,
    )
    hist_refiner_w2, _ = np.histogram(
        transform(refiner[0]),
        weights=refiner[-1] ** 2,
        bins=bins,
    )

    # Turn w2 histograms into errors
    data_err = hist_data_w2**0.5
    reweighter_err = hist_reweighter_w2**0.5
    refiner_err = hist_refiner_w2**0.5

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot the errors
    plot_axis.fill_between(
        bin_centers,
        data_err,
        label="Data",
        color=colors["data"],
        step="mid",
    )

    plot_axis.plot(
        bin_centers,
        reweighter_err,
        label="Reweighter",
        color=colors["reweighter"],
        linestyle="None",
        marker="o",
    )
    plot_axis.plot(
        bin_centers,
        refiner_err,
        label="Refiner",
        color=colors["refiner"],
        linestyle="None",
        marker="x",
    )

    # Set labels
    plot_axis.set_xlabel(r"$\xi$")
    plot_axis.set_ylabel(r"$\sqrt{\Sigma_i w_i^2}$")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # Save the plot
    if path is not None:
        savefig(path)


def plot_training(history, title="", path=None):
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    plot_axis.plot(history.history["loss"], label="train")
    plot_axis.plot(history.history["val_loss"], label="val")

    fig.suptitle(title)

    # Set labels
    plot_axis.set_xlabel("Epoch")
    plot_axis.set_ylabel("Loss")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # Save the plot
    if path is not None:
        savefig(path)
