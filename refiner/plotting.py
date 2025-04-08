from itertools import combinations
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["patch.linewidth"] = 2
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "18"


def to_color(*rgb):
    return tuple([c / 256 for c in rgb])


colors = {
    "data": to_color(205, 223, 237),
    "refiner": "green",
    "reweighter": "orange",
    "positive": to_color(16, 47, 59),
    "negative": to_color(67, 150, 228),
}


legend_kwargs = {
    "loc": "upper center",
    "frameon": False,
    "mode": "expand",
    "ncol": 3,
}


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(b), where=b != 0)


def safe_log10(arr):
    return np.log10(arr, out=np.zeros_like(arr, dtype=arr.dtype), where=arr > 0)


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


def get_fig_with_legend(figsize=(8, 8), height_ratios=[0.5, 3]):
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
    plot_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plot_axis.get_yaxis().get_offset_text().set_position((-0.1, 0))
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
    plot_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plot_axis.get_yaxis().get_offset_text().set_position((-0.1, 0))
    return fig, (legend_axis, plot_axis, ratio_axis)


def auto_log_yaxis(bin_contents, threshold_proportion=0.2, threshold_ratio=1 / 100):
    # Calculate the maximum value in the dataset
    max_value = np.max(bin_contents)

    # Calculate the threshold value
    threshold_value = max_value * threshold_ratio

    # remove empty bins
    bin_contents = bin_contents[bin_contents > 0]

    # Count the number of events that have a value less than the threshold value
    num_small_values = np.sum(bin_contents < threshold_value)

    # Check if the proportion of events that have a value less than the threshold value is greater than or equal to the threshold proportion
    if num_small_values / len(bin_contents) >= threshold_proportion:
        return True
    else:
        return False


def plot_raw(data=None, bins=100, transform=lambda x: x[:, 0], path=None):
    pos, neg, pos_weights, neg_weights = data
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    # Plot the data
    counts, bins, ___ = plot_axis.hist(
        transform(np.concatenate([pos, neg])),
        weights=np.concatenate([pos_weights, neg_weights]),
        bins=bins,
        label="All",
        color=colors["data"],
    )
    _, _, ___ = plot_axis.hist(
        transform(pos),
        weights=pos_weights,
        bins=bins,
        label="Positive",
        color=colors["positive"],
        histtype="step",
    )
    _, __, ___ = plot_axis.hist(
        transform(neg),
        weights=neg_weights,
        bins=bins,
        label="Negative",
        color=colors["negative"],
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
    ratio_y_range=(0.8, 1.2),
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
        alpha=0.5,
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
    ratio_reweighter_err = safe_divide(reweighter_err, np.abs(hist_data))
    ratio_refiner_err = safe_divide(refiner_err, np.abs(hist_data))

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
        alpha=0.5,
    )

    # Plot arrows for out-of-range values
    for ratio, color in zip(
        [ratio_reweighter, ratio_refiner], [colors["reweighter"], colors["refiner"]]
    ):
        for i, y in enumerate(ratio):
            if y <= 0 or np.isinf(y) or np.isnan(y):
                # Skip when one distribution is empty
                # Skip when ratio is negative
                continue
            if y > ratio_y_range[1]:
                ratio_axis.plot(
                    bin_centers[i],
                    ratio_y_range[1] - 0.02,
                    marker=(3, 0, 0),
                    color=color,
                    markersize=10,
                )
            elif y < ratio_y_range[0]:
                ratio_axis.plot(
                    bin_centers[i],
                    ratio_y_range[0] + 0.02,
                    marker=(3, 0, 180),
                    color=color,
                    markersize=10,
                )

    # Set the y-axis limit
    ratio_axis.set_ylim(ratio_y_range)

    # Plot the 1:1 line
    ratio_axis.axhline(y=1, linewidth=2, color="gray")

    # Set labels
    ratio_axis.set_xlabel(r"$\xi$")
    ratio_axis.set_ylabel("Ratio")

    # Save the plot
    if path is not None:
        savefig(path)


def plot_n_ratio_multi(
    data=None,
    reweighter=None,
    refiner=None,
    bins=100,
    transform=lambda x: x[:, 0],
    ratio_unc="hilo",
    ratio_y_range=(0.9, 1.1),
    path=None,
):
    # Create the Figure
    fig, (legend_axis, plot_axis, ratio_axis) = get_fig_with_legend_ratio(
        height_ratios=[0.5, 3, 2]
    )

    def get_histograms(datas, *args, **kwargs):
        hists = []
        for data in datas:
            hist, bins = np.histogram(
                transform(data[0]),
                weights=data[-1],
                *args,
                **kwargs,
            )
            hists.append(hist)
        return np.array(hists)

    hist_data = get_histograms([data], bins=bins)
    # Calculate w2 histograms
    hist_data_w2, bins = np.histogram(
        transform(data[0]),
        weights=data[-1] ** 2,
        bins=bins,
    )
    hist_reweighters = get_histograms(reweighter, bins=bins)
    hist_refiners = get_histograms(refiner, bins=bins)

    mean_data = np.mean(hist_data, axis=0)
    mean_reweighters = np.mean(hist_reweighters, axis=0)
    mean_refiners = np.mean(hist_refiners, axis=0)

    # main plot
    plot_axis.bar(
        bins[:-1],
        mean_data,
        width=bins[1:] - bins[:-1],
        label="Data",
        color=colors["data"],
        align="edge",
    )

    plot_axis.step(
        bins,
        np.concatenate(([0], mean_reweighters)),
        label="Reweighter",
        color=colors["reweighter"],
        where="pre",
    )
    plot_axis.step(
        bins,
        np.concatenate(([0], mean_refiners)),
        label="Refiner",
        color=colors["refiner"],
        where="pre",
        alpha=0.5,
    )

    if auto_log_yaxis(mean_data):
        plot_axis.set_yscale("log")

    # Set labels
    plot_axis.set_ylabel(r"$\Sigma_i w_i$")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(handles=handles, labels=labels, **legend_kwargs)

    # ratio plot
    data_stat_err = hist_data_w2**0.5
    data_stat_err_rel = safe_divide(data_stat_err, np.abs(mean_data))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ratio_axis.errorbar(
        bin_centers,
        np.ones_like(bin_centers),
        yerr=data_stat_err_rel,
        color=colors["data"],
        zorder=0,
    )

    def get_low_high(data, mode="std"):
        if mode == "std":
            low = np.mean(data, axis=0) - np.std(data, axis=0)
            high = np.mean(data, axis=0) + np.std(data, axis=0)
        elif mode == "hilo":
            low = np.min(data, axis=0)
            high = np.max(data, axis=0)
        else:
            raise ValueError("mode must be 'std' or 'hilo'")
        return low, high

    ratio_reweighter = safe_divide(mean_reweighters, mean_data)
    low_reweighter, high_reweighter = get_low_high(hist_reweighters, mode=ratio_unc)
    ratio_axis.fill_between(
        bins,
        np.concatenate(([1], safe_divide(low_reweighter, mean_data))),
        np.concatenate(([1], safe_divide(high_reweighter, mean_data))),
        step="pre",
        lw=0,
        color=colors["reweighter"],
    )

    ratio_refiner = safe_divide(mean_refiners, mean_data)
    low_refiner, high_refiner = get_low_high(hist_refiners, mode=ratio_unc)
    ratio_axis.fill_between(
        bins,
        np.concatenate(([1], safe_divide(low_refiner, mean_data))),
        np.concatenate(([1], safe_divide(high_refiner, mean_data))),
        step="pre",
        lw=0,
        color=colors["refiner"],
        alpha=0.5,
    )

    # Plot arrows for out-of-range values
    for ratio, color in [
        [ratio_reweighter, colors["reweighter"]],
        [ratio_refiner, colors["refiner"]],
    ]:
        for i, y in enumerate(ratio):
            if y <= 0 or np.isinf(y) or np.isnan(y):
                # Skip when one distribution is empty
                # Skip when ratio is negative
                continue
            if y > ratio_y_range[1]:
                ratio_axis.plot(
                    bin_centers[i],
                    ratio_y_range[1] - 0.02,
                    marker=(3, 0, 0),
                    color=color,
                    markersize=10,
                )
            elif y < ratio_y_range[0]:
                ratio_axis.plot(
                    bin_centers[i],
                    ratio_y_range[0] + 0.02,
                    marker=(3, 0, 180),
                    color=color,
                    markersize=10,
                )

    # Set the y-axis limit
    ratio_axis.set_ylim(ratio_y_range)

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
        alpha=0.5,
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


def plot_w_2d_hist(
    reweighter=None,
    refiner=None,
    bins=100,
    transform=lambda x: x[:, 0],
    path=None,
):
    weight_bins = np.arange(0.0, 1.4, 0.1)
    fig, (legend_axis, plot_axis) = get_fig_with_legend()

    # Create the 2D histogram
    hist, xedges, yedges = np.histogram2d(
        refiner[0][:, 0],
        refiner[-1],
        bins=(bins, weight_bins),
    )

    levels = np.logspace(safe_log10(np.min(hist)), safe_log10(np.max(hist)), 15)
    norm = mpl.colors.LogNorm(vmin=levels[0], vmax=levels[-1])

    plot_axis.hist2d(
        transform(reweighter[0]),
        reweighter[-1],
        norm=norm,
        bins=(bins, weight_bins),
    )

    # Create the contour plot)
    xcenter = (xedges[:-1] + xedges[1:]) / 2
    ycenter = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenter, ycenter)

    plot_axis.contour(
        X,
        Y,
        hist.T,
        levels=levels,
        norm=norm,
    )

    # Set labels
    plot_axis.set_xlabel(r"$\xi$")
    plot_axis.set_ylabel(r"$w_i$")

    # Create proxy artists for the hist2d
    proxy1 = Patch(facecolor=to_color(170, 220, 50))
    proxy2 = Line2D([0], [0], color=to_color(75, 56, 117), lw=2)

    # Add the proxy artists to the legend
    legend_axis.legend(
        [proxy1, proxy2],
        ["Reweighter", "Refiner"],
        **legend_kwargs,
    )
    # Save the plot
    if path is not None:
        savefig(path)


def plot_w_2d_scatter(
    data=None,
    reweighter=None,
    refiner=None,
    transform=lambda x: x[:, 0],
    path=None,
    n_max=None,
):
    # Limit the number of points to plot
    n_max = int(n_max) if n_max is not None else None
    if n_max is not None:
        if data is not None:
            data = [d[:n_max] for d in data]
        if reweighter is not None:
            reweighter = [r[:n_max] for r in reweighter]
        if refiner is not None:
            refiner = [r[:n_max] for r in refiner]

    fig, (legend_axis, plot_axis) = get_fig_with_legend(height_ratios=[0.3, 3])

    if data:
        # Plot the data
        plot_axis.scatter(
            transform(data[0]),
            data[-1],
            label="Data",
            color=colors["data"],
            s=0.5,
            zorder=0,
        )
    plot_axis.scatter(
        transform(reweighter[0]),
        reweighter[-1],
        label="Reweighter",
        color=colors["reweighter"],
        s=0.5,
    )
    plot_axis.scatter(
        transform(refiner[0]),
        refiner[-1],
        label="Refiner",
        color=colors["refiner"],
        alpha=1,
        s=0.5,
        zorder=0,
    )

    # Set labels
    plot_axis.set_xlabel(r"$\xi$")
    plot_axis.set_ylabel(r"$w_i$")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    leg = legend_axis.legend(
        handles=handles, labels=labels, scatterpoints=100, **legend_kwargs
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

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
        alpha=0.5,
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


def plot_training(history, title=None, plot_batches=False, path=None):
    # Create the Figure
    fig, (legend_axis, plot_axis) = get_fig_with_legend()
    steps, loss = zip(*history["loss"])
    steps, val_loss = zip(*history["val_loss"])
    if not plot_batches:
        steps = np.arange(len(steps))

    plot_axis.plot(steps, loss, label="Training")
    plot_axis.plot(steps, val_loss, label="Validation")


    # Set labels
    if plot_batches:
        plot_axis.set_xlabel("Batch")
    else:
        plot_axis.set_xlabel("Epoch")
    plot_axis.set_ylabel("Loss")

    if plot_batches:
        plt.xticks(rotation=45)

    # Scientific notation to auto for train plots
    plot_axis.ticklabel_format(axis="y", style="plain")

    # Add legend
    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(title=title, handles=handles, labels=labels, **legend_kwargs)

    # Save the plot
    if path is not None:
        savefig(path)
