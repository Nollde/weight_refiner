import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

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
    "loc": "upper left",
    "frameon": False,
}


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(b), where=b != 0)


def plot_n(data=None, reweighter=None, refiner=None, bins=100):
    # Create the Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data
    _, bins, __ = ax.hist(
        data[0],
        weights=data[-1],
        bins=bins,
        label="Data",
        color=colors["data"],
    )
    _, __, ___ = ax.hist(
        reweighter[0],
        weights=reweighter[-1],
        bins=bins,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )
    _, __, ___ = ax.hist(
        refiner[0],
        weights=refiner[-1],
        bins=bins,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )

    # Add legend
    plt.legend(**legend_kwargs)

    # Set labels
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\Sigma_i w_i$")


def plot_n_ratio(data=None, reweighter=None, refiner=None, bins=100):
    # Create the Figure
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
        squeeze=True,
    )

    # Plot the data
    hist_data, bins, __ = ax1.hist(
        data[0],
        weights=data[-1],
        bins=bins,
        label="Data",
        color=colors["data"],
    )
    hist_reweighter, _, __ = ax1.hist(
        reweighter[0],
        weights=reweighter[-1],
        bins=bins,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )
    hist_refiner, _, __ = ax1.hist(
        refiner[0],
        weights=refiner[-1],
        bins=bins,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )

    # Set labels and legend
    ax1.set_ylabel(r"$\Sigma_i w_i$")
    ax1.legend(**legend_kwargs)

    # Plot the ratio and error
    # Calculate ratio
    ratio_reweighter = safe_divide(hist_reweighter, hist_data)
    ratio_refiner = safe_divide(hist_refiner, hist_data)

    # Calculate w2 histograms
    hist_data_w2, bins = np.histogram(
        data[0],
        weights=data[-1] ** 2,
        bins=bins,
    )
    hist_reweighter_w2, _ = np.histogram(
        reweighter[0],
        weights=reweighter[-1] ** 2,
        bins=bins,
    )
    hist_refiner_w2, _ = np.histogram(
        refiner[0],
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
    ax2.errorbar(
        bin_centers,
        ratio_reweighter,
        yerr=ratio_reweighter_err,
        linestyle="none",
        color=colors["reweighter"],
    )
    ax2.errorbar(
        bin_centers,
        ratio_refiner,
        yerr=ratio_refiner_err,
        linestyle="none",
        color=colors["refiner"],
    )

    # Plot the 1:1 line
    ax2.axhline(y=1, linewidth=2, color="gray")

    # Set labels
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel("Ratio")


def plot_w(data=None, reweighter=None, refiner=None, bins=100):
    # Create the Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the weights
    _, bins, __ = ax.hist(
        data[-1],
        bins=bins,
        label="Data",
        color=colors["data"],
    )
    _, __, ___ = ax.hist(
        reweighter[-1],
        bins=bins,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )
    _, __, ___ = ax.hist(
        refiner[-1],
        bins=bins,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )

    # Add legend
    plt.legend(**legend_kwargs)

    # Set labels
    plt.xlabel(r"$w_i$")
    plt.ylabel(r"Counts")


def plot_w2(data=None, reweighter=None, refiner=None, bins=100):
    # Create the Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate w2 histograms
    hist_data_w2, bins = np.histogram(
        data[0],
        weights=data[-1] ** 2,
        bins=bins,
    )
    hist_reweighter_w2, _ = np.histogram(
        reweighter[0],
        weights=reweighter[-1] ** 2,
        bins=bins,
    )
    hist_refiner_w2, _ = np.histogram(
        refiner[0],
        weights=refiner[-1] ** 2,
        bins=bins,
    )

    # Turn w2 histograms into errors
    data_err = hist_data_w2**0.5
    reweighter_err = hist_reweighter_w2**0.5
    refiner_err = hist_refiner_w2**0.5

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot the errors
    ax.fill_between(
        bin_centers,
        data_err,
        label="Data",
        color=colors["data"],
        step="mid",
    )

    ax.plot(
        bin_centers,
        reweighter_err,
        label="Reweighter",
        color=colors["reweighter"],
        linestyle="None",
        marker="o",
    )
    ax.plot(
        bin_centers,
        refiner_err,
        label="Refiner",
        color=colors["refiner"],
        linestyle="None",
        marker="x",
    )

    # Add legend
    plt.legend(**legend_kwargs)

    # Set labels
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\sqrt{\Sigma_i w_i^2}$")


def plot_training(history, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['train', 'val'], **legend_kwargs)