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


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def plot_n(data, refiner, reweighter, bins=100):
    # Create the Figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the data
    _, bins, __ = plt.hist(
        data[0], weights=data[-1], bins=bins, label="Data", color=colors["data"]
    )
    _, __, ___ = plt.hist(
        refiner[0],
        weights=refiner[-1],
        bins=bins,
        alpha=0.5,
        label="Refiner",
        color=colors["refiner"],
        histtype="step",
    )
    _, __, ___ = plt.hist(
        reweighter[0],
        weights=reweighter[-1],
        bins=bins,
        alpha=0.5,
        label="Reweighter",
        color=colors["reweighter"],
        histtype="step",
    )

    # Add legend
    plt.legend(frameon=False)

    # Set labels
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\Sigma w$")


def plot_n_ratio(data, refiner, reweighter, bins=100):
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
    ax1.set_ylabel("Counts")
    ax1.legend(frameon=False)

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


def plot_w():
    pass


def plot_w2():
    pass
