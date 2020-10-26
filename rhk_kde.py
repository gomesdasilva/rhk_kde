"""
Functions to estimate the log(R5)
"""
import os, sys
import random
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import tqdm
import time

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import interp1d


def _kde1d(x, bw=0.07, n=100, xlims=False):
    """Gaussian Kernel Density Estimate (KDE) of distribution 'x' in
    the interval 'xlims' using the sklearn module.
    """
    x = np.asarray(x)

    kde = KernelDensity(bandwidth=bw)
    kde.fit(x[:, np.newaxis])

    if xlims:
        start = xlims[0]
        end = xlims[1]
    if not xlims:
        start = min(x)
        end = max(x)

    step = (end - start) / (n - 1)
    xi = np.linspace(start, end, n)

    density = np.exp(kde.score_samples(xi[:, np.newaxis]))
    mask = (xi >= start) & (xi <= end)
    prob = np.sum(density[mask] * step)

    return xi, density, prob


def _sample_from_pdf(x, pdf, n):
    """
    Sample data from empirical probability density function using inverse transform sampling.
    """
    cum_sum = np.cumsum(pdf)
    inverse_density_function = interp1d(cum_sum, x)
    b = np.zeros(n)
    for i in range(len( b )):
        u = random.uniform( min(cum_sum), max(cum_sum) )
        b[i] = inverse_density_function( u )
    return b


def _kde2d_sklearn(x, y, bw, thresh=1e-4, xbins=100j, ybins=100j, xlim=[-5.5, -3.75], ylim=[-3.0, 1], **kwargs):
    """Bivariate Kernel Density Estimate (KDE)."""

    xx, yy = np.mgrid[xlim[0]:xlim[1]:xbins,
                      ylim[0]:ylim[1]:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bw, **kwargs)
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))

    zz = np.reshape(z, xx.shape)

    zz[zz < thresh] = 0.0

    return xx, yy, zz


def _find_nearest(array, value):
    """Find closed value to 'value' in 'array'."""
    array = np.asarray(array)
    min_diff = np.ones(len(array))
    for k in range(len(array)):
        min_diff[k] = np.abs(array[k][0] - value)
    idx = min_diff.argmin()
    return idx


def _plot_rhk_kde2d(x, y, bw, xlabel=r"$x$", ylabel=r"$y$", show_points=True, save_plot=False, show_plot=True, savepath="2d_kde.pdf"):
    """Plot bivariate KDE of 'y' vs 'x'."""
    X, Y, Z = _kde2d_sklearn(x, y, bw=bw, xlim=[-5.5, -3.6], ylim=[-3, 0.5])

    if show_plot:
        plt.figure(figsize=(5, 2*1.5))

    plt.pcolormesh(X, Y, Z, shading="gouraud", cmap='Spectral_r')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if show_points:
        plt.plot(x, y, 'w.', ms=2, alpha=1)

    plt.tight_layout()

    if save_plot:
        plt.savefig(savepath)

    if show_plot:
        plt.show()


def get_rhk_std_pdf(log_rhk, bw=0.07, subset="all", key_x="log_rhk_med", key_y="log_sig_r5", filepath=None, show_plot=True, save_plot=False, savepath="rhk_std_kde.pdf"):
    """Estimates log(R_5) dispersion probability density function for a given log(R'HK) value using the bivariate KDE distribution presented in Gomes da Silva et al. (2020).

    Can use values from diferent catalogues if 'filepath' is not 'None'.

    Paraneters:
    -----------
    log_rhk : float
        Value of log(R'HK) activity level. Must be in range [-5.5, -3.6].

    bw : float
        Kernel bandwidth.

    subset : string
        Data set to be used. Options are:
            'all': main-sequence, subgiants and giants.
            'MS': main-sequence stars only.
            'dF': F dwarfs.
            'dG': G dwarfs.
            'dF': F dwarfs.

    key_x : string
        Column name for median log(R'HK) values to be obtained from csv file.

    key_y : string
        Column name for log(sigma(R_5)) values to be obtained from csv file.

    filepath : None, string
        If 'None' uses Gomes da Silva et al. (2020) catalogue, else is the path for catalogue in csv format to be loaded.

    show_plot : bool (default is True)
        If 'True' shows plot.

    save_plot : bool (default is False)
        If 'True' saves the plot to 'savepath' path.

    savepath : string
        Path where to save the plot.

    Returns:
    --------
    log(R_5) dispersion PDF x and y axis.

    Notes:
    ------
    - R5 = R'HK * 1e5
    - log(sigma(R5)) is the logarithm of the dispersion of R5
    - The KDE bandwidth is automatically calculated by sklearn.
    - Stars with dispersion values below that of HD60532 were removed due to insignificant variability (see paper).
    - The code finds the log(R'HK) bin closest to the 'log_rhk' input. The log(R'HK) resolution is ~0.01 dex. Can be changed by editing the 'xbins' and 'ybins' values below. Lower values will increase speed.
    """
    if not filepath:
        filepath = os.path.join(os.path.dirname(__file__), "data.csv")

    if log_rhk < -5.5 or log_rhk > -3.6:
        print("*** ERROR: log_rhk outside data boundaries [-5.5, -3.6]")
        return np.nan, np.nan

    df = pd.read_csv(filepath, index_col=0)

    if subset == 'all':
        pass
    elif subset == 'MS':
        df = df[df.lum_class == 'V']
    elif subset == 'dF':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('F')]
    elif subset == 'dG':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('G')]
    elif subset == 'dK':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('K')]
    else:
        print("*** ERROR: subset must be either 'all', 'MS', 'dF', 'dG', or 'dK'.")
        return np.nan, np.nan

    x = df[key_x].values
    y = df[key_y].values

    X, Y, Z = _kde2d_sklearn(x, y, thresh=1e-100, bw=bw, xlim=[-5.5, -3.6], ylim=[-3, 0.5], xbins=400j, ybins=400j)
    
    idx = _find_nearest(X, log_rhk)

    step = (max(Y[idx]) - min(Y[idx])) / (Y[idx].size - 1)
    probi = Z[idx]/Z[idx].max()
    probi /= sum(probi)
    probi /= step

    plt.figure(figsize=(5, 3.6*1.5))

    plt.subplot(211)
    _plot_rhk_kde2d(x, y, bw, xlabel=r"$\log~R'_\mathrm{HK}$ [dex]", ylabel = r"$\log~\sigma~(R_5)$ [dex]", show_points=True, show_plot=False)
    plt.axvline(log_rhk, color='w')

    plt.subplot(212)
    ax = plt.gca()
    ax.plot(Y[idx], probi, 'k-')

    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_xlabel(r"$\log~\sigma~(R_5)$ [dex]", fontsize=12)

    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    if save_plot:
        plt.savefig(savepath)

    if show_plot:
        plt.show()
    plt.close()

    return Y[idx], probi


def simulate_rhk_population(n_samples, subset='all', bw=0.07, key_x="log_rhk_med", key_y="log_sig_r5", filepath=None, show_plot=True, save_plot=False, savepath1="rhk_sim_hists.pdf", savepath2="rhk_sim_maps.pdf"):
    """Simulate stellar populations with median values of log(R'HK) and log(sigma(R5)) by sampling from the activity variability-level bivariate KDE presented in Gomes da Silva et al. (2020).

    Can use values from diferent catalogues if 'filepath' is not 'None'.

    Parameters:
    -----------
    n_samples : int
        Number of stars to be sampled.

    subset : string
        Data set to be used. Options are:
            'all': main-sequence, subgiants and giants.
            'MS': main-sequence stars only.
            'dF': F dwarfs.
            'dG': G dwarfs.
            'dF': F dwarfs.
    bw : float
        Kernel bandwidth.

    key_x : string
        Column name for median log(R'HK) values to be obtained from csv file.

    key_y : string
        Column name for log(sigmaR_5) values to be obtained from csv file.

    filepath : None, string
        If 'None' uses Gomes da Silva et al. (2020) catalogue, else is the path for catalogue in csv format to be loaded.

    show_plot : bool (default is True)
        If 'True' shows plots.

    save_plot : bool (default is False)
        If 'True' saves the plot to 'savepath' path.

    savepath1 : string
        Path where to save the histograms.
    
    savepath2 : string
        Path where to save the KDE maps.

    Returns:
    --------
    x_samples : array
        Simulated log(R'HK) values
    y_samples : array
        Simulated log(sigma(R5)) values

    Notes:
    ------
    - R5 = R'HK * 1e5
    - log(sigma(R5)) is the logarithm of the dispersion of R5
    """
    if not filepath:
        filepath = os.path.join(os.path.dirname(__file__), "data.csv")

    df = pd.read_csv(filepath)

    if subset == 'all':
        pass
    elif subset == 'MS':
        df = df[df.lum_class == 'V']
    elif subset == 'dF':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('F')]
    elif subset == 'dG':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('G')]
    elif subset == 'dK':
        df = df[df.lum_class == 'V']
        df = df[df.sptype.str.contains('K')]
    else:
        print("*** ERROR: subset must be either 'all', 'MS', 'dF', 'dG', or 'dK'.")
        return np.nan, np.nan

    x = df[key_x].values
    y = df[key_y].values

    kde_x, kde_xz, _ = _kde1d(x, bw=bw, n=100, xlims=[-5.5, -3.6])
    kde_y, kde_yz, _ = _kde1d(y, bw=bw, n=100, xlims=[-3, 0.5])

    x_samples = _sample_from_pdf(kde_x, kde_xz, n=n_samples)

    xx, yy, zz = _kde2d_sklearn(x, y, bw=bw, xlim=[-5.5, -3.6], ylim=[-3.0, 0.5], xbins=100j, ybins=100j)

    y_samples = np.ones_like(x_samples)
    for i, x_val in enumerate(tqdm.tqdm(x_samples)):
        time.sleep(0.01)
        idx = _find_nearest(xx, x_val)
        y_samples[i] = _sample_from_pdf(yy[idx], zz[idx], n=1)[0]

    xx_sim, yy_sim, zz_sim = _kde2d_sklearn(x_samples, y_samples, bw=bw, xlim=[-5.5, -3.6], ylim=[-3.0, 0.5], xbins=100j, ybins=100j)


    plt.figure(figsize=(5, 3.6*1.5))

    xlabel = r"$\log~R'_\mathrm{HK}$ [dex]"
    ylabel = r"$\log~\sigma~(R_5)$ [dex]"

    # Histograms:
    plt.subplot(211)
    bins = np.arange(-5.5, -3.6, 0.05)
    plt.hist(x_samples, color='r', alpha=0.5, density=True, bins=bins, label='sampled data')
    plt.hist(x, color='k', histtype='step', bins=bins, density=True, label='real data')
    plt.plot(kde_x, kde_xz, 'k-', label='PDF')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Probability density", fontsize=12)
    plt.legend(frameon=False, fontsize=10)

    plt.subplot(212)
    bins = np.arange(-3, 0.5, 0.1)
    plt.hist(y_samples, color='r', alpha=0.5, density=True, bins=bins, label='sampled data')
    plt.hist(y, color='k', histtype='step', bins=bins, density=True, label="real data")
    plt.plot(kde_y, kde_yz, 'k-', label='PDF')
    plt.xlabel(ylabel, fontsize=12)
    plt.ylabel("Probability density", fontsize=12)
    plt.legend(frameon=False, fontsize=10)

    plt.tight_layout()

    if save_plot:
        plt.savefig(savepath1)

    if show_plot:
        plt.show()
    plt.close()

    # bivariate KDE:
    plt.figure(figsize=(5, 3.6*1.5))

    plt.subplot(211)
    plt.pcolormesh(xx, yy, zz, cmap='Spectral_r', shading='gouraud')
    plt.plot(x, y, 'w.', ms=1.5)
    plt.xlim(-5.5, -3.6)
    plt.annotate(f"N = {x.size}", xy=(0.05, 0.9), xycoords='axes fraction', color='w')
    plt.ylabel(ylabel, fontsize=12)

    plt.subplot(212)
    plt.pcolormesh(xx_sim, yy_sim, zz_sim, cmap='Spectral_r', shading='gouraud')
    plt.plot(x_samples, y_samples, 'w.', ms=1.5)
    plt.xlim(-5.5, -3.6)
    plt.annotate(f"N = {x_samples.size}", xy=(0.05, 0.9), xycoords='axes fraction', color='w')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    plt.tight_layout()

    if save_plot:
        plt.savefig(savepath2)

    if show_plot:
        plt.show()
    plt.close()

    return x_samples, y_samples



