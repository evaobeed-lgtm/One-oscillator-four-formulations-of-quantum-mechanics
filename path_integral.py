#!/usr/bin/env python3
"""
Path integral Monte Carlo for the anharmonic oscillator
=======================================================

The path integral formulation (thesis section 2.8) is reviewed in

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016,

but not simulated there. This program does the simulation for
H = p^2 + q^4 and compares with the matrix mechanics results.

Method. In imaginary time the path integral becomes a statistical sum over
closed paths x(tau), 0 <= tau < beta, with weight exp(-S),

    S = sum_i [ (x_{i+1} - x_i)^2 / (4 a) + a x_i^4 ] ,   beta = M a,

(the kinetic term has 1/(4a) because the mass is m = 1/2). For large beta
only the ground state contributes. Paths are sampled with the Metropolis
algorithm; many independent paths are updated at once with NumPy, and
their spread gives honest error bars.

Measured quantities:
  * E0 = 3 <x^4>              (virial theorem, <p^2> = 2 <x^4>)
  * |psi_0(x)|^2              (histogram of all x_i)
  * E1 - E0                   (decay of the correlation <x(0) x(tau)>)

The time step a introduces an error with leading terms a^2 and a^4, which
is removed by simulating several values of a and extrapolating to a = 0.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anharmonic import AnharmonicOscillator


def simulate(a, beta=8.0, n_paths=1000, n_sweeps=2500, n_therm=500,
             rng=None, hist_bins=None):
    """Metropolis sampling of n_paths independent closed paths.

    Returns a dict with per-path averages of x^2 and x^4 (arrays of length
    n_paths), the correlation function per path, and a histogram of x.
    """
    rng = np.random.default_rng() if rng is None else rng
    M = int(round(beta / a))
    M += M % 2                                   # even number of time slices
    x = rng.normal(0.0, 0.5, size=(n_paths, M))
    step = 4.0 * np.sqrt(a)                      # proposal width
    sites = [np.arange(0, M, 2), np.arange(1, M, 2)]
    n_lag = M // 4

    def local_action(xi, left, right):
        return ((xi - left) ** 2 + (right - xi) ** 2) / (4 * a) + a * xi ** 4

    sum2 = np.zeros(n_paths)
    sum4 = np.zeros(n_paths)
    corr = np.zeros((n_paths, n_lag))
    hist = None if hist_bins is None else np.zeros(len(hist_bins) - 1)
    accepted = 0
    for sweep in range(n_therm + n_sweeps):
        # even and odd sites do not interact among themselves, so each
        # half of the lattice can be updated in one vectorized step
        for s in sites:
            left = x[:, (s - 1) % M]
            right = x[:, (s + 1) % M]
            old = x[:, s]
            new = old + step * (rng.random(old.shape) - 0.5)
            dS = local_action(new, left, right) - local_action(old, left, right)
            accept = rng.random(old.shape) < np.exp(-dS)
            x[:, s] = np.where(accept, new, old)
            accepted += accept.sum()
        if sweep >= n_therm:
            x2 = x * x
            sum2 += x2.mean(axis=1)
            sum4 += (x2 * x2).mean(axis=1)
            # <x_i x_{i+lag}> for all lags at once (Wiener-Khinchin)
            power = np.abs(np.fft.rfft(x, axis=1)) ** 2
            corr += np.fft.irfft(power, n=M, axis=1)[:, :n_lag] / M
            if hist is not None and sweep % 10 == 0:
                hist += np.histogram(x, bins=hist_bins)[0]
    return dict(a=beta / M, M=M, x2=sum2 / n_sweeps, x4=sum4 / n_sweeps,
                corr=corr / n_sweeps, hist=hist,
                acceptance=accepted / ((n_therm + n_sweeps) * n_paths * M))


def mean_and_error(samples):
    """Mean and standard error from independent paths."""
    return samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))


def energy_gap(corr, a, tau_min=0.3, tau_max=1.2, n_blocks=20):
    """E1 - E0 from C(tau) ~ exp(-(E1 - E0) tau), by a straight-line fit to
    log C over tau_min <= tau <= tau_max. Error from a jackknife over
    blocks of paths."""
    lags = np.arange(corr.shape[1]) * a
    sel = (lags >= tau_min) & (lags <= tau_max)

    def fit(c):
        return -np.polyfit(lags[sel], np.log(c[sel]), 1)[0]

    blocks = np.array_split(np.arange(corr.shape[0]), n_blocks)
    total = corr.sum(axis=0)
    jack = np.array([fit((total - corr[b].sum(axis=0)) /
                         (corr.shape[0] - len(b))) for b in blocks])
    gap = fit(corr.mean(axis=0))
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((jack - jack.mean()) ** 2))
    return gap, err


def extrapolate(a_values, y, yerr):
    """Weighted least-squares fit y = c0 + c1 a^2 + c2 a^4; returns the
    value at a = 0, its error, and the coefficients."""
    A = np.vstack([np.ones_like(a_values), a_values ** 2, a_values ** 4]).T
    W = np.diag(1 / yerr ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ A.T @ W @ y
    return coef[0], np.sqrt(cov[0, 0]), coef


def main(outdir="figures", seed=1948):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    osc = AnharmonicOscillator()
    E0, gap_exact = osc.E[0], osc.E[1] - osc.E[0]
    x2_exact = osc.expectation(0, 2)
    print("Matrix mechanics: E0 = %.8f, E1 - E0 = %.8f, <x^2> = %.8f"
          % (E0, gap_exact, x2_exact))

    bins = np.linspace(-3, 3, 121)
    a_values = np.array([0.2, 0.16, 0.125, 0.1, 0.08, 0.0625])
    runs = []
    print("\n%6s %5s %6s %20s %20s %20s" % ("a", "M", "acc.", "E0 = 3<x^4>",
                                             "<x^2>", "E1 - E0"))
    for a in a_values:
        r = simulate(a, rng=rng, hist_bins=bins)
        r["E"], r["dE"] = [3 * v for v in mean_and_error(r["x4"])]
        r["X2"], r["dX2"] = mean_and_error(r["x2"])
        r["gap"], r["dgap"] = energy_gap(r["corr"], r["a"])
        runs.append(r)
        print("%6.3f %5d %6.2f %12.5f +- %.5f %12.5f +- %.5f %12.4f +- %.4f"
              % (r["a"], r["M"], r["acceptance"], r["E"], r["dE"], r["X2"],
                 r["dX2"], r["gap"], r["dgap"]))

    a_used = np.array([r["a"] for r in runs])
    print("\nExtrapolation a -> 0 (fit c0 + c1 a^2 + c2 a^4):")
    results = {}
    for key, err, exact, label in (("E", "dE", E0, "E0"),
                                   ("X2", "dX2", x2_exact, "<x^2>"),
                                   ("gap", "dgap", gap_exact, "E1 - E0")):
        y = np.array([r[key] for r in runs])
        dy = np.array([r[err] for r in runs])
        val, dval, coef = extrapolate(a_used, y, dy)
        results[key] = (y, dy, val, dval, coef, exact)
        print("  %-8s = %.5f +- %.5f   exact %.5f   (deviation %.1f sigma)"
              % (label, val, dval, exact, (val - exact) / dval))

    # -- figure: extrapolation and ground state density ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    y, dy, val, dval, coef, exact = results["E"]
    a2 = np.linspace(0, a_used.max() ** 2 * 1.05, 50)
    axes[0].errorbar(a_used ** 2, y, dy, fmt="o", label="Monte Carlo")
    axes[0].plot(a2, coef[0] + coef[1] * a2 + coef[2] * a2 ** 2, "b--", lw=1,
                 label="fit $c_0+c_1a^2+c_2a^4$")
    axes[0].errorbar([0], [val], [dval], fmt="s", color="b",
                     label="extrapolated")
    axes[0].axhline(exact, color="r", lw=1, label="matrix mechanics")
    axes[0].set(xlabel="$a^2$", ylabel="$E_0 = 3\\langle x^4\\rangle$",
                title="Ground state energy")
    axes[0].legend(fontsize=8)

    r = runs[-1]
    centers = (bins[1:] + bins[:-1]) / 2
    density = r["hist"] / (r["hist"].sum() * (bins[1] - bins[0]))
    axes[1].plot(centers, density, "o", ms=3, label="path histogram, a = %.2f" % r["a"])
    axes[1].plot(centers, osc.psi(0, centers) ** 2, "r", lw=1.2,
                 label=r"$|\psi_0(x)|^2$, matrix mechanics")
    axes[1].set(xlabel="$x$", ylabel="probability density",
                title="Ground state density")
    axes[1].legend(fontsize=8)

    lags = np.arange(r["corr"].shape[1]) * r["a"]
    C = r["corr"].mean(axis=0)
    axes[2].semilogy(lags, C, "o", ms=3, label="Monte Carlo")
    axes[2].semilogy(lags, C[0] * np.exp(-gap_exact * lags), "r", lw=1,
                     label=r"slope $E_1-E_0$ from matrix mechanics")
    axes[2].set(xlim=(0, 2.0), ylim=(C[0] * np.exp(-gap_exact * 2.2), C[0] * 1.5),
                xlabel=r"imaginary time $\tau$",
                ylabel=r"$\langle x(0)\,x(\tau)\rangle$",
                title="Correlation function")
    axes[2].legend(fontsize=8)
    fig.savefig(os.path.join(outdir, "path_integral.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("\nFigure written to", os.path.abspath(outdir))
    return results


if __name__ == "__main__":
    main()
