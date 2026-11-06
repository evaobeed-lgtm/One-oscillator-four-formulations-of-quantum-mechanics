#!/usr/bin/env python3
"""
Bohmian (pilot wave) trajectories in the anharmonic oscillator
==============================================================

The pilot wave formulation (thesis section 2.6) is reviewed in

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016,

but not simulated there. This program computes the trajectories for a
Gaussian wave packet released in the potential of H = p^2 + q^4.

In this picture a particle has a definite position x(t), guided by the
wave function through the velocity field

    dx/dt = j / rho = (1/m) Im( psi'(x, t) / psi(x, t) ) = 2 Im(psi'/psi)

(m = 1/2 for H = p^2 + V). The wave function is known exactly from matrix
mechanics, psi(x, t) = sum_n c_n exp(-i E_n t) psi_n(x), so the only
numerical step is integrating the trajectories.

Basis size. The packet's expansion coefficients fall off slowly with
energy (roughly like exp(-sqrt(E)/2), because a high eigenstate still
overlaps the Gaussian near its turning point), so a large basis (N = 240)
is used. With a smaller basis psi(x, t) no longer solves the Schroedinger
equation accurately and the equivariance test below degrades (to about
2e-5 for N = 120).

Test (equivariance). If the starting points are distributed as
|psi(x, 0)|^2, the trajectories stay distributed as |psi(x, t)|^2. In one
dimension trajectories cannot cross, so a trajectory that starts at the
q-quantile of the initial density must sit at the q-quantile at all later
times. This is checked directly.
"""

import os

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anharmonic import AnharmonicOscillator, position_matrix


def velocity_field(osc, coef):
    """Return f(t, x) = dx/dt for the wave packet with coefficients coef."""
    def f(t, x):
        psi, dpsi = osc.wave_packet(coef, x, t, derivative=True)
        return 2.0 * np.imag(dpsi / psi)
    return f


def quantile_positions(x_grid, density, q):
    """Positions where the cumulative distribution equals q."""
    cdf = np.cumsum(density) * (x_grid[1] - x_grid[0])
    cdf -= 0.5 * density * (x_grid[1] - x_grid[0])       # midpoint correction
    return np.interp(q, cdf / cdf[-1] * 1.0, x_grid)


def cumulative_at(x_grid, density, x):
    """Cumulative distribution of the density at the points x."""
    cdf = np.cumsum(density) * (x_grid[1] - x_grid[0])
    cdf -= 0.5 * density * (x_grid[1] - x_grid[0])
    return np.interp(x, x_grid, cdf / cdf[-1])


def classical_trajectory(x0, t_eval):
    """Classical motion in H = p^2 + x^4 from rest at x0:
    dx/dt = 2p, dp/dt = -4 x^3."""
    sol = solve_ivp(lambda t, z: [2 * z[1], -4 * z[0] ** 3], (0, t_eval[-1]),
                    [x0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y[0]


def main(outdir="figures", x0=1.0, tmax=4.0, n_traj=41, N=240):
    os.makedirs(outdir, exist_ok=True)
    osc = AnharmonicOscillator(N=N)
    coef = osc.coherent_coefficients(x0)
    print("Wave packet: Gaussian centred at x0 = %.1f" % x0)
    print("  norm of expansion  = %.12f" % np.sum(coef ** 2))
    print("  <H>                = %.6f" % np.sum(coef ** 2 * osc.E))

    x_grid = np.linspace(-6, 6, 12001)
    t_eval = np.linspace(0, tmax, 401)
    q = (np.arange(n_traj) + 0.5) / n_traj
    rho0 = np.abs(osc.wave_packet(coef, x_grid, 0.0)) ** 2
    start = quantile_positions(x_grid, rho0, q)

    sol = solve_ivp(velocity_field(osc, coef), (0, tmax), start,
                    t_eval=t_eval, method="DOP853", rtol=1e-10, atol=1e-12)
    traj = sol.y                                        # (n_traj, n_times)
    print("  integrated %d trajectories to t = %.1f (%d velocity evaluations)"
          % (n_traj, tmax, sol.nfev))

    # -- equivariance test -----------------------------------------------------
    print("\nEquivariance: trajectory k should stay at quantile (k + 1/2)/%d"
          % n_traj)
    print("%6s %22s %14s %14s" % ("t", "max quantile deviation",
                                   "<x> quantum", "<x> Bohm"))
    Q = position_matrix(osc.N, 1)
    density = np.empty((len(t_eval), len(x_grid)))
    worst = 0.0
    for i, t in enumerate(t_eval):
        density[i] = np.abs(osc.wave_packet(coef, x_grid, t)) ** 2
        dev = np.abs(cumulative_at(x_grid, density[i], traj[:, i]) - q).max()
        worst = max(worst, dev)
        if i % 50 == 0:
            b = osc.S @ (coef * np.exp(-1j * osc.E * t))
            x_qm = np.real(np.conj(b) @ Q @ b)
            print("%6.2f %22.2e %14.6f %14.6f"
                  % (t, dev, x_qm, traj[:, i].mean()))
    print("largest deviation over all %d times: %.2e" % (len(t_eval), worst))
    print("trajectories never cross: %s"
          % bool(np.all(np.diff(traj, axis=0) > 0)))

    # -- figure -------------------------------------------------------------------
    x_cl = classical_trajectory(x0, t_eval)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    sel = np.abs(x_grid) <= 3
    ax.pcolormesh(t_eval, x_grid[sel], density[:, sel].T, cmap="Greys",
                  shading="auto", rasterized=True)
    ax.plot(t_eval, traj.T, color="tab:blue", lw=0.6)
    ax.plot(t_eval, x_cl, "r--", lw=1.4, label="classical particle")
    ax.plot([], [], color="tab:blue", lw=0.8, label="Bohmian trajectories")
    ax.set(xlabel="time $t$", ylabel="$x$", ylim=(-3, 3),
           title=r"Trajectories on top of $|\psi(x,t)|^2$ (grey)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for t_show, color in ((0.0, "k"), (1.0, "tab:orange"), (3.0, "tab:green")):
        i = np.argmin(np.abs(t_eval - t_show))
        ax.plot(x_grid, density[i], color=color, lw=1.2,
                label=r"$|\psi|^2$, $t=%.0f$" % t_show)
        hist, edges = np.histogram(traj[:, i], bins=np.linspace(-3, 3, 31),
                                   density=True)
        ax.step(edges[:-1], hist, where="post", color=color, lw=0.7, alpha=0.7)
    ax.set(xlabel="$x$", ylabel="probability density", xlim=(-3, 3),
           title="Histogram of trajectories (steps) follows $|\\psi|^2$ (lines)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(outdir, "bohm_trajectories.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("\nFigure written to", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
