#!/usr/bin/env python3
"""
Wigner phase space functions of the anharmonic oscillator
=========================================================

The Wigner formulation (thesis section 2.7) is reviewed in

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016,

but not simulated there. This program computes the Wigner function

    W(x, p) = (1/pi) * integral dy  psi*(x + y) psi(x - y) exp(2 i p y)

for eigenstates of H = p^2 + q^4 and for a moving wave packet, and checks
the properties that make W a "quasi-probability":

  * it is normalized,                  integral W dx dp = 1,
  * its marginal is the density,       integral W dp = |psi(x)|^2,
  * phase space averages give          integral W (p^2 + x^4) dx dp = <H>,
  * but it can be negative, which no classical probability can.

The wave functions come from the matrix mechanics solution in
anharmonic.py.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anharmonic import AnharmonicOscillator, oscillator_functions


class WignerGrid:
    """Phase space grid and the transform from wave function to W."""

    def __init__(self, osc, xmax=5.0, pmax=14.0, nx=201, np_=449, ymax=7.0,
                 ny=701):
        self.osc = osc
        self.x = np.linspace(-xmax, xmax, nx)
        self.p = np.linspace(-pmax, pmax, np_)
        self.y = np.linspace(-ymax, ymax, ny)
        self.dx, self.dp = self.x[1] - self.x[0], self.p[1] - self.p[0]
        dy = self.y[1] - self.y[0]
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        # oscillator functions at x + y and x - y: shape (nx, ny, N)
        self.phi_plus = oscillator_functions(X + Y, osc.N)
        self.phi_minus = oscillator_functions(X - Y, osc.N)
        # Fourier kernel exp(2 i p y) dy / pi: shape (ny, np)
        self.kernel = np.exp(2j * np.outer(self.y, self.p)) * dy / np.pi

    def wigner(self, b):
        """W(x, p) for a state with oscillator-basis coefficients b
        (complex allowed). Returns a real array of shape (nx, np)."""
        f = np.conj(self.phi_plus @ b) * (self.phi_minus @ b)   # (nx, ny)
        return np.real(f @ self.kernel)

    def eigenstate(self, n):
        return self.wigner(self.osc.S[:, n])

    def packet(self, c, t):
        return self.wigner(self.osc.S @ (c * np.exp(-1j * self.osc.E * t)))

    def checks(self, W, b):
        """Normalization, marginal, energy and negativity of W."""
        dA = self.dx * self.dp
        psi = oscillator_functions(self.x, self.osc.N) @ b
        X, P = np.meshgrid(self.x, self.p, indexing="ij")
        return dict(
            norm=W.sum() * dA,
            marginal_error=np.abs(W.sum(axis=1) * self.dp - np.abs(psi) ** 2).max(),
            energy=(W * (P ** 2 + X ** 4)).sum() * dA,
            minimum=W.min(),
            negative_volume=-W[W < 0].sum() * dA,
        )


def main(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    osc = AnharmonicOscillator(N=160)
    grid = WignerGrid(osc)

    print("Eigenstates of H = p^2 + q^4")
    print("%3s %12s %12s %12s %14s %12s %12s"
          % ("n", "norm", "marginal err", "<H> from W", "E_n (matrix)",
             "min W", "neg. volume"))
    states = (0, 1, 2, 5)
    W_eig = {}
    for n in states:
        W = grid.eigenstate(n)
        W_eig[n] = W
        c = grid.checks(W, osc.S[:, n])
        print("%3d %12.8f %12.1e %12.8f %14.8f %12.5f %12.6f"
              % (n, c["norm"], c["marginal_error"], c["energy"], osc.E[n],
                 c["minimum"], c["negative_volume"]))

    # Hudson's theorem: only Gaussian states have W >= 0 everywhere. The
    # ground state of q^4 is not Gaussian, so it must be slightly negative.
    # Compare with the harmonic oscillator ground state |0>.
    e0 = np.zeros(osc.N); e0[0] = 1.0
    c_h = grid.checks(grid.wigner(e0), e0)
    print("\nGround state negativity:")
    print("  harmonic (Gaussian) : min W = %+.2e, negative volume %.1e"
          % (c_h["minimum"], c_h["negative_volume"]))
    c_a = grid.checks(W_eig[0], osc.S[:, 0])
    print("  anharmonic          : min W = %+.2e, negative volume %.1e"
          % (c_a["minimum"], c_a["negative_volume"]))

    # -- moving wave packet -----------------------------------------------------
    x0 = 1.0
    coef = osc.coherent_coefficients(x0)
    energy = np.sum(coef ** 2 * osc.E)
    times = (0.0, 0.5, 1.5, 3.0)
    print("\nDisplaced Gaussian packet, x0 = %.1f, <H> = %.6f" % (x0, energy))
    print("%6s %12s %12s %12s %12s" % ("t", "norm", "<H> from W", "min W",
                                        "neg. volume"))
    W_t = {}
    for t in times:
        b = osc.S @ (coef * np.exp(-1j * osc.E * t))
        W = grid.wigner(b)
        W_t[t] = W
        c = grid.checks(W, b)
        print("%6.2f %12.8f %12.8f %12.5f %12.6f"
              % (t, c["norm"], c["energy"], c["minimum"], c["negative_volume"]))

    # -- figures --------------------------------------------------------------
    def panel(ax, W, title):
        v = np.abs(W).max()
        im = ax.pcolormesh(grid.x, grid.p, W.T, cmap="RdBu_r", vmin=-v, vmax=v,
                           shading="auto", rasterized=True)
        ax.set(xlabel="$x$", ylabel="$p$", title=title, xlim=(-3, 3),
               ylim=(-6, 6))
        return im

    fig, axes = plt.subplots(1, len(states), figsize=(4 * len(states), 3.8))
    for ax, n in zip(axes, states):
        im = panel(ax, W_eig[n], "eigenstate $n = %d$" % n)
        # classical orbit with the same energy, p^2 + x^4 = E_n
        xs = np.linspace(-osc.E[n] ** 0.25, osc.E[n] ** 0.25, 400)
        ps = np.sqrt(np.clip(osc.E[n] - xs ** 4, 0, None))
        ax.plot(xs, ps, "k--", lw=0.7)
        ax.plot(xs, -ps, "k--", lw=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Wigner functions of $H = p^2 + q^4$ "
                 "(blue = negative, dashed = classical orbit of the same energy)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wigner_eigenstates.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 3.8))
    xs = np.linspace(-energy ** 0.25, energy ** 0.25, 400)
    ps = np.sqrt(np.clip(energy - xs ** 4, 0, None))
    for ax, t in zip(axes, times):
        im = panel(ax, W_t[t], "wave packet, $t = %.1f$" % t)
        ax.plot(xs, ps, "k--", lw=0.7)
        ax.plot(xs, -ps, "k--", lw=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("A Gaussian packet in the quartic well: the phase space "
                 "distribution shears and develops negative fringes")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wigner_wave_packet.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("\nFigures written to", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
