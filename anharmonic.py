#!/usr/bin/env python3
"""
Reference solution of the anharmonic oscillator  H = p^2 + q^4
==============================================================

Shared by the path integral, Wigner and Bohm programs in this folder. It is
the matrix mechanics solution of

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016   (sections 4.1 and 6.1),

extended with the position-space wave functions, which the other
formulations need.

Units: hbar = 1. H = p^2 + V(q) corresponds to a particle of mass m = 1/2,
so the velocity is dx/dt = 2p. The basis states |n> are the eigenstates of
H0 = p^2 + q^2, with wave functions

    phi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)).
"""

import numpy as np
from scipy.linalg import eigh


def hamiltonian(N=100):
    """Matrix of H = p^2 + q^4 in the lowest N harmonic oscillator states.
    The matrix elements are those of the infinite matrices (thesis eq. 6.4
    for both parities), so only the truncation of the basis is approximate."""
    n = np.arange(N, dtype=float)
    Q20 = (2 * n + 1) / 2
    Q22 = (np.sqrt((n + 1) * (n + 2)) / 2)[:-2]
    Q40 = (6 * n * n + 6 * n + 3) / 4
    Q42 = ((4 * n + 6) * np.sqrt((n + 1) * (n + 2)) / 4)[:-2]
    Q44 = (np.sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4)[:-4]
    P2 = np.diag(Q20) - np.diag(Q22, 2) - np.diag(Q22, -2)
    Q4 = (np.diag(Q40) + np.diag(Q42, 2) + np.diag(Q42, -2)
          + np.diag(Q44, 4) + np.diag(Q44, -4))
    return P2 + Q4


def position_matrix(N, power=1):
    """Matrix of q (power=1) or q^2 (power=2) in the same basis."""
    n = np.arange(N, dtype=float)
    if power == 1:
        Q11 = np.sqrt((n[:-1] + 1) / 2)
        return np.diag(Q11, 1) + np.diag(Q11, -1)
    Q22 = (np.sqrt((n + 1) * (n + 2)) / 2)[:-2]
    return np.diag((2 * n + 1) / 2) + np.diag(Q22, 2) + np.diag(Q22, -2)


def oscillator_functions(x, N):
    """phi_n(x) for n = 0..N-1 by the stable three-term recurrence.
    Returns an array of shape x.shape + (N,)."""
    x = np.asarray(x, dtype=float)
    phi = np.empty(x.shape + (N,))
    phi[..., 0] = np.pi ** -0.25 * np.exp(-x * x / 2)
    if N > 1:
        phi[..., 1] = np.sqrt(2.0) * x * phi[..., 0]
    for n in range(1, N - 1):
        phi[..., n + 1] = (np.sqrt(2.0 / (n + 1)) * x * phi[..., n]
                           - np.sqrt(n / (n + 1.0)) * phi[..., n - 1])
    return phi


def oscillator_derivatives(x, N):
    """d(phi_n)/dx = sqrt(n/2) phi_{n-1} - sqrt((n+1)/2) phi_{n+1}."""
    phi = oscillator_functions(x, N + 1)
    n = np.arange(N, dtype=float)
    d = -np.sqrt((n + 1) / 2) * phi[..., 1:]
    d[..., 1:] += np.sqrt(n[1:] / 2) * phi[..., :-2]
    return d


class AnharmonicOscillator:
    """Eigenvalues E[n] and eigenvectors S[:, n] of H = p^2 + q^4."""

    def __init__(self, N=100):
        self.N = N
        self.E, self.S = eigh(hamiltonian(N))
        # fix the sign so that psi_n(x) is positive for large positive x
        for k in range(N):
            if self.S[np.nonzero(np.abs(self.S[:, k]) > 1e-8)[0][-1], k] < 0:
                self.S[:, k] *= -1

    def psi(self, n, x):
        """Eigenfunction psi_n(x)."""
        return oscillator_functions(x, self.N) @ self.S[:, n]

    def expectation(self, n, power):
        """<n| q^power |n> for power = 2 (or 4, from the virial theorem
        <q^4> = E/3, which holds because <p^2> = 2 <q^4>)."""
        if power == 2:
            v = self.S[:, n]
            return v @ position_matrix(self.N, 2) @ v
        if power == 4:
            return self.E[n] / 3
        raise ValueError("power must be 2 or 4")

    def coherent_coefficients(self, x0):
        """Expansion coefficients c_n = <n|psi> of the displaced Gaussian
        psi(x) = pi^(-1/4) exp(-(x - x0)^2 / 2) in the eigenstates of H."""
        alpha = x0 / np.sqrt(2.0)
        m = np.arange(self.N)
        log_fact = np.concatenate(([0.0], np.cumsum(np.log(m[1:]))))
        coh = np.exp(-alpha ** 2 / 2 + m * np.log(abs(alpha) + 1e-300)
                     - log_fact / 2) * np.sign(alpha) ** m
        return self.S.T @ coh

    def wave_packet(self, c, x, t, derivative=False):
        """psi(x, t) = sum_n c_n exp(-i E_n t) psi_n(x) (and optionally its
        x-derivative) for an array of positions x at one time t."""
        b = self.S @ (c * np.exp(-1j * self.E * t))     # in oscillator basis
        psi = oscillator_functions(x, self.N) @ b
        if not derivative:
            return psi
        return psi, oscillator_derivatives(x, self.N) @ b


if __name__ == "__main__":
    osc = AnharmonicOscillator()
    print("Lowest levels of H = p^2 + q^4:")
    print(np.array2string(osc.E[:6], precision=12))
    x = np.linspace(-8, 8, 4001)
    psi0 = osc.psi(0, x)
    print("norm of psi_0 on the grid: %.12f" % (np.sum(psi0 ** 2) * (x[1] - x[0])))
    print("<q^2> = %.10f, <q^4> = %.10f"
          % (osc.expectation(0, 2), osc.expectation(0, 4)))
