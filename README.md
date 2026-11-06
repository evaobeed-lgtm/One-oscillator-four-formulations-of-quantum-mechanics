[README.md](https://github.com/user-attachments/files/34454841/README.md)


# One oscillator, four formulations of quantum mechanics

**Path integral, Wigner phase space and Bohmian trajectory simulations of the anharmonic oscillator, each checked against matrix mechanics.**

This is a follow-up to chapter 2 of

> Eva Obeed, *Numerical Modelling of Some Quantum Systems*, master thesis, NTNU, Trondheim, May 2016.

## The gap in the thesis

Chapter 2 of the thesis reviews many formulations of quantum mechanics "to some extent from the viewpoint of how a formulation can be modelled and solved numerically". Only two of them are then simulated: Heisenberg's matrix mechanics and Schrödinger's wave mechanics. Three are reviewed but never coded:

| Section | Formulation | Simulated in the thesis |
|---|---|---|
| 2.3 | Heisenberg matrix mechanics (1925) | yes |
| 2.4 | Schrödinger wave mechanics (1926) | yes |
| 2.6 | Bohm pilot wave theory (1927) | no |
| 2.7 | Wigner phase space (1932) | no |
| 2.8 | Feynman path integral (1948) | no |

This project simulates the missing three for the same system the thesis studies, the anharmonic oscillator

```
H = p² + q⁴
```

and uses the matrix mechanics solution as the reference. The formulations are different pictures of one theory, so they must give the same numbers. They do.

Units follow the thesis: ħ = 1, and H = p² + V(q) means mass m = ½.

## 1. Path integral Monte Carlo (`path_integral.py`)

In imaginary time the path integral becomes a statistical average over closed paths x(τ) with weight exp(−S),

```
S = Σ_i [ (x_{i+1} - x_i)² / (4a) + a x_i⁴ ]
```

where a is the time step. The paths are sampled with the Metropolis algorithm. One thousand independent paths are updated together with NumPy, and their spread gives the error bars. Results at six time steps are extrapolated to a = 0.

| Quantity | Path integral | Matrix mechanics | Deviation |
|---|---|---|---|
| Ground state energy E₀ = 3⟨x⁴⟩ | 1.0603 ± 0.0027 | 1.06036 | 0.0 σ |
| ⟨x²⟩ | 0.3622 ± 0.0005 | 0.36202 | 0.4 σ |
| Gap E₁ − E₀ | 2.694 ± 0.035 | 2.73931 | −1.3 σ |

![Path integral results](figures/path_integral.png)

The histogram of path positions also reproduces the ground state density |ψ₀(x)|² from matrix mechanics (middle panel), and the correlation ⟨x(0)x(τ)⟩ decays with the slope E₁ − E₀ (right panel).

The path integral gives three significant digits where matrix mechanics gives fourteen. Its strength lies elsewhere: the cost grows gently with the number of dimensions, where a matrix grows exponentially.

## 2. Wigner phase space functions (`wigner.py`)

The Wigner function

```
W(x, p) = (1/π) ∫ dy  ψ*(x + y) ψ(x − y) exp(2ipy)
```

represents a quantum state as a distribution in phase space. It is computed for eigenstates and for a moving wave packet, using wave functions from the matrix mechanics solution.

| State n | ∫W dx dp | ⟨H⟩ from W | Eₙ from matrix mechanics | Minimum of W |
|---|---|---|---|---|
| 0 | 1.00000000 | 1.06036209 | 1.06036209 | −0.00254 |
| 1 | 1.00000000 | 3.79967303 | 3.79967303 | −0.31831 |
| 2 | 1.00000000 | 7.45569794 | 7.45569794 | −0.17741 |
| 5 | 1.00000000 | 21.23837292 | 21.23837292 | −0.31831 |

The integral of W over p reproduces |ψ(x)|² to 10⁻¹⁴, and the phase space average of p² + x⁴ reproduces every energy to eight decimals.

![Wigner functions of eigenstates](figures/wigner_eigenstates.png)

**The ground state is slightly negative.** The Wigner function of the harmonic oscillator ground state is a Gaussian and positive everywhere (the computed minimum is zero to rounding, −2 × 10⁻¹⁶). The anharmonic ground state looks similar but dips to −0.0025, with a total negative volume of 0.0024. This is Hudson's theorem at work: only Gaussian states have a non-negative Wigner function, so even the ground state of q⁴ cannot be read as a classical probability distribution.

**A wave packet develops interference.** A Gaussian packet starts with a positive Wigner function. In the quartic well it shears along the classical orbit and grows negative fringes; the negative volume rises from 0 to 0.36 by t = 3.

![Wigner function of a moving wave packet](figures/wigner_wave_packet.png)

## 3. Bohmian trajectories (`bohm.py`)

In the pilot wave picture a particle has a definite position, guided by the wave function through

```
dx/dt = 2 Im( ψ′(x, t) / ψ(x, t) )
```

The wave function of a Gaussian packet released at x₀ = 1 is known from matrix mechanics, ψ(x, t) = Σ cₙ exp(−iEₙt) ψₙ(x). The trajectories are integrated with a high-order Runge-Kutta method.

![Bohmian trajectories](figures/bohm_trajectories.png)

**Test.** If the starting points follow |ψ(x, 0)|², the trajectories must follow |ψ(x, t)|² at all later times. In one dimension trajectories cannot cross, so the trajectory that starts at a given quantile of the density must stay at that quantile. For 41 trajectories followed to t = 4:

- the largest deviation from the expected quantile is 1.1 × 10⁻⁶ over 401 checked times,
- no two trajectories ever cross.

The trajectories look nothing like the classical one (dashed red line). They bunch where the density is high and swerve around its nodes, yet their statistics are exactly those of standard quantum mechanics.

## Usage

```bash
pip install -r requirements.txt
python anharmonic.py       # reference levels from matrix mechanics
python path_integral.py    # about 3 minutes
python wigner.py           # a few seconds
python bohm.py             # under a minute
```

Figures are written to `figures/`. The path integral run uses a fixed random seed, so the numbers above are reproducible.

## Files

| File | Content |
|---|---|
| `anharmonic.py` | matrix mechanics solution of H = p² + q⁴ and its wave functions; used by the other three |
| `path_integral.py` | Metropolis sampling, error analysis, extrapolation in the time step |
| `wigner.py` | Wigner transform on a phase space grid and its checks |
| `bohm.py` | velocity field, trajectory integration, quantile test |
| `figures/` | the four figures shown above |

## Limitations

- **Path integral.** The gap is the least precise result (1.3%), and its extrapolation sits 1.3 standard errors below the exact value. That is within normal statistical scatter, but a longer run would be needed to tighten it.
- **Wigner packet.** The moving packet has slowly decaying high-momentum tails. On the grid used, its norm is recovered to 5 × 10⁻⁵ and its energy to 0.2% (5.240 to 5.246 against 5.250). The eigenstates do not have this problem.
- **Bohm.** The packet needs a large basis (240 oscillator states), because its expansion in energy eigenstates falls off slowly. With 120 states the quantile test degrades to 2 × 10⁻⁵.
- All three simulations are one-dimensional. Trajectories that cannot cross, used in the Bohm test, are special to one dimension.
