# Principles

The project's one rule about rules: **put in only fundamental rules, then let
structure emerge and compare it with reality.**

1. **Rules are universal.** A rule applies to every particle or every pair,
   selected only by properties the particles carry (mass, charge, spin, ...).
   No rule mentions atoms, bonds, molecules, or any other emergent structure.
2. **Constants are fixed before the run, from universal constants.** Values
   such as ħ, the elementary charge or particle masses are set in advance and
   never tuned to make an emergent outcome come out right. If a constant has
   to be tuned to match reality, that is itself a finding and gets reported.
3. **Agreement with reality is a prediction, never an input.** Comparisons
   with real measurements (binding energies, sizes, which molecules form) are
   made after the run and reported whether they agree or not.
4. **Numerical settings are not physics.** Time step, wall stiffness and
   similar controls must be shown not to change the result (convergence
   checks), or their effect must be reported.
5. **Known placeholders are labelled.** A rule that breaks these principles
   stays in the code only if it is clearly marked as a placeholder, with the
   reason it exists and what should replace it.

## Status of the current rules

| Rule | Status |
|---|---|
| `Coulomb` (k q_i q_j / r) | Fundamental. |
| `CoreRepulsion` (c / r^n) | **Placeholder.** c is chosen by hand, and it sets the size and binding energy of every bound pair. It stands in for quantum effects and is being replaced by `UncertaintyCore`. |
| `Yukawa` | Fundamental in form (massive mediator); unused so far. |
| `UncertaintyCore` (r · p ≥ ξħ for every pair) | Fundamental in intent: one universal rule, ξ = 1 and ħ = 1 in atomic units, fixed before any run. Its wall stiffness α is numerical and is checked for convergence. With it, hydrogen, He⁺ and positronium ground-state energies emerge within 0.03% of measurement (`results/hydrogen_emergence/`). ξ = 1 is a choice (the de Broglie condition), supported by that universality, not derived. |
| `PauliCore` (r · p ≥ ξ_P ħ for identical same-spin fermions) | **Not universal as it stands.** ξ_P has no first-principles value in this form. Scanning it: lithium alone would pick ξ_P ≈ 7, but at no value in 1–8 do H₂ with parallel spins or H₃ come out unbound (`results/multielectron/`). Not adopted; no value is fitted. |
| **Wave packets** (`emergent/wavepacket.py`): every particle a Gaussian wave of width s (kinetic 3ħ²/2ms²), Coulomb between charge clouds, Slater antisymmetry for identical same-spin fermions | **Current rule set.** Constants: ħ, masses, charges, spins; nothing else. Supersedes `UncertaintyCore` (uncertainty now follows from the wave) and `PauliCore` (antisymmetry has no strength to choose). Declared approximation: one Gaussian per particle. |
| Heat bath / friction | Stand-in for energy carried away by radiation. It removes energy only; it must not select what forms. |

## Known holes

| Hole | Evidence | Likely missing rule |
|---|---|---|
| ~~Pairwise uncertainty wall is a two-body stand-in for each particle's own spread; H₂ bound 3.8× too strongly.~~ **Resolved** by wave packets: H₂ bond 2.6 eV (measured 4.75), parallel-spin H₂, H₃ and He₂ unbound, Li ionization 5.31 eV (5.39). | `results/multielectron/`, `results/wavepacket/` | (was) per-particle width: done. |
| One Gaussian per particle cannot be the exact wave: atoms 19–23% underbound. | `results/wavepacket/FINDINGS.md` (fixed-nuclei control reproduces the analytic single-Gaussian values) | Freer wave shapes, or no assumed shape at all (imaginary-time diffusion). |
| Independent (product) clouds miss correlation: H⁻ does not bind; part of the H₂ bond is missing. Nuclei also pay ~0.5 eV each to localize independently. | `results/wavepacket/FINDINGS.md` | A correlated description of the whole system's wave. |
