# diffusion: findings

Script: `run_diffusion.py` · branch `phase4-diffusion` (from `0df78d6`) · 88 runs, 19 min on 4 cores.
Data: `chunks.csv` (every run), `summary.json` (extrapolations, pair-distance histograms).
Figure: `python plot_diffusion.py`. Units: atomic units, real masses.

![diffusion](diffusion.png)

## Why this step

Phase 3 (`results/wavepacket/FINDINGS.md`) left two holes, both caused by
assuming something about the wave: one Gaussian per particle made atoms
19–23% underbound, and independent clouds could not bind H⁻ (no
correlation). This step assumes nothing about the wave.

## Rules (`emergent/diffusion.py`), identical for every particle

The Schrödinger equation in imaginary time is a diffusion equation, so it
can be run with random walkers. Each walker is one configuration of *all*
particles, nuclei included.

1. **Diffusion.** Every particle takes a random step with diffusion constant
   ħ/2m. Light particles spread fast, heavy ones slowly.
2. **Branching.** A configuration is copied or removed with weight
   exp(−τ[V − E_ref]), where V is its Coulomb energy.

Constants: ħ, masses, charges. E_ref only holds the walker count near 2000.
τ is a numerical step, run at three values and extrapolated to zero. Every
particle of every walker starts at an independent random point; nothing is
placed near anything.

**Declared before running:** for particle sets with no two identical
same-spin fermions this is the exact non-relativistic ground state, so it is
compared with exact literature values for the same particles and masses.
The walkers carry no antisymmetry, so for H₂ with parallel spins and for
lithium the rules must give a state the Pauli principle forbids. Those two
were run to measure that gap.

## Results

Energies extrapolated to τ → 0, compared with exact values for the same
Hamiltonian:

| system | diffusion (Hartree) | exact | difference |
|---|---|---|---|
| H | −0.49965 ± 0.00086 | −0.49973 | 0.1σ |
| Ps (e⁺e⁻) | −0.2492 ± 0.0012 | −0.25 | 0.7σ |
| H⁻ | −0.5254 ± 0.0037 | −0.52745 | 0.5σ |
| He | −2.8933 ± 0.0039 | −2.90330 | 2.6σ (0.34%) |
| H₂⁺ | −0.5944 ± 0.0013 | −0.59714 | 2.2σ (0.47%) |
| H₂ | −1.1600 ± 0.0044 | −1.16403 | 0.9σ |

Derived quantities compared with measurement, and with Phase 3:

| quantity (eV) | Phase 3 wave packets | **Phase 4 diffusion** | measured |
|---|---|---|---|
| H ionization | 11.03 | **13.596 ± 0.023** | 13.598 |
| He binding (both electrons) | 61.5 | **78.73 ± 0.11** | 79.005 |
| Ps binding | not run | **6.78 ± 0.03** | 6.80 |
| H⁻ electron affinity | unbound | **0.70 ± 0.10** | 0.754 |
| H₂⁺ bond, D₀ | not run | **2.58 ± 0.04** | 2.651 |
| H₂ bond, D₀ | (Dₑ 2.61) | **4.37 ± 0.13** | 4.478 |
| H₂ with parallel spins | unbound ✓ | **bound 4.51 ± 0.12** ✗ | unbound |
| Li (3 electrons) | ionization 5.31 ✓ | **−8.57 Hartree** ✗ | −7.4775 |

D₀ is the bond energy measured from the lowest vibrational level. The
protons here are quantum particles too, so their zero-point motion is
included automatically, and D₀ can be compared with experiment directly.

### What emerged

- **Exact atoms from two rules.** Hydrogen comes out at 13.596 ± 0.023 eV
  and helium at 78.7 eV (measured 79.0), with no shape assumed and nothing
  fitted. The Phase 3 shape limit (19–23% underbinding) is gone.
- **H⁻ binds.** A second electron stays attached to hydrogen by 0.70 ±
  0.10 eV (measured 0.754). This is correlation: the two electrons avoid
  each other moment to moment. The e–e distance peaks near 7 a0, while
  each electron stays near the proton (right panel). Independent clouds
  (Phase 3) and Hartree–Fock theory cannot produce this.
- **Chemical bonds with zero-point motion.** H₂ binds by 4.37 ± 0.13 eV
  (measured 4.478) and H₂⁺ by 2.58 ± 0.04 eV (2.651). The two protons, which
  started at random points, settle at a fixed separation: the walker
  distribution peaks at 1.53 a0 (middle panel; real equilibrium 1.40 a0).
  These walkers are distributed as ψ₀, not |ψ₀|², which shifts and widens
  the peak, so 1.53 a0 is not a measured bond length.
- **Positronium** (an electron and a positron, equal masses) binds at
  6.78 eV (6.80), with the same rules and no special case.

### What the diffusion rules get wrong: Pauli

- **H₂ with parallel spins binds**, with 4.51 eV and the same proton
  distance as ordinary H₂ (dotted line, middle panel). In reality it falls
  apart. The wave-packet rules got this right.
- **Lithium collapses to one shell:** −8.57 Hartree against the real
  −7.4775, so it is overbound by about 30 eV. With no antisymmetry, the third
  electron joins the other two in the innermost region.

So the diffusion rules are complete for everything with at most one
electron per spin (H, H⁻, He, H₂⁺, H₂, Ps). They are missing antisymmetry,
which Phase 3 had. The missing rule is known: identical fermions change the
sign of the wave when exchanged. In a walker picture, that means walkers
carry a sign, and a walker and its exchanged copy cancel.

## Robustness notes (principle 4)

- **Time step.** τ ∈ {0.02, 0.01, 0.005}/c², where c is the largest |qᵢqⱼ|
  (1 for hydrogen-like sets, 2 for He, 3 for Li). The τ dependence is weak
  compared with the error bars (figure, left). The linear fits are
  consistent (χ² ≤ 2.5 for 1 degree of freedom) except hydrogen
  (χ² = 5.3), where the extrapolated value still agrees with exact.
- **Errors.** Errors from blocks within one run are 2–5× too small: the
  first and second halves of a run differ by 2–5 block errors, even for H
  and Ps. The quoted errors therefore use the spread between 4 independent
  runs (different seeds) wherever it is larger. With only 4 runs, the error
  bars are themselves uncertain by roughly ±40%.
- **Possible small bias.** 5 of the 6 exact-comparable energies lie above
  exact, with He (0.34%, 2.6σ) and H₂⁺ (0.47%, 2.2σ) the largest. That
  pattern suggests a small systematic bias toward underbinding. Candidates
  are population control with 2000 walkers, and slow equilibration of the
  heavy protons in H₂⁺. It is not resolved here.
- **Branching cap.** At most 4 copies per step. The cap was reached in fewer
  than 2 × 10⁻⁶ of branching events.
- **Heavy particles equilibrate slowly.** A proton's random walk covers 43× less
  distance than an electron's in the same time, so H₂ and H₂⁺ runs equilibrate for 60–100 a.u. The
  proton distributions from the full runs (4 seeds pooled) are smooth;
  short test runs with 10 a.u. of equilibration were visibly lumpy.

## What this means for the project

This is the first rule set in the project that contains no approximation
of the physics it claims to model: two rules, three kinds of constants, and
for Pauli-free particle sets the answer is the exact non-relativistic one.
Atoms, the negative ion, a one-electron bond, the covalent bond and
positronium all emerge within 0.1–0.5% of exact. Every earlier "hole" in
that class is closed.

What remains is the Pauli principle. Phase 3 had it and lacked
correlation; Phase 4 has correlation and lacks Pauli. The next rule is
antisymmetry expressed in the walk itself (signed walkers that cancel under
exchange of identical fermions). That is where the fermion "sign problem"
sits, so it is the hardest step so far. It needs no new constant, and
without it no atom beyond helium, and no chemistry beyond H₂, can come out
right.
