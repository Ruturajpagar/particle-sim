"""Phase 4 rules: the whole system's wave, found by diffusion.

Phase 3 gave every particle its own Gaussian cloud. That fixed a shape in
advance and treated particles as independent, and both limits showed up in
the results. Here nothing about the wave is assumed. The Schrödinger equation
in imaginary time,

    ∂ψ/∂t = Σ_i (ħ²/2m_i) ∇_i² ψ − V ψ,

is a diffusion equation with a source term, so it can be run with random
walkers. Each walker is one full configuration of every particle (nuclei
included). The same two rules apply to every particle:

1. **Diffusion.** Every particle takes a Gaussian step with variance ħτ/m
   per axis (diffusion constant ħ/2m). Lighter particles spread faster.
2. **Branching.** A walker is copied or removed with weight
   exp(−τ[(V_old + V_new)/2 − E_ref]), where V is the Coulomb energy of the
   configuration. Low-energy configurations multiply; high-energy ones die.

After a long time the walkers are distributed as the ground-state wave ψ₀
itself, with no assumed shape and with every correlation between particles
included. The walker average of V then equals the ground-state energy
exactly (for the constant guide function used here, ∫ψ₀ H 1 = E₀ ∫ψ₀).

Constants: ħ, masses, charges. E_ref is population control only (it keeps
the walker count near its target), and τ is a numerical step that is
extrapolated to zero.

What this does *not* contain: antisymmetry. Walkers are points with positive
weight, so they converge to the lowest state of any symmetry, which for
identical same-spin fermions is a state the Pauli principle forbids.
Enforcing it needs signed walkers or an assumed node surface; neither is
done here. `DiffusionSystem.pauli_free` reports whether the result is
physical for a given set of particles.

Units: atomic units (ħ = e = m_e = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import Species


@dataclass
class DiffusionSystem:
    species: list[Species]
    kind: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        self.kind = np.asarray(self.kind)
        self.spin = np.asarray(self.spin)
        self.n = len(self.kind)
        self.mass = np.array([self.species[k].mass for k in self.kind], float)
        self.charge = np.array([self.species[k].charge for k in self.kind], float)
        self._i, self._j = np.triu_indices(self.n, 1)
        self._qq = self.charge[self._i] * self.charge[self._j]

    @property
    def pauli_free(self) -> bool:
        """True if no two particles are identical fermions with the same spin,
        so the positive-walker ground state is the physical one."""
        seen = set()
        for k, s in zip(self.kind, self.spin):
            if self.species[k].fermion:
                if (int(k), float(s)) in seen:
                    return False
                seen.add((int(k), float(s)))
        return True

    @property
    def coupling(self) -> float:
        """Largest |q_i q_j|: sets how fast the potential varies, so the step
        is scaled by 1/coupling² (a numerical choice, identical for any set)."""
        return float(np.max(np.abs(self._qq))) if len(self._qq) else 1.0

    def potential(self, X: np.ndarray) -> np.ndarray:
        """Coulomb energy of each walker; X has shape (walkers, n, 3)."""
        d = X[:, self._i, :] - X[:, self._j, :]
        r = np.sqrt((d * d).sum(-1))
        return (self._qq / r).sum(-1)


def random_walkers(system: DiffusionSystem, n_walkers: int, rng: np.random.Generator,
                   radius: float = 2.0) -> np.ndarray:
    """Every particle of every walker at an independent random point in a
    sphere. Nothing is placed near anything."""
    d = rng.normal(size=(n_walkers, system.n, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d * radius * rng.uniform(0, 1, size=(n_walkers, system.n, 1)) ** (1 / 3)


@dataclass
class DiffusionResult:
    tau: float
    energy: float              # walker average of V over the measurement
    error: float               # block standard error
    growth: float              # average reference energy (second estimator)
    growth_error: float
    walkers_mean: float
    capped_fraction: float     # branching events cut at the copy limit
    series: np.ndarray = field(repr=False)   # per-step walker-mean V
    X: np.ndarray = field(repr=False)        # final walkers
    pair_hist: np.ndarray | None = field(default=None, repr=False)  # (pairs, bins) counts


def run_diffusion(system: DiffusionSystem, tau: float, t_equil: float, t_measure: float,
                  n_target: int = 2000, seed: int = 0, block_time: float = 2.0,
                  control_time: float = 1.0, max_copies: int = 4,
                  X0: np.ndarray | None = None, hist_edges: np.ndarray | None = None,
                  hist_time: float = 1.0) -> DiffusionResult:
    """Run the two rules for t_equil + t_measure (atomic time units).

    `control_time` is how quickly E_ref pulls the population back to n_target;
    `max_copies` bounds one walker's copies per step (its use is reported, and
    it should vanish as τ → 0). If `hist_edges` is given, every pair's
    distance is histogrammed every `hist_time` during the measurement (an
    observation only; walkers are distributed as ψ₀, not |ψ₀|²).
    """
    rng = np.random.default_rng(seed)
    X = random_walkers(system, n_target, rng) if X0 is None else X0.copy()
    sigma = np.sqrt(tau / system.mass)[None, :, None]
    V = system.potential(X)
    e_ref = float(V.mean())
    n_equil = int(round(t_equil / tau))
    n_meas = int(round(t_measure / tau))
    series = np.empty(n_meas)
    refs = np.empty(n_meas)
    counts = np.empty(n_meas)
    capped = branched = 0
    hist_every = max(1, int(round(hist_time / tau)))
    hist = None if hist_edges is None else np.zeros((len(system._i), len(hist_edges) - 1))

    for step in range(n_equil + n_meas):
        X = X + sigma * rng.normal(size=X.shape)
        V_new = system.potential(X)
        w = np.exp(-tau * (0.5 * (V + V_new) - e_ref))
        copies = (w + rng.uniform(size=w.shape)).astype(int)
        over = copies > max_copies
        capped += int(over.sum())
        branched += len(copies)
        copies[over] = max_copies
        X = np.repeat(X, copies, axis=0)
        V = np.repeat(V_new, copies)
        if len(V) == 0:
            raise RuntimeError("population died out; lower tau or raise n_target")
        v_mean = float(V.mean())
        e_ref = v_mean - np.log(len(V) / n_target) / control_time
        k = step - n_equil
        if k >= 0:
            series[k] = v_mean
            refs[k] = e_ref
            counts[k] = len(V)
            if hist is not None and k % hist_every == 0:
                r = np.linalg.norm(X[:, system._i, :] - X[:, system._j, :], axis=-1)
                for p in range(r.shape[1]):
                    hist[p] += np.histogram(r[:, p], bins=hist_edges)[0]

    def block_mean(x: np.ndarray, wts: np.ndarray) -> tuple[float, float]:
        per = max(1, int(round(block_time / tau)))
        nb = len(x) // per
        if nb < 2:                      # too short for a block error
            return float(np.average(x, weights=wts)), float("nan")
        xb = (x[: nb * per] * wts[: nb * per]).reshape(nb, per).sum(1) / wts[: nb * per].reshape(nb, per).sum(1)
        return float(np.average(x, weights=wts)), float(xb.std(ddof=1) / np.sqrt(nb))

    e, err = block_mean(series, counts)
    g, gerr = block_mean(refs, np.ones_like(refs))
    return DiffusionResult(tau, e, err, g, gerr, float(counts.mean()),
                           capped / max(branched, 1), series, X, hist)


def extrapolate(taus: np.ndarray, energies: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """Weighted linear fit E(τ) = E₀ + bτ. Returns (E₀, error of E₀, slope b)."""
    A = np.vstack([np.ones_like(taus), taus]).T
    W = np.diag(1 / np.asarray(errors) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ A.T @ W @ energies
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1])
