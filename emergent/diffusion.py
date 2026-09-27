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
    node_kill_rate: float = 0.0  # walkers removed at the node per walker per unit time
    energy_pc: float = float("nan")         # population-control-corrected estimate (use this)
    error_pc: float = float("nan")


def run_diffusion(system: DiffusionSystem, tau: float, t_equil: float, t_measure: float,
                  n_target: int = 2000, seed: int = 0, block_time: float = 2.0,
                  control_time: float = 1.0, max_copies: int = 4,
                  X0: np.ndarray | None = None, hist_edges: np.ndarray | None = None,
                  hist_time: float = 1.0, node=None, remove_com: bool = False,
                  pc_window: float = 3.0, trial_time: float = 5.0) -> DiffusionResult:
    """Run the two rules for t_equil + t_measure (atomic time units).

    `control_time` is how quickly E_ref pulls the population back to n_target;
    `max_copies` bounds one walker's copies per step (its use is reported, and
    it should vanish as τ → 0). If `hist_edges` is given, every pair's
    distance is histogrammed every `hist_time` during the measurement (an
    observation only; walkers are distributed as ψ₀, not |ψ₀|²).

    `remove_com` keeps every walker's centre of mass at the origin. This is
    exact: V does not depend on it and its kinetic energy separates.

    `node` (an `antisymmetry.PacketNode`) applies the antisymmetry rule as a
    fixed node: each walker keeps the sign of the region it started in and is
    removed when it crosses the node; a walker that stays on its side is
    weighted by 1 − exp(−2 d_old d_new / τ), the chance it did not cross and
    return within the step (mass-weighted coordinates). With a
    node, V averaged over the walkers is no longer the energy (the node
    boundary adds a term); the loss rate at the node is added to V instead.

    Population control (E_ref) multiplies every walker by the same factor
    each step. With a finite population that feedback biases the energy by
    O(1/n_target), strongly here because unguided Coulomb weights fluctuate a
    lot. `energy_pc` undoes it: each step's estimate is weighted by the product
    of the inverse factors applied over the preceding `pc_window` time
    (Umrigar, Nightingale & Runge 1993). It is the estimate to report.
    """
    rng = np.random.default_rng(seed)
    X = random_walkers(system, n_target, rng) if X0 is None else X0.copy()
    m = system.mass[None, :, None]

    def centre(X):
        return X - (m * X).sum(1, keepdims=True) / m.sum() if remove_com else X

    X = centre(X)
    use_node = node is not None and node.active
    if use_node:
        label, dist = node.evaluate(X)
        keep = label != 0
        X, label, dist = X[keep], label[keep], dist[keep]
    sigma = np.sqrt(tau / system.mass)[None, :, None]
    V = system.potential(X)
    e_ref = float(V.mean())
    killed_total = 0.0
    walker_steps = 0
    n_equil = int(round(t_equil / tau))
    n_meas = int(round(t_measure / tau))
    n_total = n_equil + n_meas
    applied = np.empty(n_total)          # E_ref used in each step's weights
    local = np.empty(n_total)            # V average + node loss rate after each step
    sizes = np.empty(n_total)
    series = np.empty(n_meas)
    refs = np.empty(n_meas)
    counts = np.empty(n_meas)
    capped = branched = 0
    hist_every = max(1, int(round(hist_time / tau)))
    hist = None if hist_edges is None else np.zeros((len(system._i), len(hist_edges) - 1))

    for step in range(n_equil + n_meas):
        X = centre(X + sigma * rng.normal(size=X.shape))
        V_new = system.potential(X)
        w = np.exp(-tau * (0.5 * (V + V_new) - e_ref))
        applied[step] = e_ref
        kill_frac = 0.0
        if use_node:
            sgn, d_new = node.evaluate(X)
            # survive only if still on its own side, and weighted by the chance
            # of not having crossed and come back within the step
            survive = np.where(sgn == label, 1.0 - np.exp(-2 * dist * d_new / tau), 0.0)
            w *= survive
            dist = d_new
            kill_frac = float(1.0 - survive.mean())
            if step >= n_equil:
                killed_total += float((1.0 - survive).sum())
                walker_steps += len(w)
        copies = (w + rng.uniform(size=w.shape)).astype(int)
        over = copies > max_copies
        capped += int(over.sum())
        branched += len(copies)
        copies[over] = max_copies
        X = np.repeat(X, copies, axis=0)
        V = np.repeat(V_new, copies)
        if use_node:
            label = np.repeat(label, copies)
            dist = np.repeat(dist, copies)
        if len(V) == 0:
            raise RuntimeError("population died out; lower tau or raise n_target")
        v_mean = float(V.mean())
        local[step] = v_mean - np.log((1 - kill_frac) if kill_frac < 1 else 1e-300) / tau
        sizes[step] = len(V)
        # E_ref = slowly updated energy estimate (V average + node loss rate,
        # averaged over `trial_time`) minus a pull on the population size
        # (during equilibration it follows the instantaneous value, since the
        # start is far from stationary)
        if step < n_equil:
            e_trial = local[step]
        else:
            e_trial += (local[step] - e_trial) * min(1.0, tau / trial_time)
        e_ref = e_trial - np.log(len(V) / n_target) / control_time
        if len(V) > 2 * n_target:        # hard pull back if a transient overshoots
            e_ref -= np.log(len(V) / (2 * n_target)) / tau
        k = step - n_equil
        if k >= 0:
            series[k] = v_mean
            refs[k] = e_ref
            counts[k] = len(V)
            if hist is not None and k % hist_every == 0:
                r = np.linalg.norm(X[:, system._i, :] - X[:, system._j, :], axis=-1)
                for p in range(r.shape[1]):
                    hist[p] += np.histogram(r[:, p], bins=hist_edges)[0]

    per = max(1, int(round(block_time / tau)))
    block_mean = lambda x, wts: _block_mean(x, wts, per)
    e_pc, err_pc = pc_estimate(applied, local, sizes, n_equil, tau, pc_window, per)

    e, err = block_mean(series, counts)
    g, gerr = block_mean(refs, np.ones_like(refs))
    return DiffusionResult(tau, e, err, g, gerr, float(counts.mean()),
                           capped / max(branched, 1), series, X, hist,
                           killed_total / max(walker_steps, 1) / tau, e_pc, err_pc)


def _block_mean(x: np.ndarray, wts: np.ndarray, per: int) -> tuple[float, float]:
    """Weighted mean and its standard error from blocks of `per` steps."""
    nb = len(x) // per
    if nb < 2:                      # too short for a block error
        return float(np.average(x, weights=wts)), float("nan")
    xb = (x[: nb * per] * wts[: nb * per]).reshape(nb, per).sum(1) / wts[: nb * per].reshape(nb, per).sum(1)
    return float(np.average(x, weights=wts)), float(xb.std(ddof=1) / np.sqrt(nb))


def pc_estimate(applied: np.ndarray, local: np.ndarray, sizes: np.ndarray, n_equil: int,
                tau: float, window: float, per: int) -> tuple[float, float]:
    """Energy with the population-control feedback undone.

    `applied[t]` is the E_ref used in step t's weights, `local[t]` the energy
    estimate after step t and `sizes[t]` the walker count. Each measured step
    is weighted by the product of exp(−τ(E_ref − const)) over the preceding
    `window` time (Umrigar, Nightingale & Runge 1993).
    """
    L = max(1, int(round(window / tau)))
    logf = -tau * (applied - applied[n_equil:].mean())
    csum = np.concatenate([[0.0], np.cumsum(logf)])
    idx = np.arange(n_equil, len(applied))
    logW = csum[idx + 1] - csum[np.maximum(idx + 1 - L, 0)]
    return _block_mean(local[n_equil:], np.exp(logW - logW.max()) * sizes[n_equil:], per)


def extrapolate(taus: np.ndarray, energies: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """Weighted linear fit E(τ) = E₀ + bτ. Returns (E₀, error of E₀, slope b)."""
    A = np.vstack([np.ones_like(taus), taus]).T
    W = np.diag(1 / np.asarray(errors) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ A.T @ W @ energies
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1])
