"""Phase 3 rules: every particle is a wave packet.

Each particle i is a normalized spherical Gaussian wave

    φ_i(r) = (2α_i/π)^{3/4} exp(−α_i |r − R_i|²),   α_i = 1 / s_i²

with centre R_i and width s_i. The same three rules apply to every particle,
nucleus or electron alike, using only ħ, masses, charges and spins:

1. **Waves cost kinetic energy to confine.** A packet of width s has kinetic
   energy 3ħ²/(2 m s²), the Schrödinger kinetic energy of the wave. The
   uncertainty principle is no longer a separate rule: a Gaussian has
   Δx·Δp = ħ/2 exactly.
2. **Coulomb acts between charge clouds.** Two packets interact through
   their charge densities |φ|², which softens 1/r at short range.
3. **Identical fermions with the same spin are antisymmetric** (Pauli).
   Their packets form a Slater determinant, so the energy uses the inverse
   overlap matrix. This is the quantum rule itself: it has no adjustable
   strength, unlike the PauliCore wall.

Particles that are not antisymmetrized with each other (different species,
different spins, bosons) combine as a Hartree product.

Known approximation, stated up front: one Gaussian per particle is a
restricted wave shape. Real hydrogen's electron is exponential, not
Gaussian, so this model's hydrogen is expected at −4/(3π) = −0.4244 Hartree
instead of −0.5. It is a choice of building block, not a constant tuned to
any outcome, and it applies identically to every particle.

Units: atomic units (ħ = e = m_e = 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from .state import Species


def _erf_over_r(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    """erf(√a · r) / r, with its r → 0 limit 2√(a/π)."""
    small = r < 1e-10
    safe_r = np.where(small, 1.0, r)
    return np.where(small, 2.0 * np.sqrt(a / np.pi), erf(np.sqrt(a) * safe_r) / safe_r)


@dataclass
class WavePacketSystem:
    species: list[Species]
    kind: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        self.kind = np.asarray(self.kind)
        self.spin = np.asarray(self.spin)
        n = len(self.kind)
        self.n = n
        self.mass = np.array([self.species[k].mass for k in self.kind], float)
        self.charge = np.array([self.species[k].charge for k in self.kind], float)
        # Antisymmetrized groups: identical fermions (same species) with the
        # same spin. Every other particle is a group of its own.
        groups: dict[tuple, list[int]] = {}
        for i in range(n):
            sp = self.species[self.kind[i]]
            key = (int(self.kind[i]), float(self.spin[i])) if sp.fermion else ("single", i)
            groups.setdefault(key, []).append(i)
        self.groups = [np.array(g) for g in groups.values()]

    # -- parameters -------------------------------------------------------
    # x = [R (n×3) flattened, ln s (n)]; the log keeps widths positive.
    def pack(self, R: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.concatenate([R.ravel(), np.log(s)])

    def unpack(self, x: np.ndarray):
        R = x[: 3 * self.n].reshape(self.n, 3)
        s = np.exp(x[3 * self.n:])
        return R, s

    # -- energy -----------------------------------------------------------
    def energy(self, x: np.ndarray) -> float:
        R, s = self.unpack(x)
        return self.energy_of(R, s)

    def energy_of(self, R: np.ndarray, s: np.ndarray) -> float:
        a = 1.0 / s ** 2
        ai, aj = a[:, None], a[None, :]
        p = ai + aj                                   # product exponent
        mu = ai * aj / p
        d2 = ((R[:, None, :] - R[None, :, :]) ** 2).sum(-1)
        S = (2.0 * np.sqrt(ai * aj) / p) ** 1.5 * np.exp(-mu * d2)
        T = S * mu * (3.0 - 2.0 * mu * d2)            # ⟨φ_i|−½∇²|φ_j⟩ for unit mass
        P = (ai[..., None] * R[:, None, :] + aj[..., None] * R[None, :, :]) / p[..., None]

        # Density matrices: D = S⁻¹ within each antisymmetrized group.
        D = np.zeros((self.n, self.n))
        kinetic = 0.0
        for g in self.groups:
            Sg = S[np.ix_(g, g)]
            if len(g) == 1:
                Dg = np.array([[1.0]])
            else:
                if np.linalg.cond(Sg) > 1e12:        # packets coincide: forbidden
                    return 1e6
                Dg = np.linalg.inv(Sg)
            D[np.ix_(g, g)] = Dg
            kinetic += float((Dg * T[np.ix_(g, g)]).sum()) / self.mass[g[0]]

        # Two-body Coulomb integrals (ij|kl) between Gaussian products.
        rho = p[:, :, None, None] * p[None, None, :, :] / (p[:, :, None, None] + p[None, None, :, :])
        dPQ = np.linalg.norm(P[:, :, None, None, :] - P[None, None, :, :, :], axis=-1)
        eri = S[:, :, None, None] * S[None, None, :, :] * _erf_over_r(rho, dPQ)

        Dq = D * self.charge[:, None]                  # charge-weighted (rows share a group's charge)
        coulomb = 0.5 * np.einsum("ji,lk,ijkl->", Dq, Dq, eri)
        exchange = 0.0
        for g in self.groups:
            if len(g) > 1:
                Dg = D[np.ix_(g, g)]
                q2 = self.charge[g[0]] ** 2
                exchange += 0.5 * q2 * np.einsum("vm,sl,mslv->", Dg, Dg, eri[np.ix_(g, g, g, g)])
            else:                                       # a lone packet does not repel itself
                i = g[0]
                exchange += 0.5 * self.charge[i] ** 2 * eri[i, i, i, i]
        E = kinetic + coulomb - exchange
        return float(E) if np.isfinite(E) else 1e6

    def gradient(self, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Central finite differences (systems here have at most ~30 parameters)."""
        g = np.empty_like(x)
        for k in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            g[k] = (self.energy(xp) - self.energy(xm)) / (2 * h)
        return g


def random_start(system: WavePacketSystem, rng: np.random.Generator, radius: float = 2.0):
    """Every particle, whatever it is, starts at a random point in a sphere with
    a random width from the same range. Nothing is placed near anything."""
    n = system.n
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    R = d * radius * rng.uniform(0, 1, size=(n, 1)) ** (1 / 3)
    s = np.exp(rng.uniform(np.log(0.05), np.log(2.0), size=n))
    return system.pack(R, s)


def minimize_wavepacket(system: WavePacketSystem, x0: np.ndarray, maxiter: int = 3000):
    from scipy.optimize import minimize

    res = minimize(lambda x: (system.energy(x), system.gradient(x)), x0, jac=True,
                   method="L-BFGS-B", options={"gtol": 1e-7, "ftol": 1e-15, "maxiter": maxiter})
    return float(res.fun), res.x, bool(res.success)


def ground_state(system: WavePacketSystem, n_starts: int = 24, seed: int = 0, tol: float = 1e-6):
    """Lowest minimum over random starts. Returns (E, x, hits, capped, all energies)."""
    rng = np.random.default_rng(seed)
    found = []
    capped = 0
    for _ in range(n_starts):
        e, x, ok = minimize_wavepacket(system, random_start(system, rng))
        capped += not ok
        found.append((e, x))
    e_best, x_best = min(found, key=lambda t: t[0])
    hits = sum(abs(e - e_best) < tol * max(1.0, abs(e_best)) for e, _ in found)
    return e_best, x_best, hits, capped, sorted(e for e, _ in found)
