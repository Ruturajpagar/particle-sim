"""Guided walkers (importance sampling): same rules, far less noise.

Plain diffusion (emergent/diffusion.py) is exact but noisy: walkers wander
blindly, multiply wildly near nuclei, and with a node they are killed at
random. Here each walker instead drifts along the gradient of a guide wave
ψ_G and samples f = ψ₀ψ_G. For a fixed node that is ψ_G's own node, the
energy does not depend on ψ_G at all (the mixed estimator ⟨E_L⟩ over f is
exactly E₀ of the node's region). ψ_G only changes the noise.

The guide contains nothing new:
  * the wave found by the Phase 3 rules from random starts (one Gaussian per
    particle, antisymmetrized per group of identical same-spin fermions),
    the same wave that supplies the node;
  * for every pair, the exact Kato cusp factor exp(u(r)), u = a r/(1 + b r),
    with a = μ_ij q_i q_j (halved for identical same-spin fermions). The
    slope a is a consequence of Coulomb + kinetic energy for any pair; b = 1
    only sets how fast it levels off (numerical, cannot change the energy).

Walker moves follow Umrigar, Nightingale & Runge (1993): drift-diffusion
with a limited drift near nodes and nuclei, a Metropolis accept/reject, no
crossing of the node, and branching on the local energy E_L = Hψ_G/ψ_G.

Units: atomic units (ħ = e = m_e = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .diffusion import DiffusionSystem, _block_mean, pc_estimate, random_walkers
from .wavepacket import WavePacketSystem


class PacketGuide:
    """ψ_G = Π_groups det(packets) × exp(Σ_pairs u(r_ij)), in the centre-of-mass frame."""

    def __init__(self, wp: WavePacketSystem, x: np.ndarray, b: float = 1.0):
        R, s = wp.unpack(x)
        R = R - (wp.mass[:, None] * R).sum(0) / wp.mass.sum()
        self.groups = list(wp.groups)
        self.centres = [R[g] for g in self.groups]
        self.alphas = [1.0 / s[g] ** 2 for g in self.groups]
        n = wp.n
        self.n = n
        self.mass = wp.mass
        self._i, self._j = np.triu_indices(n, 1)
        mu = wp.mass[self._i] * wp.mass[self._j] / (wp.mass[self._i] + wp.mass[self._j])
        a = mu * wp.charge[self._i] * wp.charge[self._j]
        same = np.zeros(n, int)
        for k, g in enumerate(self.groups):
            same[g] = k
        antisym = (same[self._i] == same[self._j]) & np.array(
            [len(self.groups[same[i]]) > 1 for i in self._i])
        self.a = np.where(antisym, a / 2, a)
        self.b = b

    @property
    def has_node(self) -> bool:
        return any(len(g) > 1 for g in self.groups)

    def evaluate(self, X: np.ndarray):
        """For walkers X (W, n, 3): sign, ln|ψ_G|, ∇_k ln|ψ_G| (W, n, 3), ∇²_k ψ_G/ψ_G (W, n)
        and ∇²_R ln ψ_G for a uniform translation R of all particles (W,).

        The packet matrix is rescaled by its largest entry per column, then per
        row, before the determinant, so far-out particles cannot underflow it
        to zero; every ratio returned is unchanged by that rescaling.
        """
        W = len(X)
        sign = np.ones(W)
        logabs = np.zeros(W)
        gF = np.zeros((W, self.n, 3))           # ∇ ln F  (F = product of determinants)
        lF = np.zeros((W, self.n))              # ∇²F / F
        lapR = np.zeros(W)                      # ∇²_R ln ψ under uniform translation
        for g, R, a in zip(self.groups, self.centres, self.alphas):
            diff = X[:, g, :][:, None, :, :] - R[None, :, None, :]     # (W, a, b, 3)
            d2 = (diff ** 2).sum(-1)
            expo = -a[None, :, None] * d2
            shift = expo.max(1)                                         # per particle b (column)
            expo = expo - shift[:, None, :]
            rshift = expo.max(2)                                        # then per packet a (row)
            M = np.exp(expo - rshift[:, :, None])                       # packet a at particle b, rescaled
            dM = -2 * a[None, :, None, None] * diff * M[..., None]
            lM = (4 * a[None, :, None] ** 2 * d2 - 6 * a[None, :, None]) * M
            det = np.linalg.det(M)
            sign *= np.sign(det)
            with np.errstate(divide="ignore"):
                logabs += np.log(np.abs(det)) + shift.sum(1) + rshift.sum(1)
            ok = det != 0
            inv = np.zeros_like(M)
            inv[ok] = np.linalg.inv(M[ok])
            gF[:, g, :] = np.einsum("wba,wabk->wbk", inv, dM)
            lF[:, g] = np.einsum("wba,wab->wb", inv, lM)
            # ∇²_R ln D = Σ_b ∇²_b D/D − Σ_bb' B_bb'·B_b'b,  B_bb' = Σ_a (M⁻¹)_ba ∇M_ab'
            B = np.einsum("wba,wack->wbck", inv, dM)
            lapR += lF[:, g].sum(1) - np.einsum("wbck,wcbk->w", B, B)
        # pair cusp factors
        d = X[:, self._i, :] - X[:, self._j, :]
        r = np.sqrt((d * d).sum(-1))
        br = 1 + self.b * r
        logabs += (self.a * r / br).sum(-1)
        up = self.a / br ** 2                   # u'
        upp = -2 * self.a * self.b / br ** 3    # u''
        vec = d * (up / r)[..., None]
        gU = np.zeros((W, self.n, 3))
        lU = np.zeros((W, self.n))
        np.add.at(gU, (slice(None), self._i), vec)
        np.add.at(gU, (slice(None), self._j), -vec)
        lap = upp + 2 * up / r
        np.add.at(lU, (slice(None), self._i), lap)
        np.add.at(lU, (slice(None), self._j), lap)
        grad = gF + gU
        lap_ratio = lF + 2 * (gF * gU).sum(-1) + lU + (gU * gU).sum(-1)
        return sign, logabs, grad, lap_ratio, lapR


@dataclass
class GuidedResult:
    tau: float
    energy: float              # population-control-corrected mixed estimate
    error: float
    energy_raw: float
    acceptance: float
    walkers_mean: float
    capped_fraction: float
    X: np.ndarray = field(repr=False)


def run_guided(system: DiffusionSystem, guide: PacketGuide, tau: float, t_equil: float,
               t_measure: float, n_target: int = 2000, seed: int = 0, block_time: float = 2.0,
               control_time: float = 1.0, trial_time: float = 5.0, pc_window: float = 3.0,
               max_copies: int = 4, drift_limit: float = 1.0) -> GuidedResult:
    """Guided diffusion with the node of `guide`, centre of mass removed (exact)."""
    rng = np.random.default_rng(seed)
    m = system.mass[None, :, None]
    sq = np.sqrt(system.mass)[None, :, None]

    def centre(X):
        return X - (m * X).sum(1, keepdims=True) / m.sum()

    M_tot = system.mass.sum()
    def state(X):
        sgn, la, grad, lap, lapR = guide.evaluate(X)
        EL = -(lap / (2 * system.mass[None, :])).sum(1) + system.potential(X)
        # remove the centre-of-mass part of the kinetic energy: the walkers
        # live in the space with the centre of mass fixed, so only the internal
        # Hamiltonian counts. T_cm ψ/ψ = −(|∇_R ln ψ|² + ∇²_R ln ψ) / 2M, with
        # ∇_R the uniform translation of every particle.
        gR = grad.sum(1)
        EL = EL + ((gR * gR).sum(-1) + lapR) / (2 * M_tot)
        u = grad / sq                                       # drift in mass-weighted coordinates
        # no drift along the (removed) centre-of-mass direction
        u = u - sq * (u * sq).sum(1, keepdims=True) / M_tot
        u2 = (u * u).sum((1, 2))[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(u2 > 0, (-1 + np.sqrt(1 + 2 * drift_limit * u2 * tau)) / (drift_limit * u2 * tau), 1.0)
        return sgn, la, u * f, EL

    X = centre(random_walkers(system, n_target, rng))
    sgn, la, ud, EL = state(X)
    keep = (sgn != 0) & np.isfinite(EL)
    X, sgn, la, ud, EL = X[keep], sgn[keep], la[keep], ud[keep], EL[keep]

    n_equil = int(round(t_equil / tau))
    n_total = n_equil + int(round(t_measure / tau))
    applied = np.empty(n_total)
    local = np.empty(n_total)
    sizes = np.empty(n_total)
    raw = np.empty(n_total)
    e_ref = e_trial = float(np.median(EL))
    acc_sum = acc_n = capped = branched = 0

    for step in range(n_total):
        xi = rng.normal(size=X.shape)
        Y = centre(X + (tau * ud + np.sqrt(tau) * xi) / sq)
        sY, lY, uY, ELY = state(Y)
        # Metropolis on f = ψ_G² with the drift-diffusion proposal, in mass-weighted coordinates
        fwd = ((Y - X) * sq - tau * ud) ** 2
        bwd = ((X - Y) * sq - tau * uY) ** 2
        log_ratio = 2 * (lY - la) - (bwd.sum((1, 2)) - fwd.sum((1, 2))) / (2 * tau)
        accept = (sY == sgn) & np.isfinite(ELY) & (np.log(rng.uniform(size=len(X))) < np.minimum(0, log_ratio))
        acc = float(accept.mean())
        acc_sum += acc
        acc_n += 1
        tau_eff = tau * max(acc, 1e-3)
        EL_new = np.where(accept, ELY, EL)
        w = np.exp(-tau_eff * (0.5 * (EL + EL_new) - e_ref))
        applied[step] = e_ref
        X = np.where(accept[:, None, None], Y, X)
        sgn = np.where(accept, sY, sgn)
        la = np.where(accept, lY, la)
        ud = np.where(accept[:, None, None], uY, ud)
        EL = EL_new
        copies = (w + rng.uniform(size=w.shape)).astype(int)
        over = copies > max_copies
        capped += int(over.sum())
        branched += len(copies)
        copies[over] = max_copies
        X, sgn, la, ud, EL = (np.repeat(a, copies, axis=0) for a in (X, sgn, la, ud, EL))
        if len(EL) == 0:
            raise RuntimeError("population died out")
        local[step] = raw[step] = float(EL.mean())
        sizes[step] = len(EL)
        if step < n_equil:
            e_trial = local[step]
        else:
            e_trial += (local[step] - e_trial) * min(1.0, tau / trial_time)
        e_ref = e_trial - np.log(len(EL) / n_target) / control_time
        if len(EL) > 2 * n_target:
            e_ref -= np.log(len(EL) / (2 * n_target)) / tau

    per = max(1, int(round(block_time / tau)))
    e_pc, err_pc = pc_estimate(applied, local, sizes, n_equil, tau, pc_window, per)
    e_raw, _ = _block_mean(raw[n_equil:], sizes[n_equil:], per)
    return GuidedResult(tau, e_pc, err_pc, e_raw, acc_sum / acc_n, float(sizes[n_equil:].mean()),
                        capped / max(branched, 1), X)
