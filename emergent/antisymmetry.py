"""Phase 5 rule: identical fermions are antisymmetric, inside the diffusion.

The rule. Exchanging two identical fermions with the same spin (same species,
same spin) changes the sign of the wave: ψ(…x_a…x_b…) = −ψ(…x_b…x_a…). It has
no strength and no constant.

Why the diffusion needs it. Walkers are positive, so diffusion alone finds
the lowest state of any symmetry, which is a Pauli-forbidden state whenever
two identical same-spin fermions are present (results/diffusion). An
antisymmetric wave has positive and negative regions, separated by the
surface where it is zero (the node).

How it enters the walk here (fixed node). Each walker carries the sign of the
region it starts in and is removed when it crosses the node. Within one
region the wave does not change sign, so diffusion there is exact. The
approximation is where the node lies. The exact node is unknown, and signed
walkers that find it by cancelling each other (the exact method) almost never
meet in 6–12 dimensions: that is the fermion sign problem.

Where the node comes from. It is taken from the wave that the Phase 3 rules
(emergent/wavepacket.py) find on their own from random starts: a Slater
determinant of Gaussian packets for each group of identical same-spin
fermions. Nothing about atoms is put in; the packets' centres and widths are
an output of those rules. Because a wrong node can only raise the energy, the
result is an upper bound, and its distance from the exact energy measures
how wrong the emergent node is.

Units: atomic units (ħ = e = m_e = 1).
"""
from __future__ import annotations

import numpy as np

from .wavepacket import WavePacketSystem


class PacketNode:
    """Sign and node distance of the antisymmetric packet wave.

    Only groups of two or more identical same-spin fermions have a node; every
    other factor of the packet wave is a positive Gaussian and cannot change
    sign.
    """

    def __init__(self, wp: WavePacketSystem, x: np.ndarray):
        R, s = wp.unpack(x)
        R = R - (wp.mass[:, None] * R).sum(0) / wp.mass.sum()    # centre-of-mass frame
        self.groups = [g for g in wp.groups if len(g) > 1]
        self.centres = [R[g] for g in self.groups]
        self.alphas = [1.0 / s[g] ** 2 for g in self.groups]
        self.scale = [np.sqrt(wp.mass[g[0]]) for g in self.groups]   # mass-weighted distance

    @property
    def active(self) -> bool:
        return bool(self.groups)

    @staticmethod
    def _matrix(Xg: np.ndarray, centres: np.ndarray, alphas: np.ndarray):
        # M[w, a, b] = exp(−α_a |x_b − R_a|²): packet a evaluated at particle b
        diff = Xg[:, None, :, :] - centres[None, :, None, :]           # (W, a, b, 3)
        expo = -alphas[None, :, None] * (diff ** 2).sum(-1)
        # rescale by the largest value per column, then per row: far from all
        # packets the raw values underflow to 0, which would read as a node.
        # Sign and |D|/|∇D| are unchanged by this rescaling.
        expo = expo - expo.max(1, keepdims=True)
        M = np.exp(expo - expo.max(2, keepdims=True))       # then per row, likewise
        dM = -2 * alphas[None, :, None, None] * diff * M[..., None]   # ∂M[a,b]/∂x_b at fixed scale
        return M, dM

    @staticmethod
    def _cofactors(M: np.ndarray) -> np.ndarray:
        W, k, _ = M.shape
        if k == 1:
            return np.ones_like(M)
        C = np.empty_like(M)
        for a in range(k):
            for b in range(k):
                minor = np.delete(np.delete(M, a, axis=1), b, axis=2)
                C[:, a, b] = (-1) ** (a + b) * np.linalg.det(minor)
        return C

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (sign, distance to the node) for walkers X of shape (W, n, 3).

        The distance is |D| / |∇D| per group in mass-weighted coordinates (the
        local linear estimate), minimised over groups. ∇D comes from cofactors,
        which stay finite on the node itself.
        """
        W = len(X)
        sign = np.ones(W)
        dist = np.full(W, np.inf)
        for g, R, a, sc in zip(self.groups, self.centres, self.alphas, self.scale):
            M, dM = self._matrix(X[:, g, :], R, a)
            C = self._cofactors(M)
            D = (M[:, :, 0] * C[:, :, 0]).sum(1)                       # expand along column 0
            grad = (C[..., None] * dM).sum(1)                          # (W, b, 3)
            grad2 = (grad ** 2).sum((1, 2)) / sc ** 2
            sign *= np.sign(D)
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.minimum(dist, np.where(grad2 > 0, np.abs(D) / np.sqrt(grad2), np.inf))
        return sign, dist
