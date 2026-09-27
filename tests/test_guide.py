"""Guided walkers: exact derivatives of the guide, and the energy is the guide-free one."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Species
from emergent.diffusion import DiffusionSystem
from emergent.guide import PacketGuide, run_guided
from emergent.wavepacket import WavePacketSystem

E = Species("e", 1.0, -1.0)
LI = Species("Li-7", 12786.39, +3.0)
HEAVY2 = Species("heavy+2", 1e9, +2.0, fermion=False)


def test_guide_derivatives_match_finite_differences():
    wp = WavePacketSystem([LI, E], [0, 1, 1, 1], [1, 1, -1, 1])
    rng = np.random.default_rng(0)
    g = PacketGuide(wp, wp.pack(rng.normal(size=(4, 3)) * 0.3, np.array([0.05, 0.7, 0.72, 3.0])))
    X = rng.normal(size=(4, 4, 3))
    _, l0, grad, lap, lapR = g.evaluate(X)
    L = lambda Z: g.evaluate(Z)[1]
    h = 1e-4
    num_g = np.zeros_like(grad)
    lapln = np.zeros_like(lap)
    for k in range(4):
        for c in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[:, k, c] += h
            Xm[:, k, c] -= h
            num_g[:, k, c] = (L(Xp) - L(Xm)) / (2 * h)
            lapln[:, k] += (L(Xp) - 2 * l0 + L(Xm)) / h ** 2
    num_R = sum((L(X + np.eye(3)[c] * h) - 2 * l0 + L(X - np.eye(3)[c] * h)) / h ** 2 for c in range(3))
    assert np.allclose(grad, num_g, rtol=1e-5, atol=1e-6)
    assert np.allclose(lap, lapln + (num_g ** 2).sum(-1), rtol=1e-4, atol=1e-4)
    assert np.allclose(lapR, num_R, rtol=1e-4, atol=1e-4)


def test_far_particles_do_not_underflow_to_a_node():
    wp = WavePacketSystem([LI, E], [0, 1, 1, 1], [1, 1, -1, 1])
    g = PacketGuide(wp, wp.pack(np.zeros((4, 3)), np.array([0.05, 0.7, 0.72, 3.0])))
    X = np.zeros((1, 4, 3))
    X[0, 2] = [0.4, 0, 0]
    X[0, 1] = [30.0, 0, 0]
    X[0, 3] = [0, 45.0, 0]
    sign, logabs, grad, lap, _ = g.evaluate(X)
    assert sign[0] != 0 and np.isfinite(logabs[0]) and np.all(np.isfinite(grad))


def test_a_poor_guide_still_gives_the_exact_energy():
    # He with a fixed nucleus; the guide is two identical broad Gaussians
    # (far from the real wave), yet the energy must be the exact −2.9037
    sp, kinds, spins = [HEAVY2, E], [0, 1, 1], [0, 1, -1]
    wp = WavePacketSystem(sp, kinds, spins)
    guide = PacketGuide(wp, wp.pack(np.zeros((3, 3)), np.array([0.01, 1.5, 1.5])))
    r = run_guided(DiffusionSystem(sp, kinds, spins), guide, tau=0.005, t_equil=10, t_measure=40,
                   n_target=600, seed=4)
    assert r.energy == pytest.approx(-2.9037, abs=0.03)
    assert r.acceptance > 0.95
