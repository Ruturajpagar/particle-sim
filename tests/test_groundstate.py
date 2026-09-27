"""Spin, the Pauli rule, and the ground-state finder."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Coulomb, PauliCore, Species, UncertaintyCore
from emergent.groundstate import find_ground_state, minimize_energy
from emergent.phasespace import PhaseSpaceSystem, integrate

PROTON = Species("p", mass=1836.15267, charge=+1.0)
ELECTRON = Species("e", mass=1.0, charge=-1.0)
ALPHA_NUC = Species("alpha", mass=7294.29954, charge=+2.0, fermion=False)


def test_pauli_acts_only_on_identical_same_spin_fermions():
    species = [PROTON, ELECTRON, ALPHA_NUC]
    kind = np.array([1, 1, 1, 0, 0, 2, 2])
    spin = np.array([+1, +1, -1, +1, +1, 0, 0])
    sys_ = PhaseSpaceSystem(species, kind, [Coulomb()], [PauliCore(xi=2.0)], spin=spin)
    m = sys_.momentum_rules[0].mask
    assert m[0, 1] and m[1, 0]            # electrons, same spin
    assert not m[0, 2]                    # electrons, opposite spin
    assert m[3, 4]                        # protons, same spin
    assert not m[0, 3]                    # electron vs proton: not identical
    assert not m[5, 6]                    # alpha particles are bosons
    assert not m.diagonal().any()


def test_pauli_has_no_effect_without_same_spin_pairs():
    q = np.random.default_rng(1).normal(size=(3, 3))
    p = np.random.default_rng(2).normal(size=(3, 3))
    args = ([PROTON, ELECTRON], np.array([0, 1, 1]), [Coulomb()])
    base = PhaseSpaceSystem(*args, [UncertaintyCore()], spin=np.array([0, 1, -1]))
    with_pauli = PhaseSpaceSystem(*args, [UncertaintyCore(), PauliCore(xi=3.0)],
                                  spin=np.array([0, 1, -1]))
    assert with_pauli.energy(q, p) == pytest.approx(base.energy(q, p))


def test_pauli_derivatives_match_finite_differences():
    rng = np.random.default_rng(3)
    sys_ = PhaseSpaceSystem([PROTON, ELECTRON], np.array([0, 1, 1, 1]), [Coulomb()],
                            [PauliCore(xi=2.5, alpha=5.0)], spin=np.array([1, 1, 1, -1]))
    q, p = rng.normal(size=(4, 3)) * 1.2, rng.normal(size=(4, 3))
    dq, dp, _ = sys_.derivatives(q, p)
    h = 1e-6
    for arr, grad in ((q, dq), (p, dp)):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + h
            ep = sys_.energy(q, p)
            arr[idx] = old - h
            em = sys_.energy(q, p)
            arr[idx] = old
            num[idx] = (ep - em) / (2 * h)
        assert np.allclose(grad, num, rtol=1e-5, atol=1e-6 * np.abs(num).max())


def hydrogen_start(rng):
    q = np.array([np.zeros(3), rng.normal(size=3) * 2])
    p = np.array([np.zeros(3), rng.normal(size=3)])
    return q, p


def test_minimizer_finds_the_analytic_hydrogen_ground_state():
    sys_ = PhaseSpaceSystem([PROTON, ELECTRON], np.array([0, 1]), [Coulomb()],
                            [UncertaintyCore(xi=1.0)])
    gs = find_ground_state(sys_, hydrogen_start, n_starts=4, alphas=(5.0, 10.0))
    mu = PROTON.mass / (PROTON.mass + 1)
    for a in (5.0, 10.0):
        assert gs.energy[a] == pytest.approx(-mu / (2 + 1 / a), rel=1e-7)
        assert gs.hits[a] == 4


def test_minimizer_agrees_with_friction_dynamics():
    # hydrogen settles quickly: the dynamics' end state is the minimizer's minimum
    sys_ = PhaseSpaceSystem([PROTON, ELECTRON], np.array([0, 1]), [Coulomb()],
                            [UncertaintyCore(xi=1.0, alpha=5.0)])
    q0 = np.array([[0, 0, 0], [3.0, 0.5, 0]], float)
    p0 = np.array([[0, -0.4, 0.1], [0, 0.4, -0.1]], float)
    tr = integrate(sys_, q0, p0, dt=0.002, steps=40000, gamma=0.3, record_every=40000)
    q, p = tr.q[-1], tr.p[-1]
    P = p.sum(0)  # friction slows the atom's overall drift only at rate γ/M: subtract it
    e_internal = sys_.energy(q, p) - float(P @ P) / (2 * sys_.mass.sum())
    e_min, *_ = minimize_energy(sys_, q, p)
    assert e_internal == pytest.approx(e_min, abs=1e-7)


def test_slow_dynamics_head_into_the_minimizer_basin():
    # helium has a soft electron-electron mode that friction damps slowly; the
    # dynamics keep descending toward the minimum the minimizer finds from there
    sys_ = PhaseSpaceSystem([ALPHA_NUC, ELECTRON], np.array([0, 1, 1]), [Coulomb()],
                            [UncertaintyCore(xi=1.0, alpha=5.0)], spin=np.array([0, 1, -1]))
    q0 = np.array([[0, 0, 0], [0.8, 0.1, 0], [-0.3, 0.7, 0.2]], float)
    p0 = np.array([[0, 0, 0], [0, 1.5, 0], [1.2, 0, 0.3]], float)
    tr = integrate(sys_, q0, p0, dt=0.0005, steps=60000, gamma=0.5, record_every=10000)
    assert np.all(np.diff(tr.energy) <= 1e-9)
    e_min, *_ = minimize_energy(sys_, tr.q[-1], tr.p[-1])
    assert e_min <= tr.energy[-1] and tr.energy[-1] - e_min < 1e-2
