"""Phase-space engine and the UncertaintyCore rule."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Coulomb, Species
from emergent.interactions import UncertaintyCore
from emergent.phasespace import PhaseSpaceSystem, integrate

PROTON = Species("p", mass=1836.15267, charge=+1.0)
ELECTRON = Species("e", mass=1.0, charge=-1.0)


def hydrogen(alpha=5.0, extra=()):
    return PhaseSpaceSystem([PROTON, ELECTRON], np.array([0, 1] + list(extra)),
                            [Coulomb()], [UncertaintyCore(xi=1.0, alpha=alpha)])


@pytest.mark.parametrize("alpha", [2.0, 5.0, 20.0])
def test_uncertainty_derivatives_match_finite_differences(alpha, rng):
    rule = UncertaintyCore(alpha=alpha)
    mass = np.array([1836.15, 1.0, 1.0, 1836.15])
    q = rng.normal(size=(4, 3)) * 1.5
    p = rng.normal(size=(4, 3)) * 0.8
    _, gq, gp = rule.evaluate(q, p, mass)
    h = 1e-6
    for arr, grad in ((q, gq), (p, gp)):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + h
            vp = rule.evaluate(q, p, mass)[0]
            arr[idx] = old - h
            vm = rule.evaluate(q, p, mass)[0]
            arr[idx] = old
            num[idx] = (vp - vm) / (2 * h)
        assert np.allclose(grad, num, rtol=1e-5, atol=1e-7 * np.abs(num).max())


def test_uncertainty_rule_is_translation_and_boost_invariant(rng):
    # depends only on relative position and relative momentum
    rule = UncertaintyCore(alpha=5.0)
    mass = np.array([1836.15, 1.0, 1.0])
    q = rng.normal(size=(3, 3))
    p = rng.normal(size=(3, 3))
    v0 = rule.evaluate(q, p, mass)[0]
    shift = rng.normal(size=3)
    boost = rng.normal(size=3)  # same velocity for all: p_i += m_i * boost
    assert rule.evaluate(q + shift, p + mass[:, None] * boost, mass)[0] == pytest.approx(v0)


def test_wall_minimum_is_the_analytic_one():
    # For a static pair the energy minimum is at r·p = ξħ with
    # E* = −μ / (2 + 1/α), r* = (1 + 1/(2α)) / μ  (atomic units).
    alpha = 5.0
    sys_ = hydrogen(alpha)
    mu = PROTON.mass / (PROTON.mass + 1.0)
    r_star = (1 + 1 / (2 * alpha)) / mu
    q = np.array([[0, 0, 0], [r_star, 0, 0]], float)
    p_rel = 1.0 / r_star
    p = np.array([[-p_rel, 0, 0], [p_rel, 0, 0]], float)
    assert sys_.energy(q, p) == pytest.approx(-mu / (2 + 1 / alpha), rel=1e-12)
    dq, dp, _ = sys_.derivatives(q, p)
    assert np.abs(dq).max() < 1e-10 and np.abs(dp).max() < 1e-10


def test_hamiltonian_run_conserves_energy_and_momentum():
    sys_ = hydrogen(alpha=5.0)
    q0 = np.array([[0, 0, 0], [2.0, 0.3, 0]], float)
    p0 = np.array([[0, -0.6, 0.1], [0, 0.6, -0.1]], float)
    tr = integrate(sys_, q0, p0, dt=0.002, steps=5000, gamma=0.0, record_every=50)
    assert np.abs(tr.energy - tr.energy[0]).max() < 1e-6 * max(1.0, abs(tr.energy[0]))
    assert np.allclose(tr.p[-1].sum(0), p0.sum(0), atol=1e-10)


def test_friction_only_removes_energy():
    sys_ = hydrogen(alpha=5.0)
    q0 = np.array([[0, 0, 0], [3.0, 0, 0]], float)
    p0 = np.array([[0, -0.4, 0], [0, 0.4, 0]], float)
    tr = integrate(sys_, q0, p0, dt=0.002, steps=5000, gamma=0.3, record_every=10)
    assert np.all(np.diff(tr.energy) <= 1e-9)
