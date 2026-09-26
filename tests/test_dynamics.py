"""Integrator validation against exact solutions and conservation laws."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import (Coulomb, CoreRepulsion, DirectForceField, LangevinBAOAB,
                      OpenSpace, ReflectingBox, Species, Yukawa, random_state)
from emergent import diagnostics as dg

from .conftest import SPECIES, angular_momentum_2d, make_state, momentum, total_energy

FAR = OpenSpace()  # no walls: a truly isolated system


def kepler_pair(a=3.0, e=0.5):
    """+/- pair at apoapsis of a bound Coulomb ellipse, COM at rest at origin.

    Relative motion obeys mu * r'' = -k / r^2, i.e. a Kepler problem with
    alpha = k / mu, period T = 2 pi sqrt(a^3 / alpha), energy E = -k / (2a).
    """
    m1, m2 = SPECIES[0].mass, SPECIES[1].mass
    mu, alpha = m1 * m2 / (m1 + m2), 1.0 / (m1 * m2 / (m1 + m2))
    r_a = a * (1 + e)
    v_a = np.sqrt(alpha * (1 - e) / (a * (1 + e)))
    rel_pos, rel_vel = np.array([r_a, 0.0]), np.array([0.0, v_a])
    pos = np.array([-m2 / (m1 + m2) * rel_pos, m1 / (m1 + m2) * rel_pos])
    vel = np.array([-m2 / (m1 + m2) * rel_vel, m1 / (m1 + m2) * rel_vel])
    period = 2 * np.pi * np.sqrt(a**3 / alpha)
    return make_state(pos, vel, kinds=[0, 1]), period, -1.0 / (2 * a), mu


def run(state, field, dt, steps, gamma=0.0, temperature=0.0, box=FAR, rng=None):
    integ = LangevinBAOAB(dt, gamma=gamma, temperature=temperature, rng=rng)
    fr = field.compute(state)
    for _ in range(steps):
        fr = integ.step(state, field, box, fr)
    return fr


def test_kepler_orbit_matches_exact_solution():
    state, period, exact_energy, _ = kepler_pair()
    field = DirectForceField([Coulomb()])
    start = state.pos.copy()
    assert total_energy(state, field) == pytest.approx(exact_energy, rel=1e-12)
    L0 = angular_momentum_2d(state)

    dt = 0.002
    run(state, field, dt, int(round(period / dt)))

    # back where it started after one analytic period (up to the dt remainder)
    assert np.abs(state.pos - start).max() < 5e-3
    assert total_energy(state, field) == pytest.approx(exact_energy, rel=1e-5)
    # velocity Verlet conserves angular momentum exactly for central forces
    assert angular_momentum_2d(state) == pytest.approx(L0, rel=1e-10)
    assert np.allclose(momentum(state), 0.0, atol=1e-12)


def test_energy_error_is_second_order_in_dt():
    field = DirectForceField([Coulomb()])
    errors = []
    for dt in (0.02, 0.01, 0.005):
        state, period, e0, _ = kepler_pair()
        integ = LangevinBAOAB(dt)
        fr = field.compute(state)
        worst = 0.0
        for _ in range(int(round(period / dt))):
            fr = integ.step(state, field, FAR, fr)
            worst = max(worst, abs(dg.kinetic_energy(state) + fr.potential - e0))
        errors.append(worst)
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(3.5 < r < 4.6 for r in ratios), (errors, ratios)


def test_many_body_conserves_momentum_and_angular_momentum(rng):
    species = [Species("p", 10.0, +1.0, strong=1.0), Species("e", 1.0, -1.0)]
    state = random_state(species, [8, 8], dim=2, box=8.0, temperature=0.3, rng=rng)
    field = DirectForceField([Coulomb(), CoreRepulsion(), Yukawa(lam=0.5)])
    p0, L0 = momentum(state), angular_momentum_2d(state)
    e0 = total_energy(state, field)

    run(state, field, dt=0.001, steps=2000)

    assert np.allclose(momentum(state), p0, atol=1e-10)
    assert angular_momentum_2d(state) == pytest.approx(L0, abs=1e-9)
    assert total_energy(state, field) == pytest.approx(e0, rel=1e-4)


def test_heat_bath_reaches_target_temperature(rng):
    # ideal gas: no interactions, so the bath alone sets the temperature
    species = [Species("heavy", 10.0), Species("light", 1.0)]
    state = random_state(species, [300, 300], dim=2, box=100.0, temperature=3.0,
                         rng=rng, min_sep=0.0)
    field = DirectForceField([])
    integ = LangevinBAOAB(0.01, gamma=2.0, temperature=0.7, rng=rng)
    box = ReflectingBox(100.0)
    fr = field.compute(state)
    samples = {"all": [], "heavy": [], "light": []}
    for step in range(3000):
        fr = integ.step(state, field, box, fr)
        if step >= 1000 and step % 10 == 0:
            for name, sel in (("all", slice(None)), ("heavy", state.kind == 0),
                              ("light", state.kind == 1)):
                m, v = state.mass[sel], state.vel[sel]
                samples[name].append((m[:, None] * v**2).sum() / (2 * len(m)))
    for name, series in samples.items():  # equipartition: each species at T
        assert np.mean(series) == pytest.approx(0.7, rel=0.03), name


def test_reflecting_walls_preserve_speed_and_energy():
    state = make_state([[0.05, 5.0], [9.9, 5.0]], [[-2.0, 0.3], [1.5, -0.4]], kinds=[0, 1])
    speeds = np.linalg.norm(state.vel, axis=1)
    run(state, DirectForceField([]), dt=0.01, steps=500, box=ReflectingBox(10.0))
    assert np.all((state.pos >= 0) & (state.pos <= 10.0))
    assert np.allclose(np.linalg.norm(state.vel, axis=1), speeds)


def test_same_seed_gives_identical_trajectory():
    def once():
        rng = np.random.default_rng(99)
        state = random_state(SPECIES, [10, 10], dim=2, box=10.0, temperature=1.0, rng=rng)
        run(state, DirectForceField([Coulomb(), CoreRepulsion()]), dt=0.005, steps=200,
            gamma=0.5, temperature=1.0, box=ReflectingBox(10.0), rng=rng)
        return state.pos
    assert np.array_equal(once(), once())
