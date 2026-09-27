"""Wave-packet rules: exact Gaussian results, antisymmetry, invariances."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Species
from emergent.wavepacket import WavePacketSystem

E = Species("e", 1.0, -1.0)
HEAVY1 = Species("heavy+1", 1e12, +1.0)      # effectively a fixed point nucleus
HEAVY2 = Species("heavy+2", 1e12, +2.0)
P = Species("p", 1836.15267, +1.0)
S_NUC = 1e-3                                 # tiny width for the heavy nucleus


def test_single_gaussian_hydrogen_is_exact():
    # E(α) = 3α/2 − 2√(2α/π), minimum −4/(3π) at α = 8/(9π)
    sys_ = WavePacketSystem([HEAVY1, E], [0, 1], [0, 1])
    s_e = 1 / np.sqrt(8 / (9 * np.pi))
    e = sys_.energy_of(np.zeros((2, 3)), np.array([S_NUC, s_e]))
    assert e == pytest.approx(-4 / (3 * np.pi), abs=1e-5)
    # and it is a minimum in the width
    for f in (0.97, 1.03):
        assert sys_.energy_of(np.zeros((2, 3)), np.array([S_NUC, s_e * f])) > e


def test_two_electron_same_gaussian_helium_is_exact():
    # E(α) = 3α − (4Z√2 − 2)√(α/π), minimum −(4Z√2 − 2)² / (12π)
    Z = 2.0
    sys_ = WavePacketSystem([HEAVY2, E], [0, 1, 1], [0, +1, -1])
    c = 4 * Z * np.sqrt(2) - 2
    alpha = (c / (6 * np.sqrt(np.pi))) ** 2
    s_e = 1 / np.sqrt(alpha)
    e = sys_.energy_of(np.zeros((3, 3)), np.array([S_NUC, s_e, s_e]))
    assert e == pytest.approx(-c ** 2 / (12 * np.pi), abs=1e-5)


def test_same_spin_electrons_cannot_share_a_packet():
    sys_ = WavePacketSystem([HEAVY2, E], [0, 1, 1], [0, +1, +1])
    assert sys_.energy_of(np.zeros((3, 3)), np.array([S_NUC, 1.0, 1.0])) >= 1e6


def test_spin_does_not_matter_for_far_apart_particles():
    R = np.array([[0, 0, 0], [0.3, 0, 0], [40, 0, 0], [40.2, 0.1, 0]], float)
    s = np.array([0.1, 1.2, 0.1, 1.1])
    same = WavePacketSystem([P, E], [0, 1, 0, 1], [+1, +1, -1, +1]).energy_of(R, s)
    opp = WavePacketSystem([P, E], [0, 1, 0, 1], [+1, +1, -1, -1]).energy_of(R, s)
    assert same == pytest.approx(opp, abs=1e-9)


def test_same_spin_pair_is_repelled_at_short_range():
    # at the same geometry, parallel spins cost more energy than opposite spins
    R = np.array([[0, 0, 0], [1.4, 0, 0], [0.5, 0.2, 0], [0.9, -0.2, 0]], float)
    s = np.array([0.1, 0.1, 1.2, 1.2])
    same = WavePacketSystem([P, E], [0, 0, 1, 1], [+1, -1, +1, +1]).energy_of(R, s)
    opp = WavePacketSystem([P, E], [0, 0, 1, 1], [+1, -1, +1, -1]).energy_of(R, s)
    assert same > opp


def test_energy_is_invariant_under_moves_and_swaps():
    rng = np.random.default_rng(0)
    sys_ = WavePacketSystem([P, E], [0, 0, 1, 1, 1], [+1, -1, +1, +1, -1])
    R = rng.normal(size=(5, 3))
    s = rng.uniform(0.3, 1.5, 5)
    e0 = sys_.energy_of(R, s)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert sys_.energy_of(R @ rot.T + rng.normal(size=3), s) == pytest.approx(e0, abs=1e-10)
    swap = [0, 1, 3, 2, 4]                    # exchange the two same-spin electrons
    assert sys_.energy_of(R[swap], s[swap]) == pytest.approx(e0, abs=1e-10)


def test_lone_packets_do_not_repel_themselves():
    # a single free electron at rest: only its confinement energy remains
    sys_ = WavePacketSystem([E], [0], [1])
    assert sys_.energy_of(np.zeros((1, 3)), np.array([2.0])) == pytest.approx(3 / (2 * 4.0))


def test_extreme_widths_give_a_high_energy_not_a_crash():
    sys_ = WavePacketSystem([P, E], [0, 1, 1], [+1, +1, +1])
    R = np.zeros((3, 3))
    for s in ([1e-300, 1.0, 1.0], [1.0, 1e300, 1e300], [np.nan, 1.0, 2.0]):
        assert sys_.energy_of(R, np.array(s)) >= 1e6
