"""Antisymmetry rule: sign flips under exchange, node distance, Pauli enforced."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Species
from emergent.antisymmetry import PacketNode
from emergent.diffusion import DiffusionSystem, run_diffusion
from emergent.wavepacket import WavePacketSystem

E = Species("e", 1.0, -1.0)
HE = Species("He-4", 7294.29954, +2.0, fermion=False)


def concentric_node(spins=(0, +1, +1)):
    wp = WavePacketSystem([HE, E], [0, 1, 1], list(spins))
    x = wp.pack(np.zeros((3, 3)), np.array([0.1, 0.9, 5.0]))    # packets on one point, as Phase 3 finds
    return PacketNode(wp, x)


def test_exchanging_same_spin_electrons_flips_the_sign():
    node = concentric_node()
    X = np.random.default_rng(0).normal(size=(500, 3, 3))
    s, _ = node.evaluate(X)
    s_swapped, _ = node.evaluate(X[:, [0, 2, 1], :])
    assert np.all(s != 0)
    assert np.all(s_swapped == -s)


def test_opposite_spins_have_no_node():
    assert not concentric_node((0, +1, -1)).active


def test_concentric_packets_put_the_node_at_equal_distances():
    # det of two concentric Gaussians vanishes exactly where |r1| = |r2|
    node = concentric_node()
    rng = np.random.default_rng(1)
    X = np.zeros((200, 3, 3))
    u = rng.normal(size=(200, 2, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    r = rng.uniform(0.3, 2.0, size=(200, 1))
    X[:, 1] = u[:, 0] * r
    X[:, 2] = u[:, 1] * (r + 0.01)
    s_in, d = node.evaluate(X)
    X[:, 2] = u[:, 1] * (r - 0.01)
    s_out, _ = node.evaluate(X)
    assert np.all(s_in == -s_out)
    # linear distance estimate ≈ |Δr| / √2 = 0.00707 (radial node, unit masses)
    assert np.median(d) == pytest.approx(0.01 / np.sqrt(2), rel=0.05)


def test_the_rule_stops_parallel_spins_sharing_the_lowest_state():
    # without antisymmetry this set falls to the He ground state (−2.90);
    # with it, it must stay near the parallel-spin state (−2.175)
    sys_ = DiffusionSystem([HE, E], [0, 1, 1], [0, +1, +1])
    r = run_diffusion(sys_, tau=0.005, t_equil=5, t_measure=20, n_target=800, seed=2,
                      node=concentric_node(), remove_com=True)
    assert r.growth == pytest.approx(-2.175, abs=0.1)
    assert r.node_kill_rate > 0


def test_removing_the_centre_of_mass_keeps_it_at_zero():
    sys_ = DiffusionSystem([HE, E], [0, 1, 1], [0, +1, -1])
    r = run_diffusion(sys_, tau=0.005, t_equil=1, t_measure=1, n_target=200, seed=0, remove_com=True)
    com = (sys_.mass[None, :, None] * r.X).sum(1) / sys_.mass.sum()
    assert np.abs(com).max() < 1e-12
