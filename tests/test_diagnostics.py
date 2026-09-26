"""Cluster detection, anomaly flags and replay export on hand-built systems."""
from __future__ import annotations

import base64
import json

import numpy as np

from emergent import (Coulomb, CoreRepulsion, DirectForceField, LangevinBAOAB,
                      ReflectingBox, Segment, Simulation, random_state)
from emergent import diagnostics as dg
from emergent.export import export_viewer

from .conftest import SPECIES, make_state

FIELD = DirectForceField([Coulomb(), CoreRepulsion()])


def scene():
    """At rest: one p+e pair, one p-e-p-e chain, one lone particle, plus a
    p+e pair flying apart fast enough to be unbound."""
    pos = [[0, 0], [1, 0],                          # bound pair
           [10, 0], [11, 0], [12, 0], [13, 0],      # chain
           [30, 30],                                # free
           [20, 10], [21, 10]]                      # fast, unbound
    kinds = [0, 1, 0, 1, 0, 1, 0, 0, 1]
    vel = np.zeros((9, 2))
    vel[8] = [5.0, 0.0]  # relative KE 0.5*mu*25 >> |U| ~ 0.875
    return make_state(pos, vel, kinds)


def test_bound_pairs_use_two_body_energy_and_distance():
    state = scene()
    adj = dg.bound_pairs(state, FIELD.compute(state), r_bond=2.0)
    assert adj[0, 1] and adj[1, 0]
    assert not adj[7, 8]                       # too fast to be bound
    assert adj[2, 3] and adj[3, 4] and adj[4, 5]
    assert not adj[2, 4]                       # like charges repel
    assert not adj[6].any()


def test_cluster_report_on_hand_built_scene():
    state = scene()
    fr = FIELD.compute(state)
    rep = dg.analyse_clusters(state, fr, dg.bound_pairs(state, fr, r_bond=2.0))
    assert rep.sizes == {1: 3, 2: 1, 4: 1}
    assert rep.n_clusters == 2 and rep.free_particles == 3 and rep.largest == 4
    assert rep.neutral_clusters == 2
    assert rep.compositions == {"1e+1p": 1, "2e+2p": 1}
    assert rep.mean_binding_per_particle < 0


def test_connected_components_on_known_graph():
    adj = np.zeros((7, 7), bool)
    for i, j in [(0, 3), (3, 5), (1, 2), (4, 4)]:
        adj[i, j] = adj[j, i] = True
    np.fill_diagonal(adj, False)
    labels = dg.connected_components(adj)
    assert labels.tolist() == [0, 1, 1, 0, 4, 0, 6]


def test_spatial_entropy_distinguishes_gas_from_condensate(rng):
    gas = make_state(rng.uniform(0, 30, size=(400, 2)))
    blob = make_state(rng.normal(15, 0.5, size=(400, 2)))
    assert dg.spatial_entropy(gas, 30.0) > 0.95
    assert dg.spatial_entropy(blob, 30.0) < 0.5


def test_anomaly_monitor_flags_drift_and_singularities(tmp_path):
    path = tmp_path / "anomalies.jsonl"
    mon = dg.AnomalyMonitor(str(path), energy_tol=1e-3)
    state = make_state([[0, 0], [0.01, 0]], kinds=[0, 1])
    fr = DirectForceField([Coulomb()], r_min=0.05).compute(state)
    mon.start_isolated_segment(energy=-1.0)
    mon.check(state, fr, energy=-1.01)
    kinds = {(e["category"], e["kind"]) for e in mon.events}
    assert ("numerical", "energy_drift") in kinds
    assert ("singularity", "close_approach") in kinds
    assert len(path.read_text().splitlines()) == len(mon.events)


def test_isolated_run_raises_no_anomalies(tmp_path, rng):
    state = random_state(SPECIES, [15, 15], dim=2, box=15.0, temperature=0.5, rng=rng)
    sim = Simulation(state, FIELD, ReflectingBox(15.0), LangevinBAOAB(0.0025, rng=rng),
                     log_path=str(tmp_path / "log.csv"),
                     anomaly_path=str(tmp_path / "a.jsonl"), log_every=50)
    sim.run([Segment("isolated", 1000, gamma=0.0)])
    assert sim.monitor.events == []
    assert (tmp_path / "log.csv").read_text().count("\n") == 1 + 1000 // 50


def test_replay_export_round_trips(tmp_path, rng):
    state = random_state(SPECIES, [5, 5], dim=2, box=10.0, temperature=0.5, rng=rng)
    sim = Simulation(state, FIELD, ReflectingBox(10.0), LangevinBAOAB(0.005, rng=rng),
                     log_path=str(tmp_path / "log.csv"),
                     anomaly_path=str(tmp_path / "a.jsonl"), log_every=20, record=True)
    sim.run([Segment("anneal", 100, gamma=0.5, t_start=0.5, t_end=0.1)])
    out = tmp_path / "replay.json"
    export_viewer(sim, str(out), "test", rules=["Coulomb", "CoreRepulsion"])

    d = json.loads(out.read_text())
    nf, n = d["frames"]["count"], d["meta"]["n"]
    assert nf == len(sim.frames) == 5
    pos = np.frombuffer(base64.b64decode(d["frames"]["pos"]), "<u2").reshape(nf, n, 2)
    last = pos[-1] / 65535 * d["meta"]["box"]
    assert np.abs(last - sim.frames[-1][0]).max() < 10.0 / 65535
    assert d["frames"]["bond_offsets"][-1] == sum(len(f[1]) for f in sim.frames)
    assert len(d["series"]["time"]) == nf


def test_paired_start_is_all_atoms(rng):
    from emergent import paired_state
    state = paired_state(SPECIES, [20, 20], dim=2, box=40.0, temperature=0.2, rng=rng)
    n = 20
    # each + particle has its partner exactly one unit away
    assert np.allclose(np.linalg.norm(state.pos[:n] - state.pos[n:], axis=1), 1.0)
    assert np.allclose((state.mass[:, None] * state.vel).sum(0), 0.0, atol=1e-12)
    assert np.all((state.pos > 0) & (state.pos < 40.0))
    fr = FIELD.compute(state)
    rep = dg.analyse_clusters(state, fr, dg.bound_pairs(state, fr, r_bond=2.0))
    assert rep.compositions == {"1e+1p": 20} and rep.free_particles == 0


def test_paired_start_rejects_unpairable_species(rng):
    import pytest
    from emergent import Species, paired_state
    with pytest.raises(ValueError, match="equal counts"):
        paired_state(SPECIES, [3, 4], 2, 20.0, 0.2, rng)
    like = [Species("a", 1.0, 1.0), Species("b", 1.0, 1.0)]
    with pytest.raises(ValueError, match="oppositely charged"):
        paired_state(like, [3, 3], 2, 20.0, 0.2, rng)
