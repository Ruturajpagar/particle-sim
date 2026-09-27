"""Experiment runner: config expansion, reproducibility, parallel == serial."""
from __future__ import annotations

import copy
import csv
import json

import pytest

from emergent import experiment as ex

TINY = {
    "name": "tiny",
    "seeds": 2,
    "base_seed": 7,
    "system": {"dim": 2, "box": 10.0, "dt": 0.005, "log_every": 20, "r_bond": 2.0},
    "species": [{"name": "p", "mass": 10.0, "charge": 1.0, "count": 6},
                {"name": "e", "mass": 1.0, "charge": -1.0, "count": 6}],
    "rules": [{"type": "Coulomb", "k": 1.0}, {"type": "CoreRepulsion", "c": 0.125, "n": 8}],
    "protocol": [{"name": "anneal", "steps": 60, "gamma": 0.5, "t_start": 1.0, "t_end": 0.2},
                 {"name": "isolated", "steps": 40, "gamma": 0.0}],
    "sweep": [{"name": "T", "paths": ["protocol.0.t_end"], "values": [0.2, 0.5]},
              {"name": "c", "paths": ["rules.1.c"], "values": [0.1, 0.2, 0.3]}],
}


def write_toml(path, cfg):
    """Minimal TOML writer for the TINY layout (keeps tests dependency-free)."""
    def val(v):
        return json.dumps(v) if not isinstance(v, bool) else str(v).lower()
    lines = [f"{k} = {val(v)}" for k, v in cfg.items() if not isinstance(v, (dict, list))]
    lines += ["[system]"] + [f"{k} = {val(v)}" for k, v in cfg["system"].items()]
    for section in ("species", "rules", "protocol", "sweep"):
        for item in cfg[section]:
            lines += [f"[[{section}]]"] + [f"{k} = {val(v)}" for k, v in item.items()]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_expand_builds_full_grid_with_shared_seeds():
    specs = ex.expand(TINY)
    assert len(specs) == 2 * 3 * 2
    assert {s.seed for s in specs} == {7, 8}
    assert len({s.run_id for s in specs}) == len(specs)
    one = next(s for s in specs if s.params == {"T": 0.5, "c": 0.3} and s.seed == 8)
    assert one.config["protocol"][0]["t_end"] == 0.5
    assert one.config["rules"][1]["c"] == 0.3
    assert "sweep" not in one.config
    assert TINY["rules"][1]["c"] == 0.125  # original untouched


def test_one_sweep_can_drive_several_paths():
    cfg = copy.deepcopy(TINY)
    cfg["sweep"] = [{"name": "T", "paths": ["protocol.0.t_end", "protocol.1.gamma"],
                     "values": [0.3]}]
    (spec, *_) = ex.expand(cfg)
    assert spec.config["protocol"][0]["t_end"] == 0.3
    assert spec.config["protocol"][1]["gamma"] == 0.3


def test_sweep_names_cannot_shadow_result_columns():
    # a sweep named like a metric would be overwritten by the measurement
    for name in ("atom_frac", "seed", "run_id"):
        cfg = copy.deepcopy(TINY)
        cfg["sweep"] = [{"name": name, "paths": ["protocol.0.t_end"], "values": [0.3]}]
        with pytest.raises(ValueError, match="clash"):
            ex.expand(cfg)


def test_result_rows_keep_swept_values(tmp_path):
    spec = next(s for s in ex.expand(TINY) if s.params == {"T": 0.5, "c": 0.2})
    row = ex.run_one((spec, str(tmp_path)))
    assert row["T"] == 0.5 and row["c"] == 0.2


def test_bad_sweep_path_and_unknown_rule_are_rejected(tmp_path):
    cfg = copy.deepcopy(TINY)
    cfg["sweep"] = [{"name": "x", "paths": ["rules.1.not_a_param"], "values": [1]}]
    with pytest.raises(KeyError):
        ex.expand(cfg)
    cfg = copy.deepcopy(TINY)
    cfg["rules"][0]["type"] = "Gravity"
    with pytest.raises(ValueError, match="unknown rule type"):
        ex.build(cfg, 0, str(tmp_path))


def test_same_config_and_seed_give_identical_metrics(tmp_path):
    spec = ex.expand(TINY)[3]
    a = ex.run_one((spec, str(tmp_path / "a")))
    b = ex.run_one((spec, str(tmp_path / "b")))
    a.pop("wall_s"), b.pop("wall_s")
    assert a == b


def test_parallel_results_match_serial(tmp_path):
    path = write_toml(tmp_path / "tiny.toml", TINY)
    serial, _ = ex.run_experiment(path, str(tmp_path / "s"), workers=1, progress=False)
    parallel, summary = ex.run_experiment(path, str(tmp_path / "p"), workers=3, progress=False)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    assert strip(serial) == strip(parallel)
    assert len(summary) == 6 and all(row["n"] == 2 for row in summary)


def test_outputs_and_provenance_are_written(tmp_path):
    path = write_toml(tmp_path / "tiny.toml", TINY)
    out = tmp_path / "out"
    ex.run_experiment(path, str(out), workers=2, step_factor=0.5, progress=False)
    with open(out / "results.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and {"atom_frac", "chain_frac", "seed", "T", "c"} <= set(rows[0])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["runs"] == 12 and prov["step_factor"] == 0.5
    assert prov["config"]["protocol"][0]["steps"] == 30
    assert (out / "summary.csv").exists()
    assert (out / "runs" / rows[0]["run_id"] / "log.csv").exists()


def test_shipped_experiment_configs_are_valid(tmp_path):
    import glob
    paths = glob.glob("experiments/*.toml")
    assert paths
    for p in paths:
        specs = ex.expand(ex.load_config(p))
        assert specs
        ex.build(specs[0].config, specs[0].seed, str(tmp_path))  # constructs without error


def test_settle_metric_reads_the_last_bath_stage():
    rows = ([{"segment": "anneal", "free": 100}] * 4
            + [{"segment": "hold", "free": f} for f in (80, 70, 60, 50, 40)]
            + [{"segment": "isolated", "free": 10}])
    # second half of the hold only: 60 -> 40 of 200; anneal and isolated ignored
    assert ex.settle_free_change(rows, 200) == pytest.approx(-0.1)
    flat = [{"segment": "hold", "free": 50}] * 6
    assert ex.settle_free_change(flat, 200) == 0.0
    assert ex.settle_free_change(flat[:2], 200) != ex.settle_free_change(flat[:2], 200)  # nan


def test_settle_metric_is_reported(tmp_path):
    cfg = copy.deepcopy(TINY)
    cfg["sweep"] = []
    cfg["seeds"] = 1
    (spec,) = ex.expand(cfg)
    row = ex.run_one((spec, str(tmp_path)))
    assert "settle_free_change" in row and -1.0 <= row["settle_free_change"] <= 1.0


def test_init_option_selects_starting_state(tmp_path):
    cfg = copy.deepcopy(TINY)
    cfg["system"]["init"] = "pairs"
    sim, _ = ex.build(cfg, 0, str(tmp_path / "p"))
    assert ex.measure(sim)["atom_frac"] == 1.0
    cfg["system"]["init"] = "lattice"
    with pytest.raises(ValueError, match="unknown init"):
        ex.build(cfg, 0, str(tmp_path / "x"))
