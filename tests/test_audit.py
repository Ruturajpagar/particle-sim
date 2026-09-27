"""The principles audit passes on the code base and catches violations."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit_principles", Path(__file__).resolve().parent.parent / "tools" / "audit_principles.py")
audit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_mod)


def test_code_base_passes_the_principles_audit():
    problems, _ = audit_mod.audit()
    assert problems == []


def test_structure_names_are_caught_and_generic_names_are_not():
    for bad in ("atom_count", "bond_length", "n_shells", "hydrogen", "H2", "molecule_ids", "Li"):
        assert audit_mod.FORBIDDEN.search(bad), bad
    for ok in ("charge", "mass", "spin", "potential", "walkers", "width", "helper", "hist"):
        assert not audit_mod.FORBIDDEN.search(ok), ok
