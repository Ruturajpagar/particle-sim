"""Audit the code against PRINCIPLES.md: only fundamental rules, no atom knowledge.

Runs in well under a second and prints a short report. It is wired as a
Claude Code UserPromptSubmit hook (.claude/settings.json), so the report is
in front of the assistant before every prompt; run it by hand any time:

    python tools/audit_principles.py          # human-readable report
    python tools/audit_principles.py --hook   # JSON for the hook

Checks
  1. Rule code names no emergent structure. In the rule modules, identifiers
     and non-docstring string literals must not mention atoms, bonds,
     molecules, shells, orbitals or any element. (Docstrings and comments may:
     they explain what the rules are expected to produce.)
  2. Rule code holds no measured reference values. The measured numbers we
     compare against live in run scripts and FINDINGS, never in emergent/.
  3. Every interaction class is listed in PRINCIPLES.md's rule-status table,
     so placeholders stay labelled.
  4. Starts carry no structure. Every run script must use random starts, and
     an experiment may use the pre-paired start only as a control next to a
     random start in the same sweep.
  5. The rule-status table has no "Placeholder" rule that is in active use by
     a current (Phase 3+) run script.
"""
from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "emergent"

# Modules that define physics or produce states. Observers (diagnostics, export)
# and the experiment runner may name structures: they only measure.
RULE_MODULES = ["state.py", "interactions.py", "forces.py", "integrators.py", "boundaries.py",
                "phasespace.py", "groundstate.py", "wavepacket.py", "diffusion.py", "antisymmetry.py"]
# sim.py runs the loop and also logs observations (bound-pair counts etc.);
# only its dynamics method is held to the rule-code standard.
LOOP_MODULE, LOOP_METHOD = "sim.py", "run"

FORBIDDEN = re.compile(
    r"atom|bond|molecul|shell|orbital|hydrogen|helium|lithium|positronium|"
    r"(?<![a-z])(h2|h3|he2|hminus|li|he|1s|2s|2p)(?![a-z0-9])",
    re.IGNORECASE)
# Measured values this project compares against (Hartree and eV). If one shows
# up in rule code, a result is being put in by hand.
MEASURED = ["0.527751", "0.527446", "2.903724", "2.903305", "7.478060", "7.4780",
            "1.174475", "1.164025", "0.597139", "13.598", "13.6057", "79.005",
            "203.48", "5.3917", "4.478", "4.747", "0.754195", "0.7542"]
ALLOWED_STATE_HELPERS = {"paired_state"}   # labelled control start, see check 4


def _docstring_nodes(tree: ast.AST) -> set[int]:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                lines.add(body[0].value.lineno)
    return lines


def check_rule_code(problems: list[str], notes: list[str]) -> None:
    scanned = 0
    for name in RULE_MODULES:
        path = PKG / name
        if not path.exists():
            continue
        scanned += 1
        src = path.read_text()
        doc_lines = _docstring_nodes(ast.parse(src))
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and tok.start[0] in doc_lines:
                continue
            if tok.type == tokenize.NAME and tok.string in ALLOWED_STATE_HELPERS:
                continue
            if tok.type in (tokenize.NAME, tokenize.STRING):
                m = FORBIDDEN.search(tok.string)
                if m:
                    problems.append(f"{name}:{tok.start[0]} rule code names a structure: "
                                    f"{tok.string[:40]!r} (matched {m.group(0)!r})")
            if tok.type == tokenize.NUMBER:
                for v in MEASURED:
                    if tok.string.lstrip("-").startswith(v):
                        problems.append(f"{name}:{tok.start[0]} measured value {tok.string} in rule code")
    loop = PKG / LOOP_MODULE
    if loop.exists():
        for node in ast.walk(ast.parse(loop.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == LOOP_METHOD:
                for sub in ast.walk(node):
                    ident = getattr(sub, "id", None) or getattr(sub, "attr", None)
                    if ident and FORBIDDEN.search(ident):
                        problems.append(f"{LOOP_MODULE}:{sub.lineno} observation {ident!r} "
                                        f"feeds into the dynamics loop")
    notes.append(f"scanned {scanned} rule modules + the dynamics loop")


def check_rule_table(problems: list[str], notes: list[str]) -> None:
    principles = (ROOT / "PRINCIPLES.md").read_text()
    classes = []
    for name in ("interactions.py", "wavepacket.py", "diffusion.py", "antisymmetry.py"):
        path = PKG / name
        if path.exists():
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ClassDef) and node.name != "PairInteraction" \
                        and not node.name.endswith("Result"):
                    classes.append((name, node.name))
    listed = {"WavePacketSystem": "Wave packets", "DiffusionSystem": "Diffusion",
              "PacketNode": "Antisymmetry"}
    for mod, cls in classes:
        key = listed.get(cls, cls)
        if not re.search(rf"^\| (\*\*)?`?{re.escape(key)}", principles, re.MULTILINE):
            problems.append(f"{mod}:{cls} is not in PRINCIPLES.md's rule-status table")
    placeholders = re.findall(r"^\| `(\w+)`[^|]*\| \*\*Placeholder", principles, re.MULTILINE)
    notes.append(f"{len(classes)} rule classes listed; placeholders: {', '.join(placeholders) or 'none'}")
    # A placeholder must not drive a current result.
    for script in sorted(ROOT.glob("run_*.py")):
        if script.name in ("run_phase1.py", "run_hydrogen.py"):
            continue   # historical: these runs exist to show the placeholder's effect
        text = script.read_text()
        for ph in placeholders:
            if re.search(rf"\b{ph}\s*\(", text):
                problems.append(f"{script.name} uses placeholder rule {ph}")


def check_starts(problems: list[str], notes: list[str]) -> None:
    random_scripts = []
    for script in sorted(ROOT.glob("run_*.py")):
        text = script.read_text()
        if "paired_state" in text:
            problems.append(f"{script.name} starts from pre-built pairs")
        if re.search(r"\bX0\s*=", text):
            problems.append(f"{script.name} hands the diffusion a prepared start (X0)")
        # ground_state / find_ground_state draw every start at random (checked below)
        if re.search(r"random_start|random_state|ground_state\(|run_diffusion\(|rng\.(uniform|normal)|"
                     r"init\s*=\s*['\"]random", text):
            random_scripts.append(script.name)
        else:
            problems.append(f"{script.name}: no random start found")
    for mod, fn in (("wavepacket.py", "ground_state"), ("groundstate.py", "find_ground_state"),
                    ("diffusion.py", "random_walkers"), ("diffusion.py", "run_diffusion")):
        path = PKG / mod
        if path.exists():
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.FunctionDef) and node.name == fn \
                        and "rng" not in ast.unparse(node):
                    problems.append(f"{mod}:{fn} does not draw its starts at random")
    for cfg in sorted((ROOT / "experiments").glob("*.toml")):
        text = cfg.read_text()
        body = "\n".join(l.split("#", 1)[0] for l in text.splitlines())
        if '"pairs"' in body and '"random"' not in body:
            problems.append(f"experiments/{cfg.name} uses the paired start without a random start")
    notes.append(f"random starts in {len(random_scripts)} run scripts")


def audit() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    for check in (check_rule_code, check_rule_table, check_starts):
        try:
            check(problems, notes)
        except Exception as exc:  # an audit bug must not hide as a pass
            problems.append(f"{check.__name__} could not run: {exc!r}")
    return problems, notes


def main() -> None:
    problems, notes = audit()
    status = "PASS" if not problems else f"FAIL ({len(problems)})"
    lines = [f"Principles audit: {status} — " + "; ".join(notes)]
    lines += [f"  - {p}" for p in problems[:20]]
    report = "\n".join(lines)
    if "--hook" in sys.argv:
        msg = (report + "\nReport this audit result briefly in your reply (user request: "
               "check only fundamental rules are in the system before every prompt).")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                 "additionalContext": msg}}))
    else:
        print(report)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
