# Emergent — bottom-up particle interaction simulator

Define only fundamental pair rules (charges, masses, force laws) and let
bound structures emerge. Diagnostics measure what forms; an anomaly monitor
flags where the rules or the numerics break down.

```
pip install -r requirements.txt
python run_phase1.py                  # 2D, 200 particles, ~3 min on 1 CPU
python run_phase1.py --dim 3
python run_phase1.py --no-core        # remove the short-range core -> classical collapse
```

Each run writes to `runs/<name>/`: `log.csv` (time series), `anomalies.jsonl`,
`final_state.npz`, and a `report.png` dashboard.

## Phase 2: quantum effects as fundamental rules

Phase 1's bound pairs got their size and energy from a hand-chosen core
constant. Phase 2 replaces that with one universal rule, the uncertainty
wall `UncertaintyCore` (r · p ≥ ħ for every pair, ξ = 1 fixed in advance),
integrated in phase space by `emergent/phasespace.py`. See `PRINCIPLES.md`
for what counts as a fundamental rule.

```
python run_hydrogen.py     # ~10 min on 4 cores
```

Result (`results/hydrogen_emergence/FINDINGS.md`): Coulomb alone collapses;
Coulomb + the uncertainty wall settles every random start into one ground
state, and the same rule predicts the binding energies of hydrogen, He⁺ and
positronium within 0.03% of measurement, with nothing fitted.

### Multi-electron systems and the Pauli rule

```
python run_multielectron.py     # ~3 min on 4 cores
```

Adds spin, a fermion/boson flag and `PauliCore` (identical same-spin
fermions only), and finds ground states with `emergent/groundstate.py`.
Result (`results/multielectron/FINDINGS.md`): with no free constants,
two-electron atoms reproduce Bohr's 1913 model exactly (helium 5% overbound,
as historically), but H₂ is bound 3.8× too strongly. That is a hole in the
pairwise uncertainty rule. The Pauli rule makes lithium grow a second shell
on its own, but no single Pauli strength gets lithium, H₂ with parallel spins
and H₃ right together.

## Phase 3: every particle is a wave packet

```
python run_wavepacket.py        # ~4 min on 4 cores
```

Each particle (nucleus or electron) is a Gaussian wave with its own width;
Coulomb acts between charge clouds; identical same-spin fermions are
antisymmetric. Only ħ, masses, charges and spins; every particle starts at a
random point. Result (`results/wavepacket/FINDINGS.md`): a covalent H₂ bond
forms (2.6 eV, measured 4.75), H₂ with parallel spins, H₃ and He₂ do not
bind, and lithium grows a second shell with ionization energy 5.31 eV
(measured 5.39), all with no constant to tune. Absolute atomic energies are
~20% too weak (the one-Gaussian shape), and H⁻ does not bind (no
correlation).

## Experiments

An experiment is a TOML file in `experiments/`: species, rules, protocol,
the parameters to sweep, and how many seeds. Every run is determined by
(config, parameter values, seed), and every parameter point uses the same
seeds so comparisons are not masked by different starting states.

```
python -m emergent.experiment experiments/atom_window.toml --dry-run        # list runs
python -m emergent.experiment experiments/atom_window.toml --step-factor 0.05  # quick check
python -m emergent.experiment experiments/atom_window.toml --workers 4
python plot_experiment.py results/atom_window
```

Outputs in `results/<name>/`: `results.csv` (one row per run),
`summary.csv` (mean / std / s.e.m. per parameter point), `provenance.json`
(resolved config, git commit, timing), `summary.png`, and per-run logs in
`runs/` (not committed; reproducible from the config).

A sweep parameter can drive several config paths at once:

```toml
[[sweep]]
name = "T_final"
paths = ["protocol.0.t_end", "protocol.1.t_start", "protocol.1.t_end"]
values = [0.05, 0.1, 0.2]
```

## Tests

```
pip install -e ".[dev]"
pytest -q
```

The suite checks the simulator against things that must be true before any
"discovery" can be trusted:

| Test | What it guarantees |
|---|---|
| `test_forces.py` | every force law is the exact negative gradient of its potential (2D and 3D); Newton's third law; Coulomb signs; the core puts the +/− minimum at r = 1; close approaches are counted, not hidden |
| `test_dynamics.py` | a +/− pair follows the exact Kepler ellipse and returns after one analytic period; energy error scales as dt²; many-body runs in open space conserve momentum and angular momentum; the heat bath reaches its target temperature for every species; walls conserve speed; same seed gives the same trajectory |
| `test_diagnostics.py` | bound-pair and cluster detection on hand-built scenes; entropy separates gas from condensate; anomaly flags fire correctly; clean isolated runs raise none; replay export round-trips |

When you add a new `PairInteraction`, add it to `RULESETS` in
`tests/test_forces.py` and the gradient and third-law tests cover it
automatically.

## Replay viewer

`viewer/` is a browser replay of recorded runs: particles, bound pairs,
cluster colouring and synced diagnostic charts. It plays back the Python
output frame by frame; the physics is not re-implemented in JavaScript.

```
python run_phase1.py --out runs/phase1_2d --export viewer/data/with_core.json
python run_phase1.py --no-core --out runs/no_core --export viewer/data/no_core.json
cd viewer && python -m http.server 8000     # then open http://localhost:8000
```

## Results so far

| | With core | Core removed |
|---|---|---|
| Structure | alternating +/− chains and networks (ionic-solid-like) | collapsed tight p+e pairs |
| Isolated-phase energy drift | ~3e-5 | ~1e-2 |
| Anomalies | none | `singularity:close_approach`, `numerical:energy_drift` |

![with core](docs/with_core.png)
![core removed](docs/no_core.png)

## Architecture

```
            ┌────────────────────── experiment script (run_phase1.py) ─────────────────────┐
            │  species + rules + protocol (anneal → isolated) + seed  = one reproducible run │
            └───────────────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
 state.py          interactions.py        forces.py            integrators.py     boundaries.py
 SoA arrays   ──►  PairInteraction   ──►  DirectForceField ──► LangevinBAOAB  ──► ReflectingBox
 pos/vel/kind      Coulomb, Core,         O(N²) exact; later   γ=0 ⇒ velocity
 mass/charge/…     Yukawa, …(plug-in)     cell list/BH/PME/GPU  Verlet (symplectic)
                                                    │
                                                    ▼
                          sim.py (emergence loop) ──► diagnostics.py (read-only observers)
                                                      energies, T, bound-pair graph, clusters,
                                                      binding energy, persistence, entropy,
                                                      AnomalyMonitor → anomalies.jsonl
```

Design rules:
1. **Physics lives only in `interactions.py`.** Nothing else knows about forces,
   and nothing anywhere mentions atoms or bonds.
2. **Observers never feed back into dynamics.** Diagnostics only measure.
3. **Every backend implements the same `compute(state)`**, so a fast backend can
   always be validated against the exact O(N²) one.
4. **Anomalies are categorised:** `numerical` means the simulator is wrong;
   `singularity` means the rules themselves break down.

## Units

Dimensionless: k = 1, k_B = 1, heavy mass = 10, light mass = 1, |q| = 1.
With the default core (c = 1/8, n = 8) the +/− pair potential has its minimum at
r = 1 with depth −0.875.
