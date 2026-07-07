"""Do positions jump between the walls, or slide?

The stability-vs-convergence scatters (ghost-lens artifact, section 5) show
two stable phases: input-held (L14 left wall) and output-converged (L23
bottom wall), with L20 mid-migration. This traces individual positions
through the cached per-layer ranks (out/pivot_data.json) and asks:

  - input axis:  last layer still held (ghost rank < 10) -> first layer
                 dissolved (rank > 1000): how many sampled layers in flight?
  - output axis: last layer unsettled (fwd rank of final prediction > 1000)
                 -> first layer locked (rank < 10, staying): ditto.
  - joint:       per position, does the input dissolve before, at, or after
                 the layer where the output locks?

Sharp jumps at heterogeneous layers = per-position phase flips; wide transits
at the same layers everywhere = a global rotation. Pure cached-data analysis:
  .venv/bin/python out/transition_analysis.py
"""

import json
from collections import Counter

HELD = 10       # ghost rank below this = on the held wall
DISSOLVED = 1000  # ghost rank above this = on the dissolved side
LOCKED = 10     # fwd rank of final prediction below this = converged

d = json.load(open("out/pivot_data.json"))
layers = d["layers"]
rows = d["rows"]
n_l = len(layers)


def input_transition(gr: list[int]) -> tuple[int, int] | None:
    """(release_idx, dissolve_idx) into `layers`, or None if the position
    never makes a wall-to-wall trip."""
    held = [i for i, r in enumerate(gr) if r < HELD]
    if not held:
        return None
    rel = max(held)
    for i in range(rel + 1, n_l):
        if gr[i] > DISSOLVED:
            return rel, i
    return None


def output_transition(fr: list[int]) -> tuple[int, int] | None:
    """(leave_idx, lock_idx): last layer with rank > DISSOLVED before the
    final locked stretch, first layer of that stretch (rank < LOCKED
    through the end)."""
    lock = None
    for i in range(n_l):
        if all(r < LOCKED for r in fr[i:]):
            lock = i
            break
    if lock is None or lock == 0:
        return None
    high = [i for i in range(lock) if fr[i] > DISSOLVED]
    if not high:
        return None
    return max(high), lock


def report(name: str, trips: list[tuple[int, int]]) -> None:
    transits = Counter(b - a - 1 for a, b in trips)
    total = len(trips)
    print(f"\n{name}: {total} wall-to-wall positions")
    for k in sorted(transits):
        gap = 3 * (k + 1)
        print(
            f"  {k} intermediate layers (<= {gap}-layer window): "
            f"{transits[k]:4d}  ({transits[k] / total:5.1%})"
        )
    start = Counter(layers[a] for a, _ in trips)
    end = Counter(layers[b] for _, b in trips)
    print("  departure layer:", {l: start.get(l, 0) for l in layers})
    print("  arrival layer:  ", {l: end.get(l, 0) for l in layers})


in_trips, out_trips, joint = [], [], []
for r in rows:
    it = input_transition(r["ghost_rank"])
    ot = output_transition(r["fwdpred_rank"])
    if it:
        in_trips.append(it)
    if ot:
        out_trips.append(ot)
    if it and ot:
        joint.append((it[1], ot[1]))  # dissolve idx vs lock idx

print(f"{len(rows)} positions, layers {layers}")
report("INPUT (held -> dissolved)", in_trips)
report("OUTPUT (unsettled -> locked)", out_trips)

order = Counter(
    "input dissolves first" if a < b else
    "same layer" if a == b else "output locks first"
    for a, b in joint
)
print(f"\nJOINT ({len(joint)} positions with both transitions):")
for k, v in order.most_common():
    print(f"  {k}: {v} ({v / len(joint):.1%})")
