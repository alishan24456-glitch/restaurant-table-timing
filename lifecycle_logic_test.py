#!/usr/bin/env python3
"""
Logic test for the Restaurant Table Lifecycle Tracker.

This is NOT a video demo and renders nothing. It feeds scripted
Detection streams (the same objects the YOLOv8 detector produces)
straight into the LifecycleEngine and asserts the state machine walks
the full lifecycle. It exists because real restaurant footage rarely
contains a complete sit -> leave -> dirty -> cleaned cycle in one
short clip: the phases that real footage does not show are verified
here, and labelled as exactly that.

Run:  python lifecycle_logic_test.py
Exit code 0 = all scenarios passed.

Scenarios:
  A  full cycle: party sits, leaves leftovers, worker cleans -> FREE
  B  clean leave: party sits, leaves nothing -> straight to FREE
  C  unfinished clean: worker leaves items behind (back to
     NEEDS_CLEANING), then the items disappear without a worker
     visit -> counted clear after the clear delay
  D  regression: a party already seated at t=0 still gets its full
     dwell (0.0 is a valid timestamp, not a missing value)

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

import math
import sys

from restaurant_table_lifecycle import (Detection, EventLog,
                                        LifecycleEngine)

FPS = 30.0
DT = 1.0 / FPS
ZONE = [("T1", 0.30, 0.30, 0.70, 0.80)]   # on a 1000x1000 frame


def person(tid, cx, cy):
    return Detection(cx - 40, cy - 80, cx + 40, cy + 80,
                     "person", 0.99, tid)


def cup(cx, cy):
    return Detection(cx - 10, cy - 10, cx + 10, cy + 10,
                     "cup", 0.90, -1)


def worker(t, t0):
    """A person pacing inside the zone (centroid always inside,
    speed well above the still limit)."""
    cx = 500 + 70 * math.sin(2 * math.pi * (t - t0) / 1.6)
    return person(9, cx, 550)


def run(script, t_end):
    engine = LifecycleEngine(1000, 1000, EventLog(None), ZONE)
    i = 0
    while i * DT <= t_end:
        t = i * DT
        engine.update(script(t), t)
        i += 1
    engine.finish(t_end)
    return engine


def transitions_of(engine):
    return [(f, to) for (_, _, f, to) in engine.transitions]


def times_of(engine, pair):
    return [t for (t, _, f, to) in engine.transitions if (f, to) == pair]


FAILURES = []


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILURES.append(name)


def approx(a, b, tol=0.4):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
def scenario_a():
    print("Scenario A -- full cycle (sit, leftovers, worker cleans)")
    def script(t):
        dets = []
        if 2.0 <= t < 13.0:
            dets += [cup(480, 620), cup(520, 640)]
        if 2.0 <= t < 8.0:
            dets += [person(1, 450, 550), person(2, 550, 550)]
        if 11.0 <= t < 14.0:
            dets.append(worker(t, 11.0))
        return dets
    eng = run(script, 17.0)
    seq = transitions_of(eng)
    check("transition sequence",
          seq == [("FREE", "OCCUPIED"), ("OCCUPIED", "NEEDS_CLEANING"),
                  ("NEEDS_CLEANING", "CLEANING"),
                  ("CLEANING", "FREE")], str(seq))
    t_occ = times_of(eng, ("FREE", "OCCUPIED"))
    t_dirty = times_of(eng, ("OCCUPIED", "NEEDS_CLEANING"))
    t_clean = times_of(eng, ("NEEDS_CLEANING", "CLEANING"))
    t_free = times_of(eng, ("CLEANING", "FREE"))
    check("occupied at ~3.5s", t_occ and approx(t_occ[0], 3.5), str(t_occ))
    check("dirty at ~10.0s", t_dirty and approx(t_dirty[0], 10.0),
          str(t_dirty))
    check("cleaning at ~11.5s", t_clean and approx(t_clean[0], 11.5),
          str(t_clean))
    check("free at ~15.0s", t_free and approx(t_free[0], 15.0), str(t_free))
    tb = eng.tables[0]
    check("one sitting, dwell ~6.0s",
          len(tb.sittings) == 1 and approx(tb.sittings[0][2], 6.0, 0.5),
          str(tb.sittings))
    check("one cleaning (wait ~1.5s, clean ~3.5s)",
          len(tb.cleanings) == 1 and tb.cleanings[0][1] is not None
          and approx(tb.cleanings[0][0], 1.5, 0.5)
          and approx(tb.cleanings[0][1], 3.5, 0.5), str(tb.cleanings))
    check("ends FREE", tb.state.value == "FREE", tb.state.value)


def scenario_b():
    print("Scenario B -- clean leave (nothing left behind)")
    def script(t):
        if 1.0 <= t < 6.0:
            return [person(1, 500, 550)]
        return []
    eng = run(script, 10.0)
    seq = transitions_of(eng)
    check("transition sequence",
          seq == [("FREE", "OCCUPIED"), ("OCCUPIED", "FREE")], str(seq))
    tb = eng.tables[0]
    check("one sitting, dwell ~5.0s",
          len(tb.sittings) == 1 and approx(tb.sittings[0][2], 5.0, 0.5),
          str(tb.sittings))
    check("no cleanings", tb.cleanings == [], str(tb.cleanings))
    check("ends FREE", tb.state.value == "FREE", tb.state.value)


def scenario_c():
    print("Scenario C -- worker leaves items; later self-clear")
    def script(t):
        dets = []
        if 1.0 <= t < 12.0:
            dets.append(cup(500, 620))
        if 1.0 <= t < 5.0:
            dets.append(person(1, 500, 550))
        if 8.0 <= t < 9.5:
            dets.append(worker(t, 8.0))
        return dets
    eng = run(script, 18.0)
    seq = transitions_of(eng)
    check("transition sequence",
          seq == [("FREE", "OCCUPIED"), ("OCCUPIED", "NEEDS_CLEANING"),
                  ("NEEDS_CLEANING", "CLEANING"),
                  ("CLEANING", "NEEDS_CLEANING"),
                  ("NEEDS_CLEANING", "FREE")], str(seq))
    tb = eng.tables[0]
    check("one self-clear cleaning record (clean time None)",
          len(tb.cleanings) == 1 and tb.cleanings[0][1] is None
          and tb.cleanings[0][0] > 3.0, str(tb.cleanings))
    check("ends FREE", tb.state.value == "FREE", tb.state.value)


def scenario_d():
    print("Scenario D -- party already seated at t=0 (dwell regression)")
    def script(t):
        if t < 5.0:
            return [person(1, 500, 550)]
        return []
    eng = run(script, 9.0)
    seq = transitions_of(eng)
    check("transition sequence",
          seq == [("FREE", "OCCUPIED"), ("OCCUPIED", "FREE")], str(seq))
    tb = eng.tables[0]
    check("sitting starts at 0.0, dwell ~5.0s",
          len(tb.sittings) == 1 and tb.sittings[0][0] == 0.0
          and approx(tb.sittings[0][2], 5.0, 0.5), str(tb.sittings))


def main():
    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
    print()
    if FAILURES:
        print("LOGIC TEST FAILED: %d check(s): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("LOGIC TEST PASSED: all lifecycle scenarios behave as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
