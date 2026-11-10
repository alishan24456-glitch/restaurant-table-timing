# Restaurant Table Lifecycle Tracker (v2)

**v2 upgrade of Restaurant Table-Timing** (v1, 7 Nov 2026 — the simpler
occupancy timer, still included in this repo). Where v1 only knew FREE
and OCCUPIED, v2 follows every table through its **full lifecycle** as
an explicit per-table state machine over fixed table zones (ROIs):

```
FREE ──► OCCUPIED ──► NEEDS_CLEANING ──► CLEANING ──► FREE
            │                                              ▲
            └────── party leaves, nothing left behind ─────┘
```

- **FREE** — the table is empty and clear.
- **OCCUPIED** — people are seated at the table; a **dwell timer**
  watches how long the party has been sitting.
- **NEEDS_CLEANING** — the party has left **and leftover items are
  detected on the table**. If the party leaves and nothing is left
  behind, the table returns straight to FREE.
- **CLEANING** — while NEEDS_CLEANING, a person (a worker) is detected
  inside the table zone. When the worker is gone **and** no items
  remain, the system registers the table as clean — FREE again, ready
  for the next party. If the worker leaves but items remain, the table
  falls back to NEEDS_CLEANING; if the items disappear without any
  worker visit, the table is counted clear after a longer delay.

Every transition is written to a **timestamped event log**, and the
annotated video carries a live board plus a bottom **dashboard strip**
— one state-coloured cell per table with its state and timer (the
party's dwell while OCCUPIED, the waiting time while NEEDS_CLEANING,
the cleaning time while CLEANING).

> **Honest framing:** the machine sees boxes, not intentions.
> "Seated" means a tracked person whose centroid is inside the zone
> and nearly still; "leftovers" means YOLOv8 COCO item detections
> inside the zone (cup, bowl, bottle, wine glass, fork, knife, spoon,
> cell phone, book, handbag) — a proxy that only counts items the
> detector can actually see. "CLEANING" means a person is present in
> a dirty table's zone; the system cannot prove that person is staff,
> and during CLEANING a newly seated guest is not distinguished from
> the worker. No state ever changes on a single frame — every
> transition needs its evidence sustained over time.

## How each transition is decided

| Transition | Evidence (all sustained, never single-frame) |
|---|---|
| FREE → OCCUPIED | A still person (centroid speed ≤ 60 px/s at 1280 px wide) inside the zone for ≥ 1.5 s (`--occupy`); walkers passing through never count |
| OCCUPIED → NEEDS_CLEANING | No seated person for ≥ 2 s (`--leave`) **and** leftover items confirmed in the zone (items in ≥ 40% of the frames of the last 1 s) |
| OCCUPIED → FREE | No seated person for ≥ 2 s and **no** confirmed leftovers |
| NEEDS_CLEANING → CLEANING | Any person inside the zone for ≥ 0.5 s (`--worker`) |
| CLEANING → FREE | Person gone for ≥ 1 s and items no longer confirmed |
| CLEANING → NEEDS_CLEANING | Person gone for ≥ 1 s but items still confirmed |
| NEEDS_CLEANING → FREE | Items no longer confirmed for ≥ 3 s (`--clear`) with no worker visit — counted clear, and the log says so |

## Table zones — derived where possible, manual fallback documented

With `--zones auto` (the default) the video is sampled first and
YOLOv8 **'dining table'** detections are clustered; each stable
cluster (seen in ≥ 20% of sampled frames) becomes a zone — its median
tabletop box expanded by documented factors to cover the seats around
the table. `--zones derived` forces derived-only; `--zones manual`
uses the built-in ROIs; `--tables tables.json` supplies your own
(`[{"name":"T1","x1":0.03,"y1":0.55,"x2":0.25,"y2":0.93}]`, fractions
of the frame).

Small models miss far tables often, so in `auto` mode fewer than two
stable derived tables means the run falls back to the four manual
zones and **logs that it did**. On the bundled demo clip below,
derivation found exactly **1 stable table** (the foreground one,
stability 1.00 over 22 sampled frames, 36 detections) — the other
tables in the room are real but YOLOv8n does not fire 'dining table'
on them at this distance, so the demo run used the manual fallback
zones. That is the honest, expected behaviour, not a silent shortcut.

## Run it (Windows CMD)

```
pip install -r requirements.txt
python restaurant_table_lifecycle.py                 REM opens a video file chooser
python restaurant_table_lifecycle.py --source pick   REM same chooser, explicitly
python restaurant_table_lifecycle.py --source restaurant.mp4
python restaurant_table_lifecycle.py --source 0      REM webcam
python lifecycle_logic_test.py                       REM synthetic-stream logic test
```

Press **Q** to quit. The annotated video is written to `output.mp4`
(change with `--output`); the event log goes to `events.log` (change
with `--log`). Useful options: `--zones auto|derived|manual`,
`--tables tables.json`, `--occupy 1.5`, `--leave 2.0`, `--worker 0.5`,
`--clear 3.0`, `--conf 0.25`, `--imgsz 1280` (for small/far diners),
`--weights yolov8n.pt` (downloaded automatically by Ultralytics on
first use if missing), `--max-seconds 30`, `--no-display` for
headless runs.

The v1 timer still runs exactly as before:
`python restaurant_table_timing.py --source restaurant.mp4`.

## Verified runs

**Real footage — busy restaurant interior (Pexels, free license:
"Festive Indoor Dining Experience with Christmas Decor" by Ellysa
jordens, 1280×720, full clip 10.7 s / 321 frames, YOLOv8n at imgsz
1280, conf 0.25):** the demo video
(`restaurant-table-lifecycle-demo.mp4`) is this run — the code on
real footage, nothing rendered or simulated.

- Zone derivation: 22 sampled frames, 36 'dining table' detections,
  **1 stable cluster** → documented manual fallback (4 zones), logged
  in the event log header.
- **Transitions fired: exactly 4 — FREE → OCCUPIED ×4, all at 1.5 s**
  (T1 party of 1, T2 party of 4, T3 party of 3, T4 party of 6 at first
  confirmation; live seated counts move as track IDs fragment — 65
  person track IDs over the clip).
- **NEEDS_CLEANING, CLEANING and the return to FREE did NOT fire on
  this footage** — plainly: nobody leaves any table during the
  10.7 s clip and no table is cleaned in it, so there was nothing for
  those states to fire on. All four sittings (10.7 s dwell each) are
  cut off by the end of the footage, not by anyone leaving. Item
  detections *are* live in the demo (cups on T3/T4 read "items 1–2"
  on the board) — they simply never became *leftovers*, because
  their parties never left.
- Per-table state time: FREE 1.5 s, then OCCUPIED 9.2 s, for all four
  tables. A standing staff member crossing the T3 zone mid-clip is
  correctly never counted as seated.

**Logic test — the phases the footage does not contain**
(`lifecycle_logic_test.py`, synthetic Detection streams fed to the
same engine; renders nothing, and is labelled a logic test wherever
it is mentioned): **4 scenarios, 17 checks, all passing.**

- **A — full cycle:** party of 2 sits at 2 s → OCCUPIED at 3.5 s;
  leaves at 8 s with 2 cups behind → NEEDS_CLEANING at 10.0 s
  (dwell 6.0 s); a pacing worker enters at 11 s → CLEANING at 11.5 s;
  items removed, worker leaves → FREE at 15.0 s (wait 1.5 s,
  cleaning 3.5 s).
- **B — clean leave:** party leaves nothing behind → OCCUPIED →
  FREE directly, no cleaning record.
- **C — unfinished clean:** worker leaves with items still on the
  table → CLEANING → back to NEEDS_CLEANING; the items later
  disappear with no worker visit → counted clear → FREE.
- **D — regression:** a party already seated at t = 0 still gets its
  full dwell (guards a real bug found during the v2 build, where a
  0.0 s session start was read as "missing" and dwell collapsed
  to 0).

**v1 runs (for reference):** simulation 30 s / 900 frames, 5 sessions,
average stay 16.0 s, peak 4/4 occupied; real footage 10.7 s / 321
frames, 4/4 tables occupied throughout, 59 track IDs at conf 0.30.

## Files

- `restaurant_table_lifecycle.py` — the v2 system (single file:
  zone derivation, lifecycle engine, overlay, CLI)
- `lifecycle_logic_test.py` — synthetic-stream verification of the
  full state machine (no video, no detector)
- `restaurant_table_timing.py` — the v1 occupancy timer
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer
