# Restaurant Table-Timing — Table-Occupancy Analytics

A restaurant analytics assistant that watches a fixed dining-room camera
and times how long each table stays occupied. It tracks every person with
YOLOv8 + ByteTrack and runs a live table board: each table reads **FREE**
or **OCCUPIED** with a running dwell timer, and every seating is written
to a timestamped session log (table, seated time, freed time, total stay)
with average-stay and longest-stay stats at the end of the run.

> **Honest framing:** occupancy is measured from person boxes and
> stillness only. It times "a person is seated at this table" — it cannot
> tell whether someone is eating, waiting for food, or just sitting, and
> a table with only a bag or a coat on it reads FREE. A person who walks
> through a table zone never counts: a table only becomes OCCUPIED after
> a still person is seen inside its zone for 1.5 s, and only becomes FREE
> after 3 s with nobody seated there.

## How occupancy is decided

| Step | What the pipeline actually measures |
|---|---|
| **Seated** | A tracked person's box centroid falls inside a table zone **and** their recent centroid speed is under 60 px/s (at 1280 px wide) — the camera signature of sitting still rather than walking past |
| **OCCUPIED** | A seated person is present continuously for ≥ 1.5 s (`--occupy`); the session start is the moment presence began |
| **FREE** | No seated person for ≥ 3 s (`--free`); the session end is the last frame a seated person was seen |
| **Session** | One seat → free spell per table, logged with its stay; the board shows live timers, seated counts and sessions so far |

Table zones are fixed ROIs (fractions of the frame). The four defaults
cover a 2×2 dining floor; pass `--tables tables.json` with your own zones
for a real camera (`[{"name":"T1","x1":0.03,"y1":0.55,"x2":0.25,"y2":0.93}]`).

## Run it (Windows CMD)

```
pip install -r requirements.txt
python restaurant_table_timing.py                  REM opens a video file chooser
python restaurant_table_timing.py --source pick    REM same chooser, explicitly
python restaurant_table_timing.py --source restaurant.mp4
python restaurant_table_timing.py --source 0       REM webcam
python restaurant_table_timing.py --simulate       REM rendered dining room
```

Press **Q** to quit. The annotated video is written to `output.mp4`
(change with `--output`); the session log goes to `sessions.log`
(change with `--log`). Useful options: `--tables tables.json`,
`--occupy 1.5`, `--free 3.0`, `--conf 0.30`, `--imgsz 1280` (for
small/far diners), `--weights yolov8n.pt` (downloaded automatically by
Ultralytics on first use if missing), `--max-seconds 30`,
`--no-display` for headless runs.

## Modes

- **Real mode** — YOLOv8 detects people; ByteTrack supplies track IDs;
  the engine counts a person as seated only when their centroid is
  inside a table zone and nearly stationary.
- **Simulation mode** (`--simulate`) — a dining room is rendered with
  OpenCV: four tables, a couple who sit ~17 s, a solo diner who stays
  ~27 s, a family of two who arrive mid-run, one table that turns over
  (first diner leaves, a second arrives), and a waiter who walks the
  room twice — including straight through a table zone — without ever
  counting as seated. The scene publishes person boxes derived from the
  drawn figures through a small centroid tracker, producing exactly the
  same detection stream as real mode. The timing engine only ever sees
  that stream — it never peeks at the scene's scripted ground truth.

## Verified demo runs

**Simulation (30 s, 900 frames):** 10 track IDs from 8 scripted figures
(the waiter re-enters twice, so he takes three IDs); **5 sessions**
across the four tables — T1 17.0 s, T2 27.0 s, T3 12.8 s, T4 11.2 s +
12.0 s (the turnover); **average stay 16.0 s, longest stay 27.0 s**,
total occupied time 80.0 s, peak **4/4 tables occupied**. The waiter's
two walks produced no occupancy at all, including his pass through the
T2 zone front at 133 px/s.

**Real footage — busy restaurant interior (Pexels, free license:
"Festive Indoor Dining Experience with Christmas Decor" by Ellysa
jordens, 1280×720, full clip 10.7 s / 321 frames, YOLOv8n at imgsz
1280):** **4/4 tables occupied for the whole clip** — one session per
table, each 10.7 s (the clip is shorter than any real stay, so every
session is cut off by the end of the footage, not by anyone leaving);
59 person track IDs — ByteTrack fragments IDs heavily in a crowd this
busy, but table occupancy is zone-based so the board stayed stable
throughout. At the moment each table was first marked occupied the
board read T1 1 seated, T2 4 seated, T3 3 seated, T4 6 seated.

## Files

- `restaurant_table_timing.py` — the whole system (single file)
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer
