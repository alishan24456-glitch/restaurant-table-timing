#!/usr/bin/env python3
"""
Restaurant Table Lifecycle Tracker -- the full life of a dining table.

Watches a fixed restaurant camera and follows every table through its
complete cycle, as an explicit per-table state machine over fixed
table zones (ROIs):

    FREE --> OCCUPIED --> NEEDS_CLEANING --> CLEANING --> FREE
                |                                 ^
                +---- (party leaves, no items) ---+

  * FREE            the table is empty and clear.
  * OCCUPIED        people are seated at the table; a dwell timer runs
                    for the party (how long they have been sitting).
  * NEEDS_CLEANING  the party has left AND leftover items are detected
                    on the table. If the party leaves and nothing is
                    left behind, the table returns straight to FREE.
  * CLEANING        while NEEDS_CLEANING, a person (a worker) is
                    detected inside the table zone. When the worker is
                    gone and no items remain, the table registers FREE
                    again -- clean and available for the next party.

Detection is YOLOv8 (Ultralytics) + ByteTrack. "Leftovers" are judged
from YOLOv8 COCO item detections on the table -- cup, bowl, bottle,
wine glass, fork, knife, spoon, cell phone, book, handbag. That is an
honest proxy, not x-ray vision: an item only counts when the detector
can actually see it inside the table zone.

Table zones are DERIVED from YOLOv8 'dining table' detections where
possible: the video is sampled, the table boxes are clustered, and
each stable cluster (expanded to cover the seats around the table)
becomes a zone. If fewer than two stable tables are found, the run
falls back to documented manual ROIs (the four defaults below, or a
--tables JSON file) and says so in the log header. Small models miss
far tables often, so the fallback is a normal, expected path -- on
the bundled demo clip 'dining table' fires reliably on exactly one
foreground table, and the four manual zones are used.

Run it (Windows CMD):

    pip install -r requirements.txt
    python restaurant_table_lifecycle.py                 REM opens a video file chooser
    python restaurant_table_lifecycle.py --source pick   REM same chooser, explicitly
    python restaurant_table_lifecycle.py --source restaurant.mp4
    python restaurant_table_lifecycle.py --source 0      REM webcam
    python lifecycle_logic_test.py                       REM synthetic-stream logic test

Press Q to quit. The annotated video is written to output.mp4
(change with --output); the timestamped event log goes to events.log
(change with --log). Useful options: --zones auto|derived|manual,
--tables tables.json (custom manual zones), --occupy 1.5, --leave 2.0,
--worker 0.5, --clear 3.0, --conf 0.25, --imgsz 1280 (for small/far
diners), --weights yolov8n.pt (downloaded automatically by Ultralytics
on first use if missing), --max-seconds 30, --no-display.

Honest framing: the machine sees boxes, not intentions. "Seated" means
a tracked person whose centroid is inside the zone and nearly still;
"CLEANING" means a person is present in a dirty table's zone -- the
system cannot prove that person is staff, and during CLEANING a newly
seated guest is not distinguished from the worker. A state only ever
changes on evidence sustained over time, never on a single frame.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class ids (YOLOv8) used by this project.
PERSON_ID = 0
DINING_TABLE_ID = 60
ITEM_CLASS_IDS = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    67: "cell phone",
    73: "book",
    26: "handbag",
}

# Default manual table zones as fractions of the frame (x1, y1, x2, y2).
# Four tables across the dining floor -- the documented fallback used
# when 'dining table' auto-derivation finds fewer than two stable
# tables, and the zones the bundled demo clip was verified with.
DEFAULT_TABLES = [
    ("T1", 0.03, 0.55, 0.25, 0.93),
    ("T2", 0.28, 0.55, 0.50, 0.93),
    ("T3", 0.53, 0.55, 0.75, 0.93),
    ("T4", 0.78, 0.55, 0.97, 0.93),
]

OCCUPY_DELAY_S = 1.5    # still presence before a party counts as seated
LEAVE_DELAY_S = 2.0     # seated absence before the party counts as gone
WORKER_DELAY_S = 0.5    # person presence in a dirty zone -> CLEANING
CLEAN_DONE_S = 1.0      # worker gone + items gone this long -> FREE
CLEAR_DELAY_S = 3.0     # items gone (no worker seen) this long -> FREE
ITEM_WINDOW_S = 1.0     # rolling window for leftover-item confirmation
ITEM_MIN_FRACTION = 0.4  # items must appear in >= 40% of window frames
STILL_SPEED = 60.0      # px/s at 1280 px wide -- above this: walking


class TableState(Enum):
    FREE = "FREE"
    OCCUPIED = "OCCUPIED"
    NEEDS_CLEANING = "NEEDS_CLEANING"
    CLEANING = "CLEANING"


STATE_COLOURS = {  # BGR
    TableState.FREE: (90, 200, 90),            # green
    TableState.OCCUPIED: (60, 60, 230),        # red
    TableState.NEEDS_CLEANING: (0, 165, 255),  # orange
    TableState.CLEANING: (230, 120, 200),      # pink
}
COLOUR_PERSON = (230, 200, 60)     # cyan-blue
COLOUR_ITEM = (0, 215, 255)        # yellow
COLOUR_BOARD = (24, 24, 28)
COLOUR_TEXT = (245, 245, 245)


# ---------------------------------------------------------------------------
# Shared detection type -- the one thing the detector and the engine agree on
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str = "person"
    conf: float = 0.99
    track_id: int = -1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0


# ---------------------------------------------------------------------------
# Event log -- every state transition, timestamped
# ---------------------------------------------------------------------------

class EventLog:
    def __init__(self, path: str | None):
        self.path = path
        self.events = []           # (t, type, detail)
        self.counts = Counter()
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Restaurant Table Lifecycle Tracker -- event log\n")
                fh.write("States: FREE -> OCCUPIED -> NEEDS_CLEANING -> "
                         "CLEANING -> FREE. Leftovers = YOLO item "
                         "detections inside the table zone.\n\n")

    def add(self, t: float, etype: str, detail: str):
        line = f"[{t:7.1f}s]  {etype:<16} {detail}"
        print(line, flush=True)
        self.events.append((t, etype, detail))
        self.counts[etype] += 1
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------------------
# One table -- zone, state machine data, accumulated statistics
# ---------------------------------------------------------------------------

@dataclass
class Table:
    name: str
    rect: tuple                    # absolute pixels (x1, y1, x2, y2)
    state: TableState = TableState.FREE
    state_since: float = 0.0       # when the current state began
    # evidence timers (None = evidence not currently present)
    seated_since: float | None = None    # still person present since
    last_seated_t: float | None = None   # last frame a seated person was seen
    person_since: float | None = None    # any person present since
    person_absent_since: float | None = None
    items_clear_since: float | None = None
    # current spell bookkeeping
    session_start: float | None = None   # when the seated party arrived
    dirty_since: float | None = None     # when NEEDS_CLEANING began
    items_hist: deque = field(default_factory=deque)  # (t, n_items)
    item_classes_now: Counter = field(default_factory=Counter)
    seated_now: int = 0
    items_now: int = 0
    # statistics
    sittings: list = field(default_factory=list)     # (start, end, dwell)
    cleanings: list = field(default_factory=list)    # (wait_s, clean_s|None)
    state_time: Counter = field(default_factory=Counter)
    max_party: int = 0

    def contains(self, cx: float, cy: float) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def items_confirmed(self, t: float) -> bool:
        """Leftovers confirmed: item detections inside the zone in at
        least ITEM_MIN_FRACTION of the frames of the last ITEM_WINDOW_S
        seconds (and in at least 2 frames) -- a single flickering
        detection never counts."""
        window = [(tt, n) for (tt, n) in self.items_hist
                  if t - tt <= ITEM_WINDOW_S]
        if len(window) < 2:
            return False
        with_items = sum(1 for (_, n) in window if n > 0)
        return with_items >= 2 and with_items / len(window) >= ITEM_MIN_FRACTION


# ---------------------------------------------------------------------------
# Lifecycle engine -- sees only the Detection stream, never the scene
# ---------------------------------------------------------------------------

class LifecycleEngine:
    def __init__(self, frame_w: int, frame_h: int, log: EventLog,
                 tables: list,
                 occupy_delay: float = OCCUPY_DELAY_S,
                 leave_delay: float = LEAVE_DELAY_S,
                 worker_delay: float = WORKER_DELAY_S,
                 clean_done: float = CLEAN_DONE_S,
                 clear_delay: float = CLEAR_DELAY_S):
        self.fw, self.fh = frame_w, frame_h
        self.log = log
        self.occupy_delay = occupy_delay
        self.leave_delay = leave_delay
        self.worker_delay = worker_delay
        self.clean_done = clean_done
        self.clear_delay = clear_delay
        self.tables: list[Table] = []
        for (name, fx1, fy1, fx2, fy2) in tables:
            rect = (fx1 * frame_w, fy1 * frame_h,
                    fx2 * frame_w, fy2 * frame_h)
            self.tables.append(Table(name, rect))
        # per-track centroid history for the stillness test: tid -> deque
        self.hist: dict[int, deque] = {}
        self.all_ids: set[int] = set()
        self.transitions: list = []      # (t, table, from_state, to_state)
        self.frames = 0

    # -- helpers -----------------------------------------------------------
    def speed_at(self, tid: int, t_now: float, back: float = 0.5) -> float:
        """Centroid speed (px/s) over the last `back` seconds of history."""
        h = self.hist.get(tid)
        if not h or len(h) < 2:
            return 0.0
        t1, x1, y1 = h[-1]
        past = h[0]
        for item in h:
            if item[0] <= t1 - back:
                past = item
            else:
                break
        dt = t1 - past[0]
        if dt <= 1e-6:
            return 0.0
        return math.hypot(x1 - past[1], y1 - past[2]) / dt

    def still_limit(self) -> float:
        # STILL_SPEED is defined at 1280 px wide; scale with frame width.
        return STILL_SPEED * (self.fw / 1280.0)

    def _set_state(self, table: Table, new: TableState, t: float,
                   detail: str = ""):
        old = table.state
        if old is new:
            return
        table.state_time[old.value] += t - table.state_since
        table.state = new
        table.state_since = t
        # evidence timers belong to the old spell; reset them
        table.seated_since = None
        table.person_since = None
        table.person_absent_since = None
        table.items_clear_since = None
        self.transitions.append((t, table.name, old.value, new.value))
        self.log.add(t, "STATE",
                     "%s %s -> %s%s" % (table.name, old.value, new.value,
                                        (" -- " + detail) if detail else ""))

    # -- per-frame update ----------------------------------------------------
    def update(self, dets: list, t: float):
        self.frames += 1
        persons = [d for d in dets if d.cls == "person"]
        tracked = [d for d in persons if d.track_id >= 0]
        for d in tracked:
            self.all_ids.add(d.track_id)
            h = self.hist.setdefault(d.track_id, deque(maxlen=45))
            h.append((t, d.cx, d.cy))
        items = [d for d in dets if d.cls in ITEM_CLASS_IDS.values()]
        limit = self.still_limit()

        for table in self.tables:
            seated = [d for d in tracked
                      if table.contains(d.cx, d.cy)
                      and self.speed_at(d.track_id, t) <= limit]
            present = [d for d in persons if table.contains(d.cx, d.cy)]
            on_table = [d for d in items if table.contains(d.cx, d.cy)]
            table.seated_now = len(seated)
            table.items_now = len(on_table)
            table.item_classes_now = Counter(d.cls for d in on_table)
            table.items_hist.append((t, len(on_table)))
            while table.items_hist and t - table.items_hist[0][0] > 5.0:
                table.items_hist.popleft()
            if seated:
                if table.seated_since is None:
                    table.seated_since = t
                table.last_seated_t = t
            if present:
                if table.person_since is None:
                    table.person_since = t
                table.person_absent_since = None
            elif table.person_absent_since is None:
                table.person_absent_since = t
            self._step(table, seated, present, t)

    def _step(self, table: Table, seated: list, present: list, t: float):
        st = table.state
        if st is TableState.FREE:
            if (seated and table.seated_since is not None
                    and t - table.seated_since >= self.occupy_delay):
                table.session_start = table.seated_since
                table.max_party = max(table.max_party, len(seated))
                self._set_state(table, TableState.OCCUPIED, t,
                                "party of %d seated" % len(seated))

        elif st is TableState.OCCUPIED:
            table.max_party = max(table.max_party, len(seated))
            if not seated and table.last_seated_t is not None:
                if t - table.last_seated_t >= self.leave_delay:
                    start = (table.session_start
                             if table.session_start is not None
                             else table.last_seated_t)
                    dwell = max(0.0, table.last_seated_t - start)
                    table.sittings.append((start, table.last_seated_t, dwell))
                    table.session_start = None
                    table.last_seated_t = None
                    if table.items_confirmed(t):
                        items = ", ".join(
                            "%s x%d" % (c, n) for c, n in
                            table.item_classes_now.most_common())
                        table.dirty_since = t
                        self._set_state(
                            table, TableState.NEEDS_CLEANING, t,
                            "party left after %.1fs; leftovers: %s"
                            % (dwell, items or "items seen moments before"))
                    else:
                        self._set_state(
                            table, TableState.FREE, t,
                            "party left after %.1fs; nothing left behind"
                            % dwell)

        elif st is TableState.NEEDS_CLEANING:
            if (present and table.person_since is not None
                    and t - table.person_since >= self.worker_delay):
                self._set_state(table, TableState.CLEANING, t,
                                "person in zone -- cleaning under way")
            elif table.items_confirmed(t):
                table.items_clear_since = None
            else:
                if table.items_clear_since is None:
                    table.items_clear_since = t
                elif t - table.items_clear_since >= self.clear_delay:
                    wait = t - table.state_since
                    table.cleanings.append((wait, None))
                    table.dirty_since = None
                    self._set_state(
                        table, TableState.FREE, t,
                        "items gone without a worker visit "
                        "(waited %.1fs) -- counted clear" % wait)

        elif st is TableState.CLEANING:
            if present:
                pass  # worker still here; nothing to decide yet
            elif (table.person_absent_since is not None
                    and t - table.person_absent_since >= self.clean_done):
                clean_s = t - table.state_since
                if table.items_confirmed(t):
                    table.dirty_since = t   # a fresh dirty spell begins
                    self._set_state(
                        table, TableState.NEEDS_CLEANING, t,
                        "worker left but items remain on the table")
                else:
                    wait = (table.state_since - table.dirty_since
                            if table.dirty_since is not None else 0.0)
                    table.cleanings.append((wait, clean_s))
                    table.dirty_since = None
                    self._set_state(
                        table, TableState.FREE, t,
                        "worker gone and table clear -- clean "
                        "(waited %.1fs, cleaning took %.1fs)"
                        % (wait, clean_s))

    def finish(self, t: float):
        """Close the books at the end of the stream (no fake transitions:
        open spells are recorded as cut off by the end of the footage)."""
        for table in self.tables:
            table.state_time[table.state.value] += t - table.state_since
            if table.state is TableState.OCCUPIED:
                start = (table.session_start
                         if table.session_start is not None else t)
                end = (table.last_seated_t
                       if table.last_seated_t is not None else t)
                table.sittings.append((start, end, max(0.0, end - start)))
                table.session_start = None

    def summary(self) -> dict:
        trans = Counter("%s->%s" % (f, to)
                        for (_, _, f, to) in self.transitions)
        sittings = [s for tb in self.tables for s in tb.sittings]
        dwells = [s[2] for s in sittings]
        cleanings = [c for tb in self.tables for c in tb.cleanings]
        return {
            "frames": self.frames,
            "people": len(self.all_ids),
            "tables": len(self.tables),
            "transitions": len(self.transitions),
            "transitions_by_type": dict(trans),
            "sittings": len(sittings),
            "avg_dwell": float(np.mean(dwells)) if dwells else 0.0,
            "longest_dwell": max(dwells) if dwells else 0.0,
            "cleaning_events": len(cleanings),
            "state_time": {tb.name: dict(tb.state_time)
                           for tb in self.tables},
            "final_states": {tb.name: tb.state.value
                             for tb in self.tables},
            "events": dict(self.log.counts),
        }


# ---------------------------------------------------------------------------
# Table-zone derivation from 'dining table' detections
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def derive_table_zones(model, cap, conf: float = 0.20, sample_every: int = 15,
                       min_stability: float = 0.20,
                       iou_thresh: float = 0.30):
    """Sample the video, cluster 'dining table' boxes, and turn each
    stable cluster into a zone (median tabletop box expanded to cover
    the seats around it). Returns (zones, info) where zones are
    (name, fx1, fy1, fx2, fy2) fractions; info describes what was found
    so the caller can log it honestly."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    boxes = []          # (sample_no, x1, y1, x2, y2 fractions)
    sample_no = -1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            sample_no += 1
            h, w = frame.shape[:2]
            res = model.predict(frame, imgsz=1280, conf=conf,
                                classes=[DINING_TABLE_ID], verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append((sample_no, x1 / w, y1 / h, x2 / w, y2 / h))
        idx += 1
        if total and idx >= total:
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_samples = sample_no + 1

    # greedy clustering against each cluster's running median box
    clusters = []       # {"boxes": [...], "frames": set()}
    for (s, x1, y1, x2, y2) in boxes:
        for cl in clusters:
            med = np.median([b[1:5] for b in cl["boxes"]], axis=0)
            if _iou((x1, y1, x2, y2), tuple(med)) >= iou_thresh:
                cl["boxes"].append((s, x1, y1, x2, y2))
                cl["frames"].add(s)
                break
        else:
            clusters.append({"boxes": [(s, x1, y1, x2, y2)],
                             "frames": {s}})

    zones = []
    kept = []
    for cl in clusters:
        stability = len(cl["frames"]) / max(1, n_samples)
        if stability < min_stability:
            continue
        med = np.median([b[1:5] for b in cl["boxes"]], axis=0)
        x1, y1, x2, y2 = [float(v) for v in med]
        w, h = x2 - x1, y2 - y1
        # the tabletop box alone excludes the diners; expand to the
        # seating area around it (documented factors)
        zx1 = max(0.0, x1 - 0.45 * w)
        zx2 = min(1.0, x2 + 0.45 * w)
        zy1 = max(0.0, y1 - 1.10 * h)
        zy2 = min(1.0, y2 + 0.70 * h)
        kept.append((zx1, zy1, zx2, zy2, stability, (x1, y1, x2, y2)))
    kept.sort(key=lambda z: (z[0] + z[2]) / 2.0)
    for i, (zx1, zy1, zx2, zy2, _, _) in enumerate(kept):
        zones.append(("T%d" % (i + 1), zx1, zy1, zx2, zy2))
    info = {
        "sampled_frames": n_samples,
        "dining_table_dets": len(boxes),
        "clusters": len(clusters),
        "stable_clusters": len(kept),
        "derived_zones": zones,
        "tabletops": [k[5] for k in kept],
        "stabilities": [round(k[4], 2) for k in kept],
    }
    return zones, info


# ---------------------------------------------------------------------------
# Overlay -- zones, boxes, per-table board and the dashboard strip
# ---------------------------------------------------------------------------

def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return "%02d:%02d" % (seconds // 60, seconds % 60)


def display_timer(table: Table, t: float) -> float:
    """The timer a table shows on screen: for OCCUPIED that is the
    party's dwell (time since they sat down, including the seconds
    before the state confirmed); for every other state it is the
    time spent in that state."""
    if table.state is TableState.OCCUPIED and table.session_start is not None:
        return t - table.session_start
    return t - table.state_since


def draw_overlay(img, engine: LifecycleEngine, dets: list, t: float,
                 source_label: str, frames_done: int):
    h, w = img.shape[:2]
    # table zones with state colour and live state timer
    for table in engine.tables:
        x1, y1, x2, y2 = [int(v) for v in table.rect]
        colour = STATE_COLOURS[table.state]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        label = "%s %s %s" % (table.name, table.state.value,
                              fmt_clock(display_timer(table, t)))
        if table.items_now:
            label += "  items %d" % table.items_now
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1),
                      colour, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        if table.seated_now:
            cv2.putText(img, "%d seated" % table.seated_now,
                        (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # item boxes (small, yellow) -- the leftover evidence
    for d in dets:
        if d.cls in ITEM_CLASS_IDS.values():
            cv2.rectangle(img, (int(d.x1), int(d.y1)),
                          (int(d.x2), int(d.y2)), COLOUR_ITEM, 1)
            cv2.putText(img, d.cls, (int(d.x1), max(10, int(d.y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOUR_ITEM, 1)

    # person boxes (thin, cyan) with track id
    for d in dets:
        if d.cls != "person":
            continue
        cv2.rectangle(img, (int(d.x1), int(d.y1)),
                      (int(d.x2), int(d.y2)), COLOUR_PERSON, 1)
        cv2.putText(img, "ID %d" % d.track_id,
                    (int(d.x1) + 2, max(12, int(d.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_PERSON, 1)

    # status board (top-left): one line per table
    counts = Counter(tb.state.value for tb in engine.tables)
    lines = ["TABLE LIFECYCLE  t=%s   FREE %d  OCCUPIED %d  "
             "NEEDS_CLEANING %d  CLEANING %d"
             % (fmt_clock(t), counts.get("FREE", 0),
                counts.get("OCCUPIED", 0), counts.get("NEEDS_CLEANING", 0),
                counts.get("CLEANING", 0))]
    for tb in engine.tables:
        lines.append("%s  %-14s %s   seated %d  items %d  sittings %d"
                     % (tb.name, tb.state.value,
                        fmt_clock(display_timer(tb, t)), tb.seated_now,
                        tb.items_now, len(tb.sittings)))
    lines.append("transitions so far: %d    %s    frame %d"
                 % (len(engine.transitions), source_label, frames_done))
    bw = max(len(ln) for ln in lines) * 11 + 20
    bh = len(lines) * 24 + 16
    cv2.rectangle(img, (10, 10), (10 + bw, 10 + bh), COLOUR_BOARD, -1)
    cv2.rectangle(img, (10, 10), (10 + bw, 10 + bh), (200, 200, 200), 1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (20, 10 + 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_TEXT, 1)

    # dashboard strip (bottom): one cell per table, state-coloured
    strip_h = 44
    cell_w = w // max(1, len(engine.tables))
    y0 = h - strip_h
    for i, tb in enumerate(engine.tables):
        cx1 = i * cell_w
        cx2 = w if i == len(engine.tables) - 1 else (i + 1) * cell_w
        cv2.rectangle(img, (cx1, y0), (cx2, h), STATE_COLOURS[tb.state], -1)
        cv2.rectangle(img, (cx1, y0), (cx2, h), (15, 15, 15), 2)
        cell = "%s  %s  %s" % (tb.name, tb.state.value,
                               fmt_clock(display_timer(tb, t)))
        cv2.putText(img, cell, (cx1 + 12, y0 + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2)


# ---------------------------------------------------------------------------
# Real mode -- YOLOv8 + ByteTrack
# ---------------------------------------------------------------------------

def run_real(model, cap, engine, writer, args, source_label, total_frames):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    classes = [PERSON_ID] + sorted(ITEM_CLASS_IDS)
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if args.max_seconds and t >= args.max_seconds:
            break
        results = model.track(frame, persist=True, tracker=args.tracker,
                              classes=classes, conf=args.conf,
                              imgsz=args.imgsz, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            names = results[0].names
            for b in results[0].boxes:
                cls_name = names[int(b.cls[0])]
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                tid = int(b.id[0]) if b.id is not None else -1
                dets.append(Detection(x1, y1, x2, y2, cls_name,
                                      float(b.conf[0]), tid))
        engine.update(dets, t)
        draw_overlay(frame, engine, dets, t, source_label, frame_idx + 1)
        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Restaurant Table Lifecycle Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if total_frames and frame_idx % 60 == 0:
            print("  ... frame %d/%d (%.0f%%)"
                  % (frame_idx, total_frames,
                     100.0 * frame_idx / total_frames), flush=True)
    engine.finish(frame_idx / fps if fps else 0.0)
    print("real mode: %d frames in %.1fs wall time"
          % (frame_idx, time.time() - t0), flush=True)
    return frame_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def pick_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a restaurant video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                   ("All files", "*.*")])
    root.destroy()
    return path


def load_tables(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    tables = []
    for item in data:
        if isinstance(item, dict):
            tables.append((item["name"], item["x1"], item["y1"],
                           item["x2"], item["y2"]))
        else:
            tables.append(tuple(item))
    return tables


def resolve_zones(model, cap, args, log: EventLog):
    """Decide the table zones for this run and log the decision."""
    if args.tables:
        zones = load_tables(args.tables)
        log.add(0.0, "ZONES", "manual zones from %s (%d tables)"
                % (args.tables, len(zones)))
        return zones
    if args.zones == "manual":
        log.add(0.0, "ZONES", "built-in manual zones (%d tables)"
                % len(DEFAULT_TABLES))
        return list(DEFAULT_TABLES)
    derived, info = derive_table_zones(model, cap)
    log.add(0.0, "ZONES",
            "auto-derivation: %d sampled frames, %d 'dining table' "
            "detections, %d cluster(s), %d stable (stabilities %s)"
            % (info["sampled_frames"], info["dining_table_dets"],
               info["clusters"], info["stable_clusters"],
               info["stabilities"]))
    if args.zones == "derived":
        if not derived:
            log.add(0.0, "ZONES", "ERROR: --zones derived but no stable "
                    "'dining table' cluster was found")
            return None
        log.add(0.0, "ZONES", "using %d derived zone(s)" % len(derived))
        return derived
    # auto: derived zones only when at least two tables were found
    if len(derived) >= 2:
        log.add(0.0, "ZONES", "using %d derived zone(s)" % len(derived))
        return derived
    log.add(0.0, "ZONES",
            "only %d stable table(s) derived -- falling back to the "
            "documented manual zones (%d tables)"
            % (len(derived), len(DEFAULT_TABLES)))
    return list(DEFAULT_TABLES)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Restaurant table lifecycle tracker "
                    "(FREE -> OCCUPIED -> NEEDS_CLEANING -> CLEANING -> FREE).")
    ap.add_argument("--source", default=None,
                    help="video file, webcam index, or 'pick' for a file "
                         "chooser (default: opens the chooser)")
    ap.add_argument("--output", default="output.mp4",
                    help="annotated video out (default output.mp4)")
    ap.add_argument("--log", default="events.log",
                    help="event log out (default events.log)")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="YOLOv8 weights (default yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence (default 0.25)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (default bytetrack.yaml)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference size (default 640; use 1280 for "
                         "small/far diners)")
    ap.add_argument("--zones", choices=["auto", "derived", "manual"],
                    default="auto",
                    help="table zones: derive from 'dining table' "
                         "detections (auto = derived when >= 2 stable "
                         "tables, else manual fallback), derived only, "
                         "or manual only (default auto)")
    ap.add_argument("--tables", default=None,
                    help="JSON file with custom manual table zones "
                         '[{"name":"T1","x1":0.03,"y1":0.55,"x2":0.25,"y2":0.93}] '
                         "(fractions of the frame)")
    ap.add_argument("--occupy", type=float, default=OCCUPY_DELAY_S,
                    help="seconds of still presence before a party counts "
                         "seated (default 1.5)")
    ap.add_argument("--leave", type=float, default=LEAVE_DELAY_S,
                    help="seconds of seated absence before the party "
                         "counts as gone (default 2.0)")
    ap.add_argument("--worker", type=float, default=WORKER_DELAY_S,
                    help="seconds of person presence in a dirty zone "
                         "before CLEANING starts (default 0.5)")
    ap.add_argument("--clear", type=float, default=CLEAR_DELAY_S,
                    help="seconds with items gone (no worker visit) "
                         "before a dirty table counts clear (default 3.0)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many video seconds (0 = all)")
    ap.add_argument("--no-display", action="store_true",
                    help="headless run (no preview window)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    source = args.source
    if source is None or source == "pick":
        source = pick_file()
        if not source:
            print("No video chosen -- exiting.")
            return 1
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print("Could not open video source: %s" % source)
        return 1
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("source: %s  (%dx%d @ %.1f fps, %d frames)"
          % (source, fw, fh, fps, total), flush=True)

    model = YOLO(args.weights)
    log = EventLog(args.log)
    if str(source).isdigit():
        zones = list(DEFAULT_TABLES)   # webcam: no pre-pass possible
        log.add(0.0, "ZONES", "webcam source -- built-in manual zones")
    else:
        zones = resolve_zones(model, cap, args, log)
        if zones is None:
            cap.release()
            return 1
    engine = LifecycleEngine(fw, fh, log, zones, args.occupy, args.leave,
                             args.worker, CLEAN_DONE_S, args.clear)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))
    label = ("webcam %s" % source) if str(source).isdigit() \
        else os.path.basename(str(source))
    run_real(model, cap, engine, writer, args, label, total)
    writer.release()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    s = engine.summary()
    print()
    print("=" * 64)
    print("RESTAURANT TABLE LIFECYCLE -- %s" % label)
    print("=" * 64)
    print("frames processed      : %d" % s["frames"])
    print("people tracked        : %d" % s["people"])
    print("tables                : %d" % s["tables"])
    print("transitions           : %d  %s"
          % (s["transitions"], s["transitions_by_type"]))
    print("sittings (seat->leave): %d" % s["sittings"])
    print("average dwell         : %.1f s" % s["avg_dwell"])
    print("longest dwell         : %.1f s" % s["longest_dwell"])
    print("cleaning events       : %d" % s["cleaning_events"])
    print("final states          : %s" % s["final_states"])
    print("time per state        : %s" % s["state_time"])
    print("events                : %s" % s["events"])
    print("annotated video       : %s" % args.output)
    print("event log             : %s" % args.log)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
