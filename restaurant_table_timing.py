#!/usr/bin/env python3
"""
Restaurant Table-Timing -- table-occupancy analytics for a restaurant camera.

Watches a fixed restaurant camera, tracks every person with YOLOv8 +
ByteTrack, and times how long each table stays occupied:

  * fixed table zones (ROIs) -- a table is FREE or OCCUPIED
  * a table becomes OCCUPIED when a still person is seen inside its zone
    for OCCUPY_DELAY seconds (walkers passing through never count: a
    person must be nearly stationary, which is what "seated" looks like
    to a camera)
  * it becomes FREE again after FREE_DELAY seconds with nobody seated
  * every seating is logged (table, seated time, freed time, total stay)
    and the run ends with per-table sessions, average stay and longest stay

Run it (Windows CMD):

    pip install -r requirements.txt
    python restaurant_table_timing.py                 REM opens a video file chooser
    python restaurant_table_timing.py --source pick   REM same chooser, explicitly
    python restaurant_table_timing.py --source restaurant.mp4
    python restaurant_table_timing.py --source 0      REM webcam
    python restaurant_table_timing.py --simulate      REM rendered dining room

Press Q to quit. The annotated video is written to output.mp4
(change with --output); the session log goes to sessions.log
(change with --log). Useful options: --tables tables.json (custom zones),
--occupy 1.5 (seconds of still presence before a table counts occupied),
--free 3.0 (seconds of absence before it counts free), --conf 0.30,
--weights yolov8n.pt (downloaded automatically by Ultralytics on first
use if missing), --max-seconds 30, --no-display for headless runs.

Honest framing: occupancy is measured from person boxes and stillness
only. It cannot tell whether someone is eating, waiting for food, or
just sitting -- it times "a person is seated at this table", nothing more.
A table with a bag or a coat on it and nobody seated reads FREE.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720
SIM_FPS = 30.0

# Default table zones as fractions of the frame (x1, y1, x2, y2).
# Four tables across the dining floor, matching the rendered simulation
# scene and the real restaurant clip used for verification.
DEFAULT_TABLES = [
    ("T1", 0.03, 0.55, 0.25, 0.93),
    ("T2", 0.28, 0.55, 0.50, 0.93),
    ("T3", 0.53, 0.55, 0.75, 0.93),
    ("T4", 0.78, 0.55, 0.97, 0.93),
]

OCCUPY_DELAY_S = 1.5     # still presence needed before a table counts occupied
FREE_DELAY_S = 3.0       # absence needed before it counts free again
STILL_SPEED = 60.0       # px/s at 1280 px wide -- above this a person is walking
TRACK_LOST_S = 1.0       # real-mode tracks older than this stop counting

COLOUR_FREE = (90, 200, 90)        # green
COLOUR_OCCUPIED = (60, 60, 230)    # red
COLOUR_PERSON = (230, 200, 60)     # cyan-blue
COLOUR_BOARD = (24, 24, 28)
COLOUR_TEXT = (245, 245, 245)


# ---------------------------------------------------------------------------
# Shared detection type -- the one thing real mode and simulation agree on
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

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Table-timing engine -- sees only the Detection stream, never the scene
# ---------------------------------------------------------------------------

@dataclass
class TableState:
    name: str
    rect: tuple                    # absolute pixels (x1, y1, x2, y2)
    occupied: bool = False
    present_since: float | None = None   # first still presence in this spell
    last_seated_t: float | None = None   # last frame a still person was here
    session_start: float | None = None
    sessions: list = field(default_factory=list)  # (start, end, seconds)
    people_now: int = 0
    max_people: int = 0
    total_occupied_s: float = 0.0

    def contains(self, cx: float, cy: float) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= cx <= x2 and y1 <= cy <= y2


class SessionLog:
    def __init__(self, path: str | None):
        self.path = path
        self.events = []           # (t, type, detail)
        self.counts = {}
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Restaurant Table-Timing -- table session log\n")
                fh.write("Occupancy = a still person inside a table zone. "
                         "It times seating, not eating.\n\n")

    def add(self, t: float, etype: str, detail: str):
        line = f"[{t:7.1f}s]  {etype:<12} {detail}"
        print(line, flush=True)
        self.events.append((t, etype, detail))
        self.counts[etype] = self.counts.get(etype, 0) + 1
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class TableTimingEngine:
    def __init__(self, frame_w: int, frame_h: int, log: SessionLog,
                 tables: list | None = None,
                 occupy_delay: float = OCCUPY_DELAY_S,
                 free_delay: float = FREE_DELAY_S):
        self.fw, self.fh = frame_w, frame_h
        self.log = log
        self.occupy_delay = occupy_delay
        self.free_delay = free_delay
        self.tables: list[TableState] = []
        for (name, fx1, fy1, fx2, fy2) in (tables or DEFAULT_TABLES):
            rect = (fx1 * frame_w, fy1 * frame_h,
                    fx2 * frame_w, fy2 * frame_h)
            self.tables.append(TableState(name, rect))
        # per-track centroid history for the stillness test: tid -> deque
        self.hist: dict[int, deque] = {}
        self.all_ids: set[int] = set()
        self.peak_occupied = 0
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

    # -- per-frame update ----------------------------------------------------
    def update(self, dets: list, t: float):
        self.frames += 1
        persons = [d for d in dets if d.cls == "person" and d.track_id >= 0]
        for d in persons:
            self.all_ids.add(d.track_id)
            h = self.hist.setdefault(d.track_id, deque(maxlen=45))
            h.append((t, d.cx, d.cy))

        limit = self.still_limit()
        for table in self.tables:
            seated = [d for d in persons
                      if table.contains(d.cx, d.cy)
                      and self.speed_at(d.track_id, t) <= limit]
            table.people_now = len(seated)
            table.max_people = max(table.max_people, len(seated))

            if seated:
                if table.present_since is None:
                    table.present_since = t
                table.last_seated_t = t
                if (not table.occupied
                        and t - table.present_since >= self.occupy_delay):
                    table.occupied = True
                    table.session_start = table.present_since
                    self.log.add(t, "TABLE SEATED",
                                 "%s occupied by %d seated (%s)"
                                 % (table.name, len(seated),
                                    ", ".join("ID %d" % d.track_id
                                              for d in seated)))
            else:
                if table.occupied:
                    # nobody seated -- start the freeing countdown
                    if (table.last_seated_t is not None
                            and t - table.last_seated_t >= self.free_delay):
                        self._free_table(table, table.last_seated_t)
                # presence spell broken either way
                table.present_since = None

        occupied_now = sum(1 for tb in self.tables if tb.occupied)
        self.peak_occupied = max(self.peak_occupied, occupied_now)

    def _free_table(self, table: TableState, end_t: float):
        start = table.session_start if table.session_start is not None else end_t
        stay = max(0.0, end_t - start)
        table.sessions.append((start, end_t, stay))
        table.total_occupied_s += stay
        self.log.add(end_t, "TABLE FREED",
                     "%s free after %.1fs seated"
                     % (table.name, stay))
        table.occupied = False
        table.session_start = None
        table.present_since = None
        table.last_seated_t = None

    def finish(self, t: float):
        """Close any session still open at the end of the stream."""
        for table in self.tables:
            if table.occupied:
                self._free_table(table, table.last_seated_t or t)

    def summary(self) -> dict:
        all_sessions = [s for tb in self.tables for s in tb.sessions]
        stays = [s[2] for s in all_sessions]
        return {
            "frames": self.frames,
            "people": len(self.all_ids),
            "tables": len(self.tables),
            "sessions": len(all_sessions),
            "sessions_by_table": {tb.name: len(tb.sessions)
                                  for tb in self.tables},
            "avg_stay": float(np.mean(stays)) if stays else 0.0,
            "longest_stay": max(stays) if stays else 0.0,
            "total_occupied_s": sum(tb.total_occupied_s for tb in self.tables),
            "peak_occupied": self.peak_occupied,
            "events": dict(self.log.counts),
        }


# ---------------------------------------------------------------------------
# Simulation -- renders a dining room and publishes its detection stream.
# The engine only ever sees the published boxes, via the same centroid
# tracker a real detector + tracker would feed it. No ground-truth peeking.
# ---------------------------------------------------------------------------

@dataclass
class SimDiner:
    name: str
    colour: tuple
    # path segments: (t0, t1, x0, y0, x1, y1, seated) -- feet position lerped
    segments: list
    height: float = 132.0     # standing height (px)
    width: float = 48.0

    def pos(self, t: float):
        for (t0, t1, x0, y0, x1, y1, seated) in self.segments:
            if t0 <= t <= t1:
                if t1 == t0:
                    return x1, y1, seated, True
                f = (t - t0) / (t1 - t0)
                return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f,
                        seated, True)
        return 0.0, 0.0, False, False


def build_sim_diners():
    """Scripted evening: two at T1, one long-stay at T2, a family at T3,
    two sittings at T4, and a waiter who walks through (never seated)."""
    return [
        # Couple at T1: arrive, sit ~16 s, leave together.
        SimDiner(name="couple-a", colour=(60, 120, 220),
                 segments=[(1.0, 3.0, 30, 640, 180, 600, False),
                           (3.0, 20.0, 180, 600, 180, 600, True),
                           (20.0, 21.5, 180, 600, 30, 660, False)]),
        SimDiner(name="couple-b", colour=(220, 130, 60),
                 segments=[(1.0, 3.0, 30, 660, 250, 600, False),
                           (3.0, 20.0, 250, 600, 250, 600, True),
                           (20.0, 21.5, 250, 600, 30, 680, False)]),
        # Solo diner at T2: already seated when the run starts, stays ~27 s.
        SimDiner(name="solo", colour=(70, 180, 120),
                 segments=[(0.0, 27.0, 470, 600, 470, 600, True),
                           (27.0, 29.5, 470, 600, 1240, 660, False)]),
        # Family of two at T3: arrive mid-run, sit ~13 s.
        SimDiner(name="family-a", colour=(200, 90, 200),
                 segments=[(9.0, 12.0, 1250, 640, 790, 600, False),
                           (12.0, 25.0, 790, 600, 790, 600, True),
                           (25.0, 28.0, 790, 600, 1250, 660, False)]),
        SimDiner(name="family-b", colour=(90, 200, 200),
                 segments=[(9.0, 12.0, 1250, 660, 870, 600, False),
                           (12.0, 25.0, 870, 600, 870, 600, True),
                           (25.0, 28.0, 870, 600, 1250, 680, False)]),
        # T4 turns over: first diner leaves, a second one arrives later.
        SimDiner(name="diner-1", colour=(230, 200, 60),
                 segments=[(0.0, 2.0, 1080, 600, 1080, 600, True),
                           (2.0, 11.0, 1080, 600, 1080, 600, True),
                           (11.0, 12.5, 1080, 600, 1250, 660, False)]),
        SimDiner(name="diner-2", colour=(120, 90, 220),
                 segments=[(15.0, 17.0, 30, 640, 1130, 600, False),
                           (17.0, 29.5, 1130, 600, 1130, 600, True)]),
        # Waiter: walks the room twice -- behind the tables, then right
        # through the T2 zone front. Moving, so never counts as seated.
        SimDiner(name="waiter", colour=(240, 240, 240),
                 segments=[(5.0, 9.0, 60, 340, 1220, 340, False),
                           (12.0, 15.0, 500, 660, 900, 660, False),
                           (18.0, 22.0, 1220, 340, 60, 340, False)]),
    ]


class CentroidTracker:
    """Minimal centroid tracker used by the simulation publisher."""

    def __init__(self, max_dist=90.0, max_age_s=1.0):
        self.max_dist = max_dist
        self.max_age_s = max_age_s
        self.next_id = 1
        self.tracks = {}   # id -> (cx, cy, last_t)

    def update(self, boxes, t):
        cents = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        used = set()
        out = []
        for (cx, cy) in cents:
            best_id, best_d = None, self.max_dist
            for tid, (tx, ty, tt) in self.tracks.items():
                if tid in used or t - tt > self.max_age_s:
                    continue
                d = math.hypot(cx - tx, cy - ty)
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = (cx, cy, t)
            used.add(best_id)
            out.append(best_id)
        self.tracks = {tid: v for tid, v in self.tracks.items()
                       if t - v[2] <= self.max_age_s}
        return out


class SimScene:
    """Renders the dining room and publishes its detection stream."""

    def __init__(self, seconds=30.0):
        self.seconds = seconds
        self.diners = build_sim_diners()
        self.tracker = CentroidTracker()

    def _draw_room(self, img):
        img[:] = (58, 50, 44)                        # floor
        # back wall + wainscot
        cv2.rectangle(img, (0, 64), (SIM_WIDTH, 220), (96, 78, 62), -1)
        cv2.rectangle(img, (0, 200), (SIM_WIDTH, 224), (70, 56, 44), -1)
        # header band
        cv2.rectangle(img, (0, 0), (SIM_WIDTH, 64), (30, 30, 34), -1)
        cv2.putText(img, "THE CORNER BISTRO  -  TABLE VIEW", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        # menu boards on the back wall
        for i, label in enumerate(["TODAY'S MENU", "DESSERTS", "DRINKS"]):
            x = 120 + i * 380
            cv2.rectangle(img, (x, 96), (x + 300, 176), (34, 30, 26), -1)
            cv2.rectangle(img, (x, 96), (x + 300, 176), (190, 170, 140), 2)
            cv2.putText(img, label, (x + 24, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 220, 200), 2)
        # windows
        for x in (980, 1120):
            cv2.rectangle(img, (x, 96), (x + 110, 176), (70, 90, 110), -1)
            cv2.rectangle(img, (x, 96), (x + 110, 176), (200, 200, 200), 2)
        # door (left) and serving hatch (right)
        cv2.rectangle(img, (24, 120), (84, 220), (52, 40, 30), -1)
        cv2.putText(img, "IN", (40, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1)
        cv2.rectangle(img, (1180, 120), (1256, 220), (52, 40, 30), -1)
        cv2.putText(img, "KITCHEN", (1184, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

    def _draw_table(self, img, table: TableState):
        x1, y1, x2, y2 = [int(v) for v in table.rect]
        # chairs: far side (behind the table) and near side (in front)
        mid_y = (y1 + y2) // 2
        for cx in (x1 + 55, (x1 + x2) // 2, x2 - 55):
            cv2.rectangle(img, (cx - 26, y1 + 8), (cx + 26, y1 + 52),
                          (74, 60, 48), -1)                       # far chair
            cv2.rectangle(img, (cx - 26, y2 - 52), (cx + 26, y2 - 8),
                          (74, 60, 48), -1)                       # near chair
        # table top with cloth + plates
        cv2.ellipse(img, ((x1 + x2) // 2, mid_y),
                    ((x2 - x1) // 2 - 8, 46), 0, 0, 360,
                    (196, 188, 172), -1)
        cv2.ellipse(img, ((x1 + x2) // 2, mid_y),
                    ((x2 - x1) // 2 - 8, 46), 0, 0, 360,
                    (120, 110, 96), 2)
        for px in (x1 + 55, (x1 + x2) // 2, x2 - 55):
            cv2.circle(img, (px, mid_y), 12, (245, 245, 245), -1)
            cv2.circle(img, (px, mid_y), 12, (160, 160, 160), 1)

    def _figure_geometry(self, diner: SimDiner, t: float):
        """Drawn extent of a diner: returns box or None."""
        x, y, seated, active = diner.pos(t)
        if not active:
            return None
        moving = any(t0 <= t <= t1 and (x0, y0) != (x1, y1)
                     for (t0, t1, x0, y0, x1, y1, s) in diner.segments)
        bob = math.sin(t * 9.0) * 2.0 if moving else 0.0
        h = 78.0 if seated else diner.height
        x1 = x - diner.width / 2.0
        x2 = x + diner.width / 2.0
        y2 = y + bob
        y1 = y2 - h
        return (x1, y1, x2, y2)

    def _draw_diner(self, img, diner: SimDiner, t: float):
        box = self._figure_geometry(diner, t)
        if box is None:
            return
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2.0
        _, _, seated, _ = diner.pos(t)
        moving = any(t0 <= t <= t1 and (a, b) != (c, d)
                     for (t0, t1, a, b, c, d, s) in diner.segments)
        head_r = 13
        head_cy = y1 + head_r
        # torso
        cv2.rectangle(img, (int(x1 + 6), int(y1 + 2 * head_r)),
                      (int(x2 - 6), int(y2 - 26)), diner.colour, -1)
        if seated:
            # seated: legs folded under the table, only the torso shows
            cv2.rectangle(img, (int(x1 + 6), int(y2 - 26)),
                          (int(x2 - 6), int(y2)), diner.colour, -1)
        else:
            step = int(math.sin(t * 9.0) * 7) if moving else 0
            cv2.line(img, (int(x - 8), int(y2 - 26)),
                     (int(x - 8 + step), int(y2)), diner.colour, 6)
            cv2.line(img, (int(x + 8), int(y2 - 26)),
                     (int(x + 8 - step), int(y2)), diner.colour, 6)
        # head
        cv2.circle(img, (int(x), int(head_cy)), head_r, (210, 180, 160), -1)
        # arm resting on the table when seated
        if seated:
            cv2.line(img, (int(x + 14), int(y1 + 2 * head_r + 10)),
                     (int(x + 30), int(y2 - 30)), diner.colour, 6)

    def frame(self, t: float, engine: TableTimingEngine):
        """Returns (image, [Detection, ...]) for sim time t."""
        img = np.zeros((SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
        self._draw_room(img)
        for table in engine.tables:
            self._draw_table(img, table)
        boxes = []
        for diner in self.diners:
            box = self._figure_geometry(diner, t)
            if box is not None:
                boxes.append(box)
        ids = self.tracker.update(boxes, t)
        for diner in self.diners:
            self._draw_diner(img, diner, t)
        dets = [Detection(b[0], b[1], b[2], b[3], "person", 0.99, tid)
                for b, tid in zip(boxes, ids)]
        return img, dets


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return "%02d:%02d" % (seconds // 60, seconds % 60)


def draw_overlay(img, engine: TableTimingEngine, dets: list, t: float,
                 source_label: str, frames_done: int):
    h, w = img.shape[:2]
    # table zones
    for table in engine.tables:
        x1, y1, x2, y2 = [int(v) for v in table.rect]
        colour = COLOUR_OCCUPIED if table.occupied else COLOUR_FREE
        thick = 3 if table.occupied else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        status = "OCCUPIED" if table.occupied else "FREE"
        if table.occupied and table.session_start is not None:
            label = "%s  %s  %s" % (table.name, status,
                                    fmt_clock(t - table.session_start))
        else:
            label = "%s  %s" % (table.name, status)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), colour, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        if table.people_now:
            cv2.putText(img, "%d seated" % table.people_now,
                        (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # person boxes (thin, cyan) with track id + stillness
    for d in dets:
        if d.cls != "person":
            continue
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      COLOUR_PERSON, 2)
        speed = engine.speed_at(d.track_id, t)
        label = "ID %d  %.0f px/s" % (d.track_id, speed)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(th + 6, int(d.y1) - 6)
        cv2.rectangle(img, (int(d.x1), ly - th - 6),
                      (int(d.x1) + tw + 8, ly + 2), COLOUR_PERSON, -1)
        cv2.putText(img, label, (int(d.x1) + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)

    # status board (top-left)
    occupied = sum(1 for tb in engine.tables if tb.occupied)
    lines = ["TABLE BOARD  --  %d/%d OCCUPIED   t=%s"
             % (occupied, len(engine.tables), fmt_clock(t))]
    for tb in engine.tables:
        if tb.occupied and tb.session_start is not None:
            lines.append("%s  OCCUPIED  %s   (%d seated, %d sessions)"
                         % (tb.name, fmt_clock(t - tb.session_start),
                            tb.people_now, len(tb.sessions)))
        else:
            lines.append("%s  FREE       --       (%d sessions)"
                         % (tb.name, len(tb.sessions)))
    total_sessions = sum(len(tb.sessions) for tb in engine.tables)
    lines.append("sessions so far: %d    %s    frame %d"
                 % (total_sessions, source_label, frames_done))
    bw = max(len(ln) for ln in lines) * 11 + 20
    bh = len(lines) * 24 + 16
    cv2.rectangle(img, (10, 10), (10 + bw, 10 + bh), COLOUR_BOARD, -1)
    cv2.rectangle(img, (10, 10), (10 + bw, 10 + bh), (200, 200, 200), 1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (20, 10 + 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_TEXT, 1)


# ---------------------------------------------------------------------------
# Real mode -- YOLOv8 + ByteTrack
# ---------------------------------------------------------------------------

def run_real(model, cap, engine, writer, args, source_label, total_frames):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
                              classes=[0], conf=args.conf,
                              imgsz=args.imgsz, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                if b.id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append(Detection(x1, y1, x2, y2, "person",
                                      float(b.conf[0]),
                                      int(b.id[0])))
        engine.update(dets, t)
        draw_overlay(frame, engine, dets, t, source_label, frame_idx + 1)
        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Restaurant Table-Timing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if total_frames and frame_idx % 60 == 0:
            print("  ... frame %d/%d (%.0f%%)"
                  % (frame_idx, total_frames, 100.0 * frame_idx / total_frames),
                  flush=True)
    engine.finish(frame_idx / fps if fps else 0.0)
    print("real mode: %d frames in %.1fs wall time"
          % (frame_idx, time.time() - t0), flush=True)
    return frame_idx


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def run_sim(engine, writer, args):
    scene = SimScene(seconds=args.seconds)
    n_frames = int(args.seconds * SIM_FPS)
    t0 = time.time()
    for i in range(n_frames):
        t = i / SIM_FPS
        frame, dets = scene.frame(t, engine)
        engine.update(dets, t)
        draw_overlay(frame, engine, dets, t, "SIMULATION", i + 1)
        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Restaurant Table-Timing (simulation)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    engine.finish(args.seconds)
    print("simulation: %d frames in %.1fs wall time"
          % (n_frames, time.time() - t0), flush=True)
    return n_frames


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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Restaurant table-occupancy analytics (YOLOv8 + ByteTrack).")
    ap.add_argument("--source", default=None,
                    help="video file, webcam index, or 'pick' for a file chooser "
                         "(default: opens the chooser)")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered dining-room simulation instead of a camera")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="simulation length in seconds (default 30)")
    ap.add_argument("--output", default="output.mp4",
                    help="annotated video out (default output.mp4)")
    ap.add_argument("--log", default="sessions.log",
                    help="session log out (default sessions.log)")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="YOLOv8 weights (default yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="detection confidence (default 0.30)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (default bytetrack.yaml)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference size (default 640; use 1280 for "
                         "small/far diners)")
    ap.add_argument("--tables", default=None,
                    help="JSON file with custom table zones "
                         '[{"name":"T1","x1":0.03,"y1":0.55,"x2":0.25,"y2":0.93}] '
                         "(fractions of the frame)")
    ap.add_argument("--occupy", type=float, default=OCCUPY_DELAY_S,
                    help="seconds of still presence before a table counts "
                         "occupied (default 1.5)")
    ap.add_argument("--free", type=float, default=FREE_DELAY_S,
                    help="seconds of absence before a table counts free "
                         "(default 3.0)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop real mode after this many video seconds (0 = all)")
    ap.add_argument("--no-display", action="store_true",
                    help="headless run (no preview window)")
    args = ap.parse_args(argv)

    tables = load_tables(args.tables) if args.tables else None

    if args.simulate:
        engine = TableTimingEngine(SIM_WIDTH, SIM_HEIGHT,
                                   SessionLog(args.log), tables,
                                   args.occupy, args.free)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, SIM_FPS,
                                 (SIM_WIDTH, SIM_HEIGHT))
        frames = run_sim(engine, writer, args)
        writer.release()
        source = "simulation"
    else:
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
        engine = TableTimingEngine(fw, fh, SessionLog(args.log), tables,
                                   args.occupy, args.free)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))
        frames = run_real(model, cap, engine, writer, args,
                          os.path.basename(str(source)), total)
        writer.release()
        cap.release()

    if not args.no_display:
        cv2.destroyAllWindows()

    s = engine.summary()
    print()
    print("=" * 64)
    print("RESTAURANT TABLE-TIMING -- %s" % source)
    print("=" * 64)
    print("frames processed      : %d" % s["frames"])
    print("people tracked        : %d" % s["people"])
    print("tables                : %d" % s["tables"])
    print("sessions (seat -> free): %d  %s" % (s["sessions"], s["sessions_by_table"]))
    print("average stay          : %.1f s" % s["avg_stay"])
    print("longest stay          : %.1f s" % s["longest_stay"])
    print("total occupied time   : %.1f s" % s["total_occupied_s"])
    print("peak tables occupied  : %d/%d" % (s["peak_occupied"], s["tables"]))
    print("events                : %s" % s["events"])
    print("annotated video       : %s" % args.output)
    print("session log           : %s" % args.log)
    print("=" * 64)
    for tb in engine.tables:
        for (st, en, dur) in tb.sessions:
            print("  %s  seated %5.1fs -> freed %5.1fs   stay %.1fs"
                  % (tb.name, st, en, dur))
    return 0


if __name__ == "__main__":
    sys.exit(main())
