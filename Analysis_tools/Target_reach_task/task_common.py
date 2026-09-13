"""
task_common.py
Shared 2D target-reaching task engine for arm_eazy input-method comparison.

Draws a pygame window with a moving cursor and a target circle. A target
counts as "reached" once the cursor stays inside it for HOLD_TIME seconds.
Tracks, per target: completion time, path length traveled, straight-line
distance, path efficiency (straight/path), and final positional error.

Used by gamepad_reach_task.py and phone_reach_task.py, which are only
responsible for feeding cursor movement in each frame.
"""

import csv
import math
import random
import time

import pygame

WIDTH, HEIGHT = 800, 600
CURSOR_RADIUS = 8
TARGET_RADIUS = 22
HOLD_TIME = 0.6          # seconds cursor must stay inside target to count as reached
MARGIN = 60              # keep targets away from window edges
BG_COLOR = (18, 18, 24)
CURSOR_COLOR = (80, 200, 255)
TARGET_COLOR = (255, 140, 60)
TARGET_HIT_COLOR = (90, 220, 120)
TEXT_COLOR = (230, 230, 230)


class TargetTask:
    def __init__(self, out_csv, n_targets=8, seed=None, extra_header=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("arm_eazy input-method reach task")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()

        if seed is not None:
            random.seed(seed)

        self.n_targets = n_targets
        self.target_index = 0
        self.done = False
        self.results = []

        self.cursor_x = WIDTH / 2
        self.cursor_y = HEIGHT / 2

        self.session_start = time.time()
        self.out_csv = out_csv
        self.csv_file = open(out_csv, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.extra_header = extra_header or []
        header = [
            "timestamp", "elapsed_s", "target_index", "cursor_x", "cursor_y",
            "target_x", "target_y", "distance", "hit_event",
        ] + self.extra_header
        self.writer.writerow(header)

        self._spawn_target()

    def _new_target_pos(self):
        return (
            random.uniform(MARGIN, WIDTH - MARGIN),
            random.uniform(MARGIN, HEIGHT - MARGIN),
        )

    def _spawn_target(self):
        self.target_x, self.target_y = self._new_target_pos()
        self.origin_x, self.origin_y = self.cursor_x, self.cursor_y
        self.path_length = 0.0
        self.prev_x, self.prev_y = self.cursor_x, self.cursor_y
        self.target_start_time = time.time()
        self.inside_since = None

    def move_cursor_to(self, x, y):
        x = max(CURSOR_RADIUS, min(WIDTH - CURSOR_RADIUS, x))
        y = max(CURSOR_RADIUS, min(HEIGHT - CURSOR_RADIUS, y))
        self.path_length += math.hypot(x - self.prev_x, y - self.prev_y)
        self.prev_x, self.prev_y = x, y
        self.cursor_x, self.cursor_y = x, y

    def nudge_cursor(self, dx, dy):
        self.move_cursor_to(self.cursor_x + dx, self.cursor_y + dy)

    def _distance_to_target(self):
        return math.hypot(self.cursor_x - self.target_x, self.cursor_y - self.target_y)

    def update(self, extra_values=None):
        """Call once per frame after moving the cursor.
        Returns False once all targets are completed (task finished)."""
        if self.done:
            return False

        dist = self._distance_to_target()
        now = time.time()
        hit_event = ""

        if dist <= TARGET_RADIUS:
            if self.inside_since is None:
                self.inside_since = now
            elif now - self.inside_since >= HOLD_TIME:
                completion_time = now - self.target_start_time
                straight_dist = math.hypot(
                    self.target_x - self.origin_x, self.target_y - self.origin_y
                )
                self.results.append({
                    "target_index": self.target_index,
                    "completion_time_s": round(completion_time, 3),
                    "path_length_px": round(self.path_length, 1),
                    "straight_line_px": round(straight_dist, 1),
                    "final_error_px": round(dist, 2),
                })
                hit_event = "TARGET_HIT"
                self.target_index += 1
                if self.target_index >= self.n_targets:
                    self.done = True
                else:
                    self._spawn_target()
        else:
            self.inside_since = None

        self.log_row(hit_event, extra_values)
        return not self.done

    def log_row(self, hit_event="", extra_values=None):
        now = time.time()
        row = [
            now, round(now - self.session_start, 4),
            self.target_index, round(self.cursor_x, 2), round(self.cursor_y, 2),
            round(self.target_x, 2), round(self.target_y, 2),
            round(self._distance_to_target(), 2), hit_event,
        ] + (extra_values if extra_values is not None else [""] * len(self.extra_header))
        self.writer.writerow(row)

    def draw(self, hud_lines=None):
        self.screen.fill(BG_COLOR)
        color = TARGET_HIT_COLOR if self.inside_since else TARGET_COLOR
        pygame.draw.circle(self.screen, color, (int(self.target_x), int(self.target_y)), TARGET_RADIUS, 3)
        pygame.draw.circle(self.screen, CURSOR_COLOR, (int(self.cursor_x), int(self.cursor_y)), CURSOR_RADIUS)

        lines = [f"Target {self.target_index + 1}/{self.n_targets}  (ESC to quit early)"] + (hud_lines or [])
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (10, 10 + i * 22))

        pygame.display.flip()

    def print_summary(self):
        print("\n--- Task summary ---")
        if not self.results:
            print("No targets completed.")
            return
        for r in self.results:
            eff = (r["straight_line_px"] / r["path_length_px"]) if r["path_length_px"] > 0 else 0.0
            print(
                f"Target {r['target_index']+1}: time={r['completion_time_s']}s  "
                f"path={r['path_length_px']}px  straight={r['straight_line_px']}px  "
                f"efficiency={eff:.2f}  final_error={r['final_error_px']}px"
            )
        times = [r["completion_time_s"] for r in self.results]
        effs = [
            (r["straight_line_px"] / r["path_length_px"]) if r["path_length_px"] > 0 else 0.0
            for r in self.results
        ]
        print(f"\nTargets completed: {len(self.results)}/{self.n_targets}")
        print(f"Mean completion time: {sum(times)/len(times):.3f}s")
        print(f"Mean path efficiency:  {sum(effs)/len(effs):.3f}  (1.0 = perfectly straight path)")
        print(f"Log saved to {self.out_csv}")

    def close(self):
        self.csv_file.close()
        pygame.quit()
