"""
gamepad_reach_task.py
Target-reaching task driven by gamepad sticks (rate/velocity control).
Cursor velocity = axis value x max speed. Reach each target and hold for
0.6s to complete it. Logs completion time, path efficiency, and final
error per target to CSV, for direct comparison against phone_reach_task.py
and literature-reported 3D mouse metrics.

Requirements:
    pip install pygame

Usage:
    python gamepad_reach_task.py
    python gamepad_reach_task.py --targets 10 --speed 400 --out gamepad_task_log.csv
    python gamepad_reach_task.py --x-axis 0 --y-axis 1   (override stick axis mapping)
"""

import argparse
import sys
import time

import pygame

from task_common import TargetTask, WIDTH, HEIGHT


def main():
    parser = argparse.ArgumentParser(description="Gamepad target-reaching task.")
    parser.add_argument("--targets", type=int, default=8, help="Number of targets (default: 8)")
    parser.add_argument("--speed", type=float, default=350.0, help="Max cursor speed in px/s (default: 350)")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone (default: 0.08)")
    parser.add_argument("--x-axis", type=int, default=0, help="Joystick axis index for X (default: 0)")
    parser.add_argument("--y-axis", type=int, default=1, help="Joystick axis index for Y (default: 1)")
    parser.add_argument("--out", type=str, default="gamepad_task_log.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for target positions")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Plug in a controller and try again.")
        sys.exit(1)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Using controller: {joystick.get_name()}")

    extra_header = ["axis_x_raw", "axis_y_raw"]
    task = TargetTask(args.out, n_targets=args.targets, seed=args.seed, extra_header=extra_header)

    print(f"Reach {args.targets} targets. Hold cursor inside each for 0.6s to complete it.")
    print("ESC or close window to quit early.\n")

    running = True
    last_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = time.time()
        dt = now - last_time
        last_time = now

        pygame.event.pump()
        ax = joystick.get_axis(args.x_axis)
        ay = joystick.get_axis(args.y_axis)

        ax_used = ax if abs(ax) > args.deadzone else 0.0
        ay_used = ay if abs(ay) > args.deadzone else 0.0

        task.nudge_cursor(ax_used * args.speed * dt, ay_used * args.speed * dt)

        still_running = task.update(extra_values=[round(ax, 5), round(ay, 5)])

        hud = [
            f"axis_x={ax:+.2f}  axis_y={ay:+.2f}",
            f"targets completed: {len(task.results)}/{args.targets}",
        ]
        task.draw(hud_lines=hud)
        task.clock.tick(60)

        if not still_running:
            running = False

    task.print_summary()
    task.close()


if __name__ == "__main__":
    main()
