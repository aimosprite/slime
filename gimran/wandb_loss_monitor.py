#!/usr/bin/env python3
"""Fetch latest losses from W&B and plot in the terminal (pure ASCII)."""

import wandb

PROJECT = "aimosprite/slime-dev"
N_LATEST = 5
PLOT_WIDTH = 70
PLOT_HEIGHT = 18


def ascii_plot(series_list, width=PLOT_WIDTH, height=PLOT_HEIGHT):
    """Plot multiple series as pure ASCII. series_list: [(name, marker, [(x,y), ...])]"""
    all_x = [x for _, _, pts in series_list for x, y in pts]
    all_y = [y for _, _, pts in series_list for x, y in pts]
    if not all_x:
        return "No data"

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.05 or 1.0
    y_min -= y_pad
    y_max += y_pad

    grid = [[' '] * width for _ in range(height)]

    def to_col(x):
        if x_max == x_min:
            return width // 2
        return int((x - x_min) / (x_max - x_min) * (width - 1))

    def to_row(y):
        if y_max == y_min:
            return height // 2
        return height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))

    for name, marker, pts in series_list:
        for x, y in pts:
            c, r = to_col(x), to_row(y)
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = marker

    y_label_w = max(len(f"{y_max:.1f}"), len(f"{y_min:.1f}")) + 1
    lines = []
    legend = "  ".join(f"{m} {n}" for n, m, _ in series_list)
    lines.append(" " * y_label_w + "  " + legend)
    lines.append("")

    for i, row in enumerate(grid):
        if i == 0:
            label = f"{y_max:.1f}"
        elif i == height - 1:
            label = f"{y_min:.1f}"
        elif i == height // 2:
            label = f"{(y_max + y_min) / 2:.1f}"
        else:
            label = ""
        lines.append(f"{label:>{y_label_w}} |{''.join(row)}|")

    lines.append(" " * y_label_w + " +" + "-" * width + "+")
    x_axis = " " * y_label_w + f"  {x_min:<{width // 2}}{x_max:>{width - width // 2}}"
    lines.append(x_axis)
    lines.append(" " * (y_label_w + width // 2) + "Step")
    return "\n".join(lines)


def report_run(api, run_id):
    """Report on a single run with losses and plot."""
    run = api.run(f"{PROJECT}/{run_id}")
    hist = run.history(samples=500, pandas=False)

    train_points = [(r["_step"], r["train/loss"]) for r in hist if "train/loss" in r and r["train/loss"] is not None]
    test_points = [(r["_step"], r["test/loss"]) for r in hist if "test/loss" in r and r["test/loss"] is not None]

    print(f"\n{'='*60}")
    print(f"Run: {run.name}")
    print(f"ID:  {run.id}  |  State: {run.state}  |  Created: {run.created_at}")
    print(f"URL: https://wandb.ai/{PROJECT}/runs/{run.id}")
    print(f"{'='*60}")

    if train_points:
        print(f"\nLatest {N_LATEST} train/loss:")
        for step, loss in train_points[-N_LATEST:]:
            print(f"  step {step:>5}: {loss:.4f}")

    if test_points:
        print(f"\nLatest {N_LATEST} test/loss:")
        for step, loss in test_points[-N_LATEST:]:
            print(f"  step {step:>5}: {loss:.4f}")

    # Throughput
    timed_points = [(r["_timestamp"], r["_step"]) for r in hist if "_timestamp" in r and "_step" in r and r["_timestamp"] is not None]
    if len(timed_points) >= 2:
        timed_points.sort(key=lambda x: x[0])
        # Use last 10 points for recent throughput
        recent = timed_points[-10:]
        dt = recent[-1][0] - recent[0][0]
        ds = recent[-1][1] - recent[0][1]
        if dt > 0 and ds > 0:
            steps_per_min = ds / dt * 60
            secs_per_step = dt / ds
            print(f"\nThroughput: {steps_per_min:.1f} steps/min ({secs_per_step:.1f}s/step)")

    # Checkpoint info
    save_interval = run.config.get("save_interval", None)
    if save_interval and hist:
        latest_step = max((r["_step"] for r in hist if "_step" in r), default=0)
        last_ckpt = (latest_step // save_interval) * save_interval
        next_ckpt = last_ckpt + save_interval
        steps_left = next_ckpt - latest_step
        print(f"Checkpoints (every {save_interval} steps): last={last_ckpt}, next={next_ckpt} ({steps_left} steps away)", end="")
        if len(timed_points) >= 2 and dt > 0 and ds > 0:
            eta_mins = steps_left * secs_per_step / 60
            print(f" (~{eta_mins:.0f}min)")
        else:
            print()

    # Plot
    series = []
    if train_points:
        series.append(("train/loss", "*", train_points))
    if test_points:
        series.append(("test/loss", "o", test_points))
    if series:
        print()
        print(ascii_plot(series))


def main():
    api = wandb.Api()

    # Find all active runs
    active_runs = list(api.runs(PROJECT, filters={"state": "running"}, per_page=20))
    finished_recent = list(api.runs(PROJECT, filters={"state": "finished"}, per_page=3, order="-created_at"))

    # Report new/recently finished runs
    if finished_recent:
        recent_ids = [r.id for r in finished_recent]
        print(f"Recently finished runs: {', '.join(r.name + ' (' + r.id + ')' for r in finished_recent)}")

    if not active_runs:
        print("\nNo active runs in aimosprite/slime-dev")
        return

    print(f"\nActive runs: {len(active_runs)}")
    for run in active_runs:
        report_run(api, run.id)


if __name__ == "__main__":
    main()
