"""CSV-driven calculation at each available GEMINI output time."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import numpy as np
from .model import Model
from .tec import xyz, integrate

FRAME = re.compile(r"(\d{8})_(\d{5}\.\d{6})\.h5$")


def utc(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("CSV times must include Z or a UTC offset")
    return t.astimezone(timezone.utc)


def rows(path):
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    if not data:
        raise ValueError(f"Empty input: {path}")
    return data


def position(row):
    values = [float(row[k]) for k in ("lat_deg", "lon_deg", "alt_m")]
    if not np.isfinite(values).all() or not -90 <= values[0] <= 90 or values[2] < 0:
        raise ValueError(f"Invalid position: {row}")
    return xyz(*values)


def run(args):
    receivers = {r["receiver_id"]: position(r) for r in rows(args.receivers)}
    if len(receivers) != len(rows(args.receivers)):
        raise ValueError("Duplicate receiver IDs")
    tracks = {}
    for row in rows(args.trajectories):
        tracks.setdefault(row["rocket_id"], []).append((utc(row["time_utc"]).timestamp(), position(row)))
    for key, samples in tracks.items():
        samples.sort(key=lambda item: item[0])
        times = np.array([s[0] for s in samples])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Duplicate trajectory times for {key}")
        tracks[key] = times, np.array([s[1] for s in samples])
    corrected = {}
    if not getattr(args, "frame_times", None) and (args.model / "frame_times.csv").exists():
        args.frame_times = args.model / "frame_times.csv"
    if getattr(args, "frame_times", None):
        for row in rows(args.frame_times):
            if row["filename"] in corrected:
                raise ValueError("Duplicate filename in frame-time manifest")
            corrected[row["filename"]] = utc(row["model_time_utc"])
    frames = []
    for f in sorted(args.model.glob("*.h5")):
        match = FRAME.fullmatch(f.name)
        if match:
            t = datetime.strptime(match[1], "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(seconds=float(match[2]))
            if corrected:
                if f.name not in corrected:
                    continue  # The manifest explicitly selects the science frames.
                t = corrected[f.name]
            if (args.start is None or t >= utc(args.start)) and (args.end is None or t <= utc(args.end)):
                frames.append((f, t))
    if not frames:
        raise ValueError("No timestamped output frames selected")
    model = Model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_filename", "model_time_utc", "trajectory_time_utc", "rocket_id", "receiver_id", "stec_tecu", "modeled_stec_tecu", "coverage_fraction", "path_m", "status"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame, time in frames:
            target = time + timedelta(seconds=args.trajectory_offset_seconds)
            stamp = target.timestamp()
            positions = {}
            for rocket, (times, points) in tracks.items():
                if times[0] <= stamp <= times[-1]:
                    idx = np.searchsorted(times, stamp)
                    exact = idx < len(times) and times[idx] == stamp
                    if exact or (0 < idx < len(times) and times[idx]-times[idx-1] <= args.max_gap_seconds):
                        positions[rocket] = np.array([np.interp(stamp, times, points[:, i]) for i in range(3)])
            density = model.density(frame) if positions else None
            print(f"{time.isoformat()}: {len(positions)}/{len(tracks)} rocket positions", flush=True)
            for rocket in tracks:
                for receiver, location in receivers.items():
                    result = dict(stec_tecu=np.nan, modeled_stec_tecu=np.nan, coverage_fraction=np.nan, path_m=np.nan, status="missing_trajectory")
                    if rocket in positions:
                        result = integrate(location, positions[rocket], lambda a, o, l: density(model.coordinates(a, o, l)), args.step_m)
                    writer.writerow(dict(source_filename=frame.name, model_time_utc=time.isoformat(), trajectory_time_utc=target.isoformat(), rocket_id=rocket, receiver_id=receiver, **result))
            f.flush()
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(pygemini_commit="6168ba3cebdfd43690f1eecf30404cf482ab860b", earth_radius_m=6370000, geometry="spherical geographic; straight Cartesian rays", frames=[p.name for p, _ in frames])
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--receivers", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/slant_tec.csv"))
    p.add_argument("--step-m", type=float, default=500)
    p.add_argument("--max-gap-seconds", type=float, default=10)
    p.add_argument("--trajectory-offset-seconds", type=float, default=0,
                   help="Trajectory UTC = model UTC + this explicit offset")
    p.add_argument("--frame-times", type=Path, help="CSV selecting frames and mapping filename to corrected model_time_utc; auto-detected in model directory")
    p.add_argument("--start", help="Inclusive model UTC (ISO 8601)")
    p.add_argument("--end", help="Inclusive model UTC (ISO 8601)")
    args = p.parse_args()
    if not all(np.isfinite(v) for v in (args.step_m, args.max_gap_seconds, args.trajectory_offset_seconds)) or min(args.step_m, args.max_gap_seconds) <= 0:
        p.error("Step and maximum gap must be positive; numeric options must be finite")
    run(args)


if __name__ == "__main__":
    main()
