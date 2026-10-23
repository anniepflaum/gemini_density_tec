"""Plot TEC CSV results with common UTC and TEC limits across all panels."""
import argparse
import csv
from datetime import timezone
from pathlib import Path
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .cli import utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, default=Path('results/gneiss_slant_tec.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/gneiss_slant_tec.png'))
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Fixed TECU limits; default is zero to the global maximum plus 5 percent')
    parser.add_argument('--individual-dir', type=Path, help='Also save one plot per rocket/receiver pair with the same limits')
    args = parser.parse_args()
    with args.csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error('CSV is empty')
    groups = {}
    seen = set()
    for r in rows:
        key = (r['rocket_id'], r['receiver_id'])
        time = utc(r['model_time_utc'])
        identity = (*key, time)
        if identity in seen:
            parser.error(f'Duplicate pair/time: {identity}')
        seen.add(identity)
        value = float(r['modeled_stec_tecu'])
        groups.setdefault(key, []).append((time, value))
    for values in groups.values():
        values.sort(key=lambda v: v[0])
    times = [t for values in groups.values() for t, _ in values]
    finite = [v for values in groups.values() for _, v in values if np.isfinite(v)]
    if not finite:
        parser.error('No finite modeled TEC values to plot')
    ylim = args.ylim or [0, max(max(finite)*1.05, .01)]
    if not np.isfinite(ylim).all() or ylim[0] >= ylim[1]:
        parser.error('--ylim requires two finite increasing values')
    tmin, tmax = min(times), max(times)
    if tmin == tmax:
        from datetime import timedelta
        tmin -= timedelta(seconds=1)
        tmax += timedelta(seconds=1)
    rockets = sorted({k[0] for k in groups})
    receivers = sorted({k[1] for k in groups})
    colors = {r: plt.get_cmap('tab10')(i % 10) for i, r in enumerate(rockets)}
    def draw(ax, rocket, receiver):
        data = groups.get((rocket, receiver))
        if data:
            ax.plot([x[0] for x in data], [x[1] for x in data], '.-', color=colors[rocket], label=f'Rocket {rocket}')
    def format_axis(ax):
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(*ylim)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5, tz=timezone.utc)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
        ax.tick_params(axis='x', rotation=30, labelbottom=True)
        ax.set_xlabel('UTC')
        ax.set_ylabel('Modeled slant TEC [TECU]')
        ax.grid(alpha=.25)
    ncols = min(3, len(receivers))
    fig, axes = plt.subplots(math.ceil(len(receivers)/ncols), ncols,
                             figsize=(5*ncols, 3.5*math.ceil(len(receivers)/ncols)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for ax, receiver in zip(axes.flat, receivers):
        for rocket in rockets:
            draw(ax, rocket, receiver)
        format_axis(ax)
        ax.set_title(receiver)
        ax.legend(fontsize='small')
    for ax in list(axes.flat)[len(receivers):]:
        ax.set_visible(False)
    fig.suptitle(f'Modeled slant TEC • {tmin:%Y-%m-%d} • common axes\nModel-domain contribution only; see CSV coverage_fraction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    if args.individual_dir:
        args.individual_dir.mkdir(parents=True, exist_ok=True)
        for rocket, receiver in sorted(groups):
            fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            draw(ax, rocket, receiver)
            format_axis(ax)
            ax.set_title(f'{receiver} / {rocket} • {tmin:%Y-%m-%d}\nModeled contribution only')
            safe = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{receiver}_{rocket}')
            fig.savefig(args.individual_dir/f'{safe}.png', dpi=200)
            plt.close(fig)
    print(f'Saved {args.output}; all axes: {tmin.isoformat()} to {tmax.isoformat()}, {ylim[0]:g}–{ylim[1]:g} TECU')
    if args.individual_dir:
        print(f'Saved {len(groups)} pair plots to {args.individual_dir}')


if __name__ == '__main__':
    main()
