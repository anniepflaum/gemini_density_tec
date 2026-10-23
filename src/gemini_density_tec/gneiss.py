"""Prepare the user-selected GNEISS run and native GPS/receiver CSV inputs."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

RUN = 'gneiss_20260210_35400_09_SD_UM_xa_gneiss'
URL = 'https://rcweb.dartmouth.edu/LynchK/Gemini3D/' + RUN + '/'
RAW = Path('/Users/anniepflaum/lab317/data/raw')


def write_csv(path, fields, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def prepare(destination, trajectory_dir, receiver_file):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    sources = []
    for rocket, filename in [('397', '36397_GPS_Time_Export_01.csv'), ('398', '36398_GPS_Time_Export_00.csv')]:
        source = trajectory_dir / filename
        sources.append(str(source))
        with source.open(newline='') as f:
            for row in csv.DictReader(f):
                token = row['Time (UTC / From GPS Receiver)'].strip()
                if not token:
                    continue
                day, clock = token.split()
                t = datetime.strptime('2026 ' + day + ' ' + clock, '%Y %j %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                records.append(dict(time_utc=t.isoformat(), rocket_id=rocket,
                                    lat_deg=float(row['Latitude']), lon_deg=float(row['Longitude']),
                                    alt_m=1000*float(row['Altitude (km)'])))
    write_csv(destination/'trajectories.csv', ['time_utc','rocket_id','lat_deg','lon_deg','alt_m'], records)
    with receiver_file.open(newline='') as f:
        receivers = [dict(receiver_id=r['acronym'].strip(), lat_deg=float(r['Lat']), lon_deg=float(r['Lon']), alt_m=float(r['Alt_m'])) for r in csv.DictReader(f)]
    write_csv(destination/'receivers.csv', ['receiver_id','lat_deg','lon_deg','alt_m'], receivers)
    start = datetime(2026, 2, 10, 10, 24, tzinfo=timezone.utc)
    frames = [dict(filename=f'20260210_{35345+i*5:05d}.000000.h5', model_time_utc=(start+timedelta(seconds=5*i)).isoformat()) for i in range(12)]
    write_csv(destination/'frame_times.csv', ['filename','model_time_utc'], frames)
    (destination/'inputs.json').write_text(json.dumps(dict(source_url=URL, trajectories=sources, receivers=str(receiver_file), time_rule='Exclude 20260210_35340.000000.h5 (blank-run copy); 20260210_35345.000000.h5 is 2026-02-10T10:24:00Z, then every 5 seconds; user correction', rocket_ids=['397','398'], receiver_ids=[r['receiver_id'] for r in receivers]), indent=2)+'\n')
    print(f'Prepared {len(records)} trajectory samples, {len(receivers)} receivers, {len(frames)} corrected frame times in {destination}')
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=Path('data')/RUN)
    p.add_argument('--trajectory-dir', type=Path, default=RAW/'rocket/trajectories/GNEISS')
    p.add_argument('--receiver-file', type=Path, default=RAW/'receiver-data/asi-mapping/receivers.csv')
    p.add_argument('--download', action='store_true', help='Download/resume grid and all 12 science output frames (tens of GB)')
    args = p.parse_args()
    frames = prepare(args.destination, args.trajectory_dir, args.receiver_file)
    if args.download:
        for name in ['inputs/simsize.h5', 'inputs/simgrid.h5'] + [f['filename'] for f in frames]:
            target = args.destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            partial = target.with_suffix(target.suffix+'.part')
            subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-', '--output',str(partial),URL+name], check=True)
            partial.replace(target)


if __name__ == '__main__':
    main()
