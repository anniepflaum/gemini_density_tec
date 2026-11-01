"""Density slice and line-of-sight diagnostics using the TEC calculator geometry."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .cli import rows, position, utc, FRAME
from datetime import datetime, timedelta, timezone
from .model import Model
from .tec import Re, geographic, integrate


def ray_samples(receiver, rocket, step_m):
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError('step_m must be finite and positive')
    length = np.linalg.norm(rocket-receiver)
    s = np.linspace(0, length, max(1, int(np.ceil(length/step_m)))+1)
    points = receiver + np.linspace(0, 1, len(s))[:, None]*(rocket-receiver)
    return s, points


def slice_geometry(receiver, rocket, ground_km, altitude_km):
    """Radial plane through both endpoints; horizontal coordinate is surface arc length."""
    up = receiver/np.linalg.norm(receiver)
    direction = rocket - np.dot(rocket, up)*up
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('Vertical ray has no unique azimuthal slice plane')
    along = direction/np.linalg.norm(direction)
    angle = np.asarray(ground_km)*1000/Re
    radial = np.cos(angle)[:, None]*up + np.sin(angle)[:, None]*along
    points = (Re+np.asarray(altitude_km)*1000)[:, None, None]*radial[None, :, :]
    return points, up, along


def cumulative_tec(s, ne):
    """Only segments with two finite endpoints contribute; identical to integrate()."""
    valid = np.isfinite(ne[:-1]) & np.isfinite(ne[1:])
    increments = np.where(valid, .5*(ne[:-1]+ne[1:])*np.diff(s)/1e16, 0.)
    cumulative = np.r_[0., np.cumsum(increments)]
    # Hide values at unsupported points; this is modeled contribution, never total TEC.
    visible = cumulative.copy()
    visible[~np.isfinite(ne)] = np.nan
    return cumulative, visible, valid


class LOSViewer:
    def __init__(self, directory, *, time_source="manifest", trajectories=None, receivers=None):
        self.directory = Path(directory)
        self.model = Model(directory)
        if time_source == "filename":
            self.frames = []
            for path in sorted(self.directory.glob("*.h5")):
                match = FRAME.fullmatch(path.name)
                if match:
                    time = datetime.strptime(match[1], "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(seconds=float(match[2]))
                    self.frames.append(dict(filename=path.name, model_time_utc=time.isoformat()))
        elif time_source == "manifest":
            self.frames = sorted(rows(self.directory/'frame_times.csv'), key=lambda r: utc(r['model_time_utc']))
        else:
            raise ValueError("time_source must be filename or manifest")
        if not self.frames:
            raise ValueError("No output frames found")
        self.receivers = {r['receiver_id']: position(r) for r in rows(receivers or self.directory/'receivers.csv')}
        self.tracks = {}
        for r in rows(trajectories or self.directory/'trajectories.csv'):
            self.tracks.setdefault(r['rocket_id'], []).append((utc(r['time_utc']).timestamp(), position(r)))
        for key, track in self.tracks.items():
            track.sort(key=lambda item: item[0])
            t = np.array([p[0] for p in track])
            if np.any(np.diff(t) <= 0):
                raise ValueError(f'Duplicate trajectory times for {key}')
            self.tracks[key] = t, np.array([p[1] for p in track])
        self._frame = None
        self._density = None

    def plot(self, index=3, receiver_id='VEE', rocket_id='397', step_m=500,
             density_limits=(1e8, 2e12), altitude_max_km=500, export=None):
        frame = self.frames[index]
        time = utc(frame['model_time_utc'])
        times, points = self.tracks[str(rocket_id)]
        target = time.timestamp()
        i = np.searchsorted(times, target)
        if not times[0] <= target <= times[-1]:
            raise ValueError('Selected time is outside trajectory coverage')
        if times[i] != target and times[i]-times[i-1] > 10:
            raise ValueError('Trajectory gap exceeds calculator limit of 10 seconds')
        rocket = np.array([np.interp(target, times, points[:, j]) for j in range(3)])
        receiver = self.receivers[receiver_id]
        path = self.directory/frame['filename']
        if path != self._frame:
            self._density = None  # release previous frame before loading another
            self._density = self.model.density(path)
            self._frame = path
        def sample(alt, lon, lat):
            return self._density(self.model.coordinates(alt, lon, lat))
        s, ray = ray_samples(receiver, rocket, step_m)
        alt, lon, lat = geographic(ray)
        ne = sample(alt, lon, lat)
        result = integrate(receiver, rocket, sample, step_m)
        if result['status'] == 'earth_blocked':
            raise ValueError('Line of sight intersects the Earth')
        cumulative, visible, valid = cumulative_tec(s, ne)
        if np.isfinite(result['modeled_stec_tecu']):
            np.testing.assert_allclose(cumulative[-1], result['modeled_stec_tecu'], rtol=1e-12)
        angular_span = np.arccos(np.clip(np.dot(receiver, rocket)/np.linalg.norm(receiver)/np.linalg.norm(rocket), -1, 1))
        ground_end = Re*angular_span/1000
        margin = max(20., .15*ground_end)
        ground = np.linspace(-margin, ground_end+margin, 400)
        heights = np.linspace(0, max(altitude_max_km, alt.max()/1000+30), 350)
        plane, up, along = slice_geometry(receiver, rocket, ground, heights)
        a, o, l = geographic(plane.reshape(-1, 3))
        density_slice = sample(a, o, l).reshape(len(heights), len(ground))
        ray_ground = Re*np.arctan2(ray@along, ray@up)/1000
        fig, (ax, profile) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        ax.set_facecolor('#d6d6d6')
        color = ax.pcolormesh(ground, heights, np.ma.masked_invalid(density_slice),
                              norm=LogNorm(*density_limits), cmap='viridis', shading='auto', rasterized=True)
        fig.colorbar(color, ax=ax, label='Electron density [m⁻³]', extend='both')
        ax.plot(ray_ground, alt/1000, color='white', linewidth=3.5)
        ax.plot(ray_ground, alt/1000, color='crimson', linewidth=1.7, label='Straight Cartesian ray')
        ax.scatter(ray_ground[[0,-1]], alt[[0,-1]]/1000, c=['black','crimson'], marker='o', zorder=5)
        ax.annotate(receiver_id, (ray_ground[0], alt[0]/1000), xytext=(6, 9), textcoords='offset points')
        ax.annotate(f'Rocket {rocket_id}', (ray_ground[-1], alt[-1]/1000), xytext=(6, 9), textcoords='offset points')
        ax.set(xlabel='Ground arc distance from receiver [km]', ylabel='Altitude [km]',
               title='Density in the ray plane • gray = outside model')
        ax.legend(loc='upper left')
        profile.plot(s/1000, ne, color='tab:blue', label='Electron density')
        profile.set_yscale('log')
        profile.set_ylim(*density_limits)
        profile.set(xlabel='Distance along ray from receiver [km]', ylabel='Electron density [m⁻³]')
        profile.tick_params(axis='y', colors='tab:blue')
        accumulated = profile.twinx()
        accumulated.plot(s/1000, visible, color='tab:orange', label='Accumulated modeled TEC')
        accumulated.set_ylabel('Accumulated modeled TEC [TECU]', color='tab:orange')
        accumulated.set_ylim(bottom=0)
        # Shade contiguous unsupported segments rather than implying zero density.
        bad = ~valid
        edges = np.diff(np.r_[False, bad, False].astype(int))
        for start, stop in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
            profile.axvspan(s[start]/1000, s[stop]/1000, color='gray', alpha=.25)
        profile.grid(alpha=.2)
        profile.set_title(f"Modeled TEC = {result['modeled_stec_tecu']:.5f} TECU\n"
                          f"Path coverage = {result['coverage_fraction']:.2%} ({result['status']})")
        fig.suptitle(f"{receiver_id} → {rocket_id} | {time:%Y-%m-%d %H:%M:%S} UTC\n"
                     'Spherical GEMINI geometry; gray path intervals are excluded, not zero density', fontsize=12)
        if export:
            export = Path(export)
            export.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(export, dpi=180)
        return fig, dict(**result, source_filename=frame['filename'], model_time_utc=time.isoformat(),
                         distance_m=s, electron_density_m3=ne, cumulative_modeled_tecu=cumulative)
