"""Ray geometry and conservative handling of out-of-domain samples."""
import numpy as np
from gemini3d.grid.convert import Re


def xyz(lat, lon, alt):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return (Re + np.asarray(alt))[..., None] * np.stack(
        (np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


def geographic(points):
    radius = np.linalg.norm(points, axis=-1)
    return radius-Re, np.degrees(np.arctan2(points[..., 1], points[..., 0])), np.degrees(np.arcsin(points[..., 2]/radius))


def integrate(receiver, rocket, sample_density, step_m=500):
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be finite and positive")
    length = float(np.linalg.norm(rocket-receiver))
    n = max(1, int(np.ceil(length/step_m)))
    s = np.linspace(0, length, n+1)
    points = receiver + np.linspace(0, 1, n+1)[:, None]*(rocket-receiver)
    # Analytic closest point to the Earth's center, not just sampled nodes.
    delta = rocket-receiver
    t = np.clip(-np.dot(receiver, delta)/length**2, 0, 1) if length else 0
    if np.linalg.norm(receiver+t*delta) < Re-1e-5:
        return dict(stec_tecu=np.nan, modeled_stec_tecu=np.nan, coverage_fraction=0., path_m=length, status="earth_blocked")
    ne = np.asarray(sample_density(*geographic(points)))
    good = np.isfinite(ne[:-1]) & np.isfinite(ne[1:])
    ds = np.diff(s)
    partial = float(np.sum((0.5*(ne[:-1]+ne[1:])*ds)[good])/1e16)
    coverage = float(ds[good].sum()/length) if length else float(np.isfinite(ne).all())
    complete = bool(np.isfinite(ne).all())
    return dict(stec_tecu=partial if complete else np.nan,
                modeled_stec_tecu=partial if good.any() or complete else np.nan,
                coverage_fraction=coverage, path_m=length,
                status="complete" if complete else ("partial" if good.any() else "outside_grid"))
