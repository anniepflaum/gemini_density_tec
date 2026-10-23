"""Pinned pygemini coordinates and selective HDF5 electron-density reads."""
from pathlib import Path
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from gemini3d import read, LSP
from gemini3d.grid.convert import Re, geog2geomag
from gemini3d.grid.gridmodeldata import geog2dipole


class Model:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.grid = read.grid(self.directory, var={"x1", "x2", "x3"})
        self.axes = tuple(self.grid[k][2:-2] for k in ("x1", "x2", "x3"))
        if any(len(a) < 2 for a in self.axes):
            raise ValueError("Only three-dimensional grids are supported")
        # Small slices avoid loading multi-GB grid metrics. HDF5 order is x3,x2,x1.
        with h5py.File(self.grid["filename"], "r") as f:
            h = np.array([f["h1"][i, j, k] for i in (0, -1) for j in (0, -1) for k in (0, -1)])
            self.dipole = not np.allclose(h, 1, atol=1e-4, rtol=0)
            self.phi0 = self.theta0 = None
            if not self.dipole:
                theta = f["theta"][:, 0, 0]
                phi = np.unwrap(f["phi"][0, :, 0])
                # Grid theta/phi can include ghost cells or only interior cells.
                x2 = self.grid["x2"] if len(phi) == len(self.grid["x2"]) else self.axes[1]
                x3 = self.grid["x3"] if len(theta) == len(self.grid["x3"]) else self.axes[2]
                theta0 = theta + x3 / Re
                self.theta0 = float(np.mean(theta0))
                phi0 = phi - x2 / (Re * np.sin(self.theta0))
                self.phi0 = float(np.mean(phi0))
                if not (np.allclose(theta0, self.theta0, atol=2e-6, rtol=0)
                        and np.allclose(phi0, self.phi0, atol=2e-6, rtol=0)):
                    raise ValueError("Unsupported Cartesian grid geometry: theta/phi are not fixed-origin UEN")

    def coordinates(self, altitude, longitude, latitude):
        if self.dipole:
            q, p, phi = geog2dipole(altitude, longitude, latitude)
            center = np.mean(self.axes[2])
            phi = center + (phi - center + np.pi) % (2*np.pi) - np.pi
            return np.column_stack((q, p, phi))
        phi, theta = geog2geomag(longitude, latitude)
        dphi = (phi - self.phi0 + np.pi) % (2*np.pi) - np.pi
        return np.column_stack((altitude, Re*np.sin(self.theta0)*dphi, Re*(self.theta0-theta)))

    def density(self, frame):
        # Equivalent to pygemini read.frame(..., var={'ne'}), but select the
        # electron species before loading instead of loading all seven species.
        with h5py.File(frame, "r") as f:
            if "nsall" in f:
                ne = f["nsall"][LSP-1, ...].transpose(2, 1, 0)
            elif "neall" in f:
                ne = f["neall"][:].transpose(2, 1, 0)
            elif "ne" in f:
                ne = f["ne"][:].transpose(2, 1, 0)
            else:
                raise ValueError(f"No electron density in {frame}")
        if ne.shape != tuple(map(len, self.axes)):
            raise ValueError(f"Density/grid shape mismatch: {ne.shape}")
        if not np.isfinite(ne).all() or np.any(ne < 0):
            raise ValueError("Electron density must be finite and nonnegative")
        return RegularGridInterpolator(self.axes, ne, bounds_error=False, fill_value=np.nan)
