import csv
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from gemini3d.grid.convert import Re, geog2geomag
from gemini_density_tec.model import Model
from gemini_density_tec.cli import run

class ModelTests(unittest.TestCase):
    def test_cartesian_origin_density_order_and_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'inputs').mkdir()
            axes = (np.arange(-2, 5)*100000., np.arange(-3, 4)*10000., np.arange(-3, 4)*10000.)
            phi0, theta0 = geog2geomag(np.array([-147.5]), np.array([66.5]))
            shape = (7, 7, 7)
            with h5py.File(root/'inputs/simsize.h5', 'w') as f:
                f['lx'] = [3, 3, 3]
            with h5py.File(root/'inputs/simgrid.h5', 'w') as f:
                for name, axis in zip(('x1','x2','x3'), axes):
                    f[name] = axis
                f['h1'] = np.ones(shape)
                f['theta'] = np.broadcast_to((theta0-axes[2]/Re)[:, None, None], shape)
                f['phi'] = np.broadcast_to((phi0+axes[1]/(Re*np.sin(theta0)))[None, :, None], shape)
            frame = root/'20260210_35340.000000.h5'
            with h5py.File(frame, 'w') as f:
                ns = np.zeros((7,3,3,3))
                ns[-1] = np.broadcast_to(np.array([1e11,2e11,3e11]), (3,3,3))
                f['nsall'] = ns
            model = Model(root)
            coordinates = model.coordinates(np.array([100000.]), np.array([-147.5]), np.array([66.5]))
            np.testing.assert_allclose(coordinates, [[100000.,0.,0.]], atol=1e-7)
            self.assertAlmostEqual(model.density(frame)(coordinates)[0], 2e11)
            (root/'receivers.csv').write_text('receiver_id,lat_deg,lon_deg,alt_m\nR,66.5,-147.5,1000\n')
            (root/'trajectories.csv').write_text('time_utc,rocket_id,lat_deg,lon_deg,alt_m\n2026-02-10T09:49:00Z,A,66.5,-147.5,199000\n2026-02-10T10:19:00Z,B,66.5,-147.5,199000\n')
            (root/'times.csv').write_text('filename,model_time_utc\n20260210_35340.000000.h5,2026-02-10T10:24:00Z\n')
            trajectory = root/'trajectories.csv'
            trajectory.write_text(trajectory.read_text().replace('09:49:00', '10:24:00'))
            run(Namespace(frame_times=root/'times.csv', model=root, receivers=root/'receivers.csv', trajectories=root/'trajectories.csv', output=root/'out.csv', start=None, end=None, trajectory_offset_seconds=0, step_m=500, max_gap_seconds=10))
            with (root/'out.csv').open() as f:
                results = list(csv.DictReader(f))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['model_time_utc'], '2026-02-10T10:24:00+00:00')
            self.assertEqual(results[0]['source_filename'], frame.name)
            self.assertAlmostEqual(float(results[0]['stec_tecu']), 3.96)
            self.assertEqual(results[1]['status'], 'missing_trajectory')
