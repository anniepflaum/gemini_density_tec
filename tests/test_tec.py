import unittest
import numpy as np
from gemini_density_tec.tec import xyz, integrate

class IntegrationTests(unittest.TestCase):
    def test_constant_density(self):
        a, b = xyz(65., -147., 0.), xyz(65., -147., 200000.)
        r = integrate(a, b, lambda alt, lon, lat: np.full_like(alt, 1e11))
        self.assertAlmostEqual(r['stec_tecu'], 2.)
        self.assertEqual(r['status'], 'complete')

    def test_linear_density(self):
        a, b = xyz(0., 0., 0.), xyz(0., 0., 200000.)
        r = integrate(a, b, lambda alt, lon, lat: alt*1e6)
        self.assertAlmostEqual(r['stec_tecu'], 2.)

    def test_partial_does_not_bridge_gap(self):
        a, b = xyz(0., 0., 0.), xyz(0., 0., 200000.)
        r = integrate(a, b, lambda alt, lon, lat: np.where(alt >= 80000, 1e11, np.nan), 1000)
        self.assertTrue(np.isnan(r['stec_tecu']))
        self.assertAlmostEqual(r['modeled_stec_tecu'], 1.2)
        self.assertAlmostEqual(r['coverage_fraction'], .6)

    def test_earth_blocked(self):
        r = integrate(xyz(0., 0., 0.), xyz(0., 180., 200000.), lambda a, o, l: a*0)
        self.assertEqual(r['status'], 'earth_blocked')
