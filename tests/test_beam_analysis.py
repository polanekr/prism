"""Native footprint and exposure identities with analytical reference cases."""
import math
import unittest
import numpy as np
from beam_analysis import Experiment, temporal_metrics, analyze_measured_footprint
from bio_analysis import SampleRegion


class MeasuredBeamTests(unittest.TestCase):
    def analyze(self,data,experiment=None,ellipse=False,spacing=.1):
        return analyze_measured_footprint(data,np.full(data.shape,.2),spacing,
            SampleRegion(0,data.shape,((1,0),(0,1)),(0,0),None,None),
            (0,data.shape[1],0,data.shape[0]),experiment or Experiment(),ellipse=ellipse)

    def test_exposure_example_and_gaussian_peak(self):
        e=Experiment(shots=1000,duration_s=2.,frequency_hz=500.,particle_pulse_fs=100.,
            pulse_plane='film',pulse_shape='rectangular',pulse_structure='one_per_event')
        t=temporal_metrics(11.,e)
        self.assertAlmostEqual(t['dose_per_shot_gy'],.011)
        self.assertEqual(t['average_rate_gy_s'],5.5)
        self.assertAlmostEqual(t['conditional_peak_rate_gy_s']/1e11,1.1)
        g=Experiment(**{**e.to_dict(),'pulse_shape':'gaussian_fwhm'})
        self.assertAlmostEqual(temporal_metrics(11.,g)['conditional_peak_rate_gy_s']/1.1e11,math.sqrt(4*math.log(2)/math.pi))

    def test_no_implicit_counts_or_laser_duration_substitution(self):
        t=temporal_metrics(11.,Experiment(duration_s=2.,frequency_hz=500.,laser_pulse_fs=30.))
        self.assertIsNone(t['dose_per_shot_gy']);self.assertEqual(t['average_rate_gy_s'],5.5)
        self.assertIsNone(t['conditional_peak_rate_gy_s'])
        t=temporal_metrics(11.,Experiment(shots=1000,particle_pulse_fs=100,pulse_plane='source',pulse_shape='rectangular',pulse_structure='one_per_event'))
        self.assertIsNone(t['conditional_peak_rate_gy_s'])
        self.assertIsNone(temporal_metrics(11.,Experiment(shots=1000,particle_pulse_fs=100,pulse_plane='film',pulse_shape='rectangular'))['conditional_peak_rate_gy_s'])

    def test_window_validation_and_no_single_shot_zero_duration(self):
        e=Experiment(shots=1,start_s=10.,end_s=12.)
        self.assertEqual(temporal_metrics(11.,e)['average_rate_gy_s'],5.5)
        for kwargs in ({'shots':0},{'shots':1.2},{'duration_s':0},{'start_s':1},
            {'start_s':2,'end_s':1},{'duration_s':3,'start_s':1,'end_s':3},{'particle_pulse_fs':float('nan')}):
            with self.assertRaises(ValueError):Experiment(**kwargs)

    def test_uniform_ellipse_has_no_exterior_zero_bias(self):
        m=self.analyze(np.full((100,100),10.),Experiment(shots=100),ellipse=True)['metrics']
        self.assertEqual(m['mean_dose_gy'],10.)
        self.assertEqual(m['spatial_std_gy'],0.)
        self.assertLess(m['roi_pixels'],10000)
        np.testing.assert_allclose(m['centroid_mm'],[4.95,4.95])
        self.assertIsNone(m['orientation_deg_y_down'])
        self.assertAlmostEqual(m['temporal']['dose_only_uncertainty_bounds']['mean_dose_per_event_gy'],.002)

    def test_gaussian_principal_widths_centroid_and_enclosed_radii(self):
        y,x=np.mgrid[:241,:241];sx,sy=20.,10.
        d=np.exp(-.5*((x-120)**2/sx**2+(y-120)**2/sy**2))
        m=self.analyze(d)['metrics']
        np.testing.assert_allclose(m['centroid_mm'],[12,12],atol=1e-6)
        self.assertAlmostEqual(m['sigma_major_mm'],2.,places=5)
        self.assertAlmostEqual(m['sigma_minor_mm'],1.,places=5)
        self.assertAlmostEqual(m['ellipticity_major_minor'],2.,places=5)
        self.assertAlmostEqual(m['orientation_deg_y_down'],0.,places=5)
        iso=np.exp(-.5*((x-120)**2+(y-120)**2)/20**2)
        m=self.analyze(iso)['metrics']
        for p in (50,80,90):self.assertAlmostEqual(m['radii_mm'][str(p)],2*np.sqrt(-2*np.log(1-p/100)),delta=.02)

    def test_background_zero_signal_and_pixel_scale(self):
        d=np.ones((20,20))
        m=self.analyze(d,Experiment(background_gy=1.))['metrics']
        self.assertEqual(m['mean_dose_gy'],1.)
        self.assertIsNone(m['centroid_mm']);self.assertIsNone(m['radii_mm'])
        self.assertEqual(m['signal_integral_gy_mm2'],0.)
        z=self.analyze(np.zeros((20,20)))['metrics']
        self.assertEqual(z['mean_dose_gy'],0.);self.assertIsNone(z['sigma_major_mm'])
        a=self.analyze(d,spacing=.1)['metrics'];b=self.analyze(d,spacing=.2)['metrics']
        self.assertAlmostEqual(b['sigma_major_mm'],2*a['sigma_major_mm'])
        self.assertAlmostEqual(b['roi_area_mm2'],4*a['roi_area_mm2'])

    def test_declared_core_radius_defines_halo_without_a_tautological_percentile(self):
        data=np.ones((20,20))
        m=self.analyze(data,Experiment(core_radius_mm=.5))['metrics']
        yy,xx=np.mgrid[:20,:20]
        expected=np.mean(np.hypot((xx-9.5)*.1,(yy-9.5)*.1)>.5)
        self.assertAlmostEqual(m['halo_signal_fraction'],expected)
        self.assertAlmostEqual(m['halo_core_ratio'],expected/(1-expected))
        self.assertIsNone(self.analyze(data)['metrics']['halo_signal_fraction'])

    def test_clipping_warning_and_no_outside_pixels(self):
        d=np.ones((30,30));region=SampleRegion(0,(20,20),((1,0),(0,1)),(-5,-5),None,None)
        r=analyze_measured_footprint(d,None,.1,region,(0,30,0,30),Experiment())
        self.assertEqual(r['metrics']['mean_dose_gy'],1.)
        self.assertTrue(any('clipped' in warning for warning in r['metrics']['warnings']))
