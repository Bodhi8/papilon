import numpy as np

from papilon.mmm import delayed_adstock, geometric_adstock, hill


def test_geometric_kernel_preserves_steady_state():
    x = np.full(40, 10.0)
    out = geometric_adstock(x, alpha=0.6, l_max=8)
    assert np.allclose(out[8:], 10.0)


def test_delayed_effect_peaks_at_peak_lag():
    pulse = np.zeros(20)
    pulse[0] = 1.0
    out = delayed_adstock(pulse, alpha=0.6, peak=2.0, l_max=8)
    assert int(np.argmax(out)) == 2
    assert np.isclose(out.sum(), 1.0)


def test_hill_is_half_at_half_saturation():
    assert np.isclose(hill(np.array([35.0]), 35.0)[0], 0.5)
    assert hill(np.array([0.0]), 35.0)[0] == 0.0
