import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscillators import (
    damped_harmonic_analytic,
    resonance_curve,
    simulate_forced_oscillator,
    steady_state_amplitude,
)


def test_analytic_initial_conditions():
    t = np.array([0.0])
    x = damped_harmonic_analytic(t, 1.0, 0.0, omega0=3.0, gamma=0.2)
    assert abs(x[0] - 1.0) < 1e-12


def test_forced_amplitude_near_theory():
    omega0, gamma, F0, wd = 4.0, 0.25, 1.0, 4.0
    t, x = simulate_forced_oscillator((0, 50), (0, 0), omega0, gamma, F0, wd, n_eval=5000)
    mask = t > 40
    amp_num = 0.5 * (np.max(x[mask]) - np.min(x[mask]))
    amp_th = steady_state_amplitude(omega0, gamma, F0, wd)
    assert abs(amp_num - amp_th) / amp_th < 0.08


def test_resonance_peak_near_natural_freq():
    omega0, gamma = 5.0, 0.2
    omegas, amps = resonance_curve(omega0, gamma, 1.0)
    w_peak = omegas[np.argmax(amps)]
    # pour faible amortissement, pic proche de omega0
    assert abs(w_peak - omega0) / omega0 < 0.05
