import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscillators import (
    damped_harmonic_analytic,
    damping_regime,
    quality_factor,
    resonance_curve,
    resonance_peak_frequency,
    simulate_forced_oscillator,
    simulate_free_oscillator,
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


def test_free_numeric_matches_analytic():
    """Résultat du rapport §7 : erreur L∞ libre sous-amortie < 1e-5 (RK45)."""
    omega0, gamma = 2.0 * np.pi, 0.3
    t_span = (0.0, 8.0)
    t, x_num, _ = simulate_free_oscillator(t_span, (1.0, 0.0), omega0, gamma, n_eval=2000)
    x_ana = damped_harmonic_analytic(t, 1.0, 0.0, omega0, gamma)
    err = float(np.max(np.abs(x_num - x_ana)))
    assert err < 1e-5


def test_regimes_and_quality_factor():
    assert damping_regime(2.0, 0.5) == "sous"
    assert damping_regime(2.0, 2.0) == "critique"
    assert damping_regime(2.0, 3.0) == "sur"
    assert abs(quality_factor(10.0, 0.5) - 10.0) < 1e-12
    # pic de résonance : √(ω0² − 2γ²)
    w_r = resonance_peak_frequency(5.0, 0.2)
    assert abs(w_r - np.sqrt(25.0 - 2 * 0.04)) < 1e-12
