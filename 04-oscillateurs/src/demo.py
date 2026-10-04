#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oscillators import (
    damped_harmonic_analytic,
    fft_spectrum,
    resonance_curve,
    simulate_duffing,
    simulate_forced_oscillator,
    steady_state_amplitude,
)


def main() -> None:
    omega0, gamma = 2.0 * np.pi, 0.3
    t = np.linspace(0, 8, 1000)
    x_ana = damped_harmonic_analytic(t, x0=1.0, v0=0.0, omega0=omega0, gamma=gamma)
    print("=== Oscillateurs ===")
    print(f"Libre amorti : |x|(t=0)={x_ana[0]:.4f}, |x|(fin)={abs(x_ana[-1]):.4e}")

    omega_drive = omega0
    t_f, x_f = simulate_forced_oscillator(
        (0, 40), (0.0, 0.0), omega0, gamma, F0=1.0, omega_drive=omega_drive
    )
    # régime permanent : dernière période ~ 5
    mask = t_f > 30
    amp_num = 0.5 * (np.max(x_f[mask]) - np.min(x_f[mask]))
    amp_th = steady_state_amplitude(omega0, gamma, 1.0, omega_drive)
    print(f"Résonance : amp num≈{amp_num:.4f}, amp th={amp_th:.4f}")

    omegas, amps = resonance_curve(omega0, gamma, 1.0)
    print(f"Courbe de résonance : pic à ω={omegas[np.argmax(amps)]:.3f} (ω0={omega0:.3f})")

    t_d, x_d, v_d = simulate_duffing((0, 80), (0.1, 0.0))
    freqs, spectrum = fft_spectrum(t_d[t_d > 40], x_d[t_d > 40])
    peak = freqs[np.argmax(spectrum[1:]) + 1]
    print(f"Duffing FFT peak ≈ {peak:.3f} Hz")
    print(f"Phase space range : x∈[{x_d.min():.2f},{x_d.max():.2f}], v∈[{v_d.min():.2f},{v_d.max():.2f}]")


if __name__ == "__main__":
    main()
