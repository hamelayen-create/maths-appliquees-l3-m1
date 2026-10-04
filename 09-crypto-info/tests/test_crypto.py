import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_info import (
    binary_entropy,
    bsc_capacity,
    egcd,
    euler_phi,
    generate_rsa_keypair,
    hamming74_decode,
    hamming74_encode,
    rsa_decrypt,
    rsa_encrypt,
    shannon_entropy,
    simulate_hamming_ber,
)


def test_rsa_roundtrip():
    keys = generate_rsa_keypair(bits=18, seed=1)
    for msg in [0, 1, 42, keys.n - 1]:
        assert rsa_decrypt(rsa_encrypt(msg, keys.n, keys.e), keys.n, keys.d) == msg


def test_entropy_fair_coin():
    assert abs(shannon_entropy(np.array([0.5, 0.5])) - 1.0) < 1e-12


def test_bsc_capacity_bounds():
    assert abs(bsc_capacity(0.0) - 1.0) < 1e-12
    assert bsc_capacity(0.5) < 1e-12


def test_hamming_corrects_single_bit():
    data = np.array([1, 1, 0, 1])
    code = hamming74_encode(data)
    for i in range(7):
        broken = code.copy()
        broken[i] ^= 1
        recovered, pos = hamming74_decode(broken)
        assert pos == i + 1
        assert np.array_equal(recovered, data)


def test_euler_phi_prime_product():
    # Résultat du rapport : p=61, q=53 => φ(n)=(p-1)(q-1)=3120
    assert euler_phi(61 * 53) == 60 * 52
    assert euler_phi(61 * 53) == 3120


def test_egcd_bezout():
    g, x, y = egcd(240, 46)
    assert g == 2
    assert 240 * x + 46 * y == g


def test_binary_entropy_symmetric():
    assert abs(binary_entropy(0.1) - binary_entropy(0.9)) < 1e-12
    assert abs(binary_entropy(0.5) - 1.0) < 1e-12


def test_hamming_ber_improves_on_moderate_noise():
    """Résultat numérique du rapport : pour p=0.05, la BER corrigée < BER brute."""
    ber_raw, ber_corr = simulate_hamming_ber(0.05, n_blocks=2000, seed=7)
    assert ber_raw > 0.0
    assert ber_corr < ber_raw


def test_H_G_zero():
    """Propriété du rapport : H G = 0 sur F_2 (code linéaire bien formé)."""
    from crypto_info import G, H
    assert np.all((H @ G) % 2 == 0)
