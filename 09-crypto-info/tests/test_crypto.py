import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_info import (
    bsc_capacity,
    generate_rsa_keypair,
    hamming74_decode,
    hamming74_encode,
    rsa_decrypt,
    rsa_encrypt,
    shannon_entropy,
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
