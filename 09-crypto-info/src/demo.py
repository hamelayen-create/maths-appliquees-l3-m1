#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crypto_info import (
    bsc_capacity,
    generate_rsa_keypair,
    hamming74_decode,
    hamming74_encode,
    rsa_decrypt,
    rsa_encrypt,
    shannon_entropy,
)


def main() -> None:
    print("=== Crypto / théorie de l'information ===")
    keys = generate_rsa_keypair(bits=16, seed=42)
    msg = 12345 % keys.n
    c = rsa_encrypt(msg, keys.n, keys.e)
    m2 = rsa_decrypt(c, keys.n, keys.d)
    print(f"RSA: n={keys.n}, msg={msg}, cipher={c}, decrypt={m2}")

    print(f"Entropie Bernoulli(0.1) = {shannon_entropy(np.array([0.9, 0.1])):.4f} bits")
    print(f"Capacité BSC(p=0.1) = {bsc_capacity(0.1):.4f} bits")

    data = np.array([1, 0, 1, 1])
    code = hamming74_encode(data)
    broken = code.copy()
    broken[3] ^= 1  # flip bit position 4 (1-based)
    recovered, pos = hamming74_decode(broken)
    print(f"Hamming: data={data.tolist()}, err_pos={pos}, recovered={recovered.tolist()}")


if __name__ == "__main__":
    main()
