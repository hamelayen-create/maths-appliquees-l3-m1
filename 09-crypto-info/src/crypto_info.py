"""RSA pédagogique, entropie de Shannon, Hamming (7,4)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("inverse modulaire inexistant")
    return x % m


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_prime(bits: int, rng: random.Random) -> int:
    while True:
        candidate = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
        if is_prime(candidate):
            return candidate


@dataclass
class RSAKeyPair:
    n: int
    e: int
    d: int


def generate_rsa_keypair(bits: int = 16, seed: int = 0) -> RSAKeyPair:
    """Génère une petite clé RSA (pédagogique)."""
    rng = random.Random(seed)
    half = max(bits // 2, 8)
    p = generate_prime(half, rng)
    q = generate_prime(half, rng)
    while q == p:
        q = generate_prime(half, rng)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537 if math.gcd(65537, phi) == 1 else 3
    while math.gcd(e, phi) != 1:
        e += 2
    d = modinv(e, phi)
    return RSAKeyPair(n=n, e=e, d=d)


def rsa_encrypt(message: int, pub_n: int, pub_e: int) -> int:
    if not (0 <= message < pub_n):
        raise ValueError("message must be in [0, n)")
    return pow(message, pub_e, pub_n)


def rsa_decrypt(ciphertext: int, priv_n: int, priv_d: int) -> int:
    return pow(ciphertext, priv_d, priv_n)


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def bsc_capacity(p_error: float) -> float:
    """Capacité du canal binaire symétrique (bits)."""
    p = float(p_error)
    if p <= 0.0 or p >= 1.0:
        return 0.0 if p >= 1.0 else 1.0
    h = shannon_entropy(np.array([1 - p, p]))
    return 1.0 - h


# --- Hamming (7,4) ---
# Positions de parité 1,2,4 ; données 3,5,6,7 (index 1-based)

G = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=int,
)

H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=int,
)


def hamming74_encode(data4: np.ndarray) -> np.ndarray:
    d = np.asarray(data4, dtype=int).reshape(4)
    return (G @ d) % 2


def hamming74_decode(code7: np.ndarray) -> tuple[np.ndarray, int]:
    """Retourne (data4, position_erreur 0 si aucune, 1..7 sinon)."""
    c = np.asarray(code7, dtype=int).reshape(7)
    syndrome = H @ c % 2
    # syndrome bits = b0,b1,b2 -> position = b0 + 2 b1 + 4 b2
    pos = int(syndrome[0] + 2 * syndrome[1] + 4 * syndrome[2])
    corrected = c.copy()
    if pos != 0:
        corrected[pos - 1] ^= 1
    data = corrected[np.array([2, 4, 5, 6])]
    return data, pos
