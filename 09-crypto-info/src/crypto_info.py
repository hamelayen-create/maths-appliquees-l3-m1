"""RSA pédagogique, entropie de Shannon, Hamming (7,4).

Avertissement : l'implémentation RSA utilise de petites clés et sert
uniquement à l'enseignement. Elle n'offre aucune sécurité réelle.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Algorithme d'Euclide étendu : retourne (g, x, y) avec ax + by = g = gcd(a,b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def modinv(a: int, m: int) -> int:
    """Inverse de a modulo m (suppose gcd(a, m) = 1)."""
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("inverse modulaire inexistant")
    return x % m


def euler_phi(n: int) -> int:
    """Indicatrice d'Euler φ(n) = |{k : 1 ≤ k ≤ n, gcd(k,n)=1}|."""
    if n <= 0:
        raise ValueError("n must be positive")
    result = n
    p = 2
    x = n
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if x > 1:
        result -= result // x
    return result


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
    """Génère une petite clé RSA (pédagogique uniquement, pas de sécurité réelle)."""
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
    """Entropie de Shannon H(X) = -Σ p_i log2(p_i) en bits."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def binary_entropy(p: float) -> float:
    """Entropie binaire h(p) = H(Bernoulli(p))."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return shannon_entropy(np.array([1.0 - p, p]))


def bsc_capacity(p_error: float) -> float:
    """Capacité du canal binaire symétrique C = 1 - h(p) (bits par usage)."""
    p = float(p_error)
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    if abs(p - 0.5) < 1e-15:
        return 0.0
    return 1.0 - binary_entropy(p)


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


def bsc_flip(bits: np.ndarray, p_error: float, rng: np.random.Generator) -> np.ndarray:
    """Applique un canal BSC : chaque bit est inversé avec proba p_error."""
    x = np.asarray(bits, dtype=int).copy()
    flips = rng.random(x.shape) < p_error
    x[flips] ^= 1
    return x


def simulate_hamming_ber(
    p_error: float,
    n_blocks: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Simule le taux d'erreur bit avant/après correction Hamming(7,4) sur un BSC.

    Retourne (ber_raw, ber_corrected) où :
    - ber_raw : fraction de bits de données erronés si on lit les positions data
      sans correction ;
    - ber_corrected : fraction après décodage à syndrome.
    """
    rng = np.random.default_rng(seed)
    raw_errors = 0
    corr_errors = 0
    total_bits = 0
    for _ in range(n_blocks):
        data = rng.integers(0, 2, size=4)
        code = hamming74_encode(data)
        received = bsc_flip(code, p_error, rng)
        # lecture naïve des positions data (indices 2,4,5,6)
        raw_data = received[np.array([2, 4, 5, 6])]
        recovered, _ = hamming74_decode(received)
        raw_errors += int(np.sum(raw_data != data))
        corr_errors += int(np.sum(recovered != data))
        total_bits += 4
    return raw_errors / total_bits, corr_errors / total_bits
