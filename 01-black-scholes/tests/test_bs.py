import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bs_closed_form import call_price, put_price
from bs_mc import price_call_mc
from bs_pde import price_call_crank_nicolson


@pytest.fixture
def params():
    return dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_put_call_parity(params):
    c = call_price(**params)
    p = put_price(**params)
    S, K, T, r = params["S"], params["K"], params["T"], params["r"]
    assert abs(c - p - (S - K * np.exp(-r * T))) < 1e-10


def test_mc_close_to_closed(params):
    closed = call_price(**params)
    mc, se = price_call_mc(
        params["S"], params["K"], params["T"], params["r"], params["sigma"],
        n_paths=150_000, seed=0,
    )
    assert abs(mc - closed) < 4 * se + 0.05


def test_pde_close_to_closed(params):
    closed = call_price(**params)
    pde = price_call_crank_nicolson(
        params["S"], params["K"], params["T"], params["r"], params["sigma"],
        n_space=300, n_time=300,
    )
    assert abs(pde - closed) < 0.15
