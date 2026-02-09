import numpy as np
import pandas as pd
from harmonypy import run_harmony

def test_harmony_deterministic_exact():
    np.random.seed(0)

    X = np.random.randn(50, 200).astype(np.float32)
    meta = pd.DataFrame({
        "batch": np.repeat(["a", "b"], 100)
    })

    ho1 = run_harmony(
        X, meta, vars_use="batch",
        deterministic=True,
        device="cpu"
    )

    ho2 = run_harmony(
        X, meta, vars_use="batch",
        deterministic=True,
        device="cpu"
    )

    assert np.array_equal(ho1.Z_corr, ho2.Z_corr)
