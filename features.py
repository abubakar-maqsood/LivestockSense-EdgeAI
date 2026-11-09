"""Feature extraction MATCHING Phase 1 notebook (17 numeric features)."""
import numpy as np
import pandas as pd


def extract_features(a):
    a = np.asarray(a, dtype=np.float32)
    if a.shape != (50, 3) or not np.isfinite(a).all():
        raise ValueError("Input must contain exactly 50 valid rows of X, Y, Z readings.")
    mag = np.linalg.norm(a, axis=1)
    return np.r_[a.mean(axis=0), a.std(axis=0), a.min(axis=0), a.max(axis=0),
                 (a * a).mean(axis=0), mag.mean(), mag.std()].astype(np.float32)


def read_first_window(path):
    df = pd.read_csv(path)
    cols = {str(col).lower().strip(): col for col in df.columns}
    aliases = [("accx", "x", "acc_x_g"), ("accy", "y", "acc_y_g"), ("accz", "z", "acc_z_g")]
    selected = []
    for names in aliases:
        found = next((cols[k] for k in names if k in cols), None)
        if found is None:
            raise ValueError("CSV must contain AccX, AccY, AccZ (or x, y, z) columns.")
        selected.append(found)
    if len(df) < 50:
        raise ValueError(f"At least 50 consecutive samples needed; file has {len(df)}.")
    data = df[selected].iloc[:50].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("The first 50 sensor rows have missing or non-numeric values.")
    # Only validate time continuity when timestamp is supplied.
    for name in ("timestamp_unix", "timestamp_ms", "timestamp_s"):
        if name in cols:
            t = pd.to_numeric(df[cols[name]].iloc[:50], errors="coerce").to_numpy(dtype=float)
            dt = np.diff(t)
            if name == "timestamp_ms": dt /= 1000
            if not np.isfinite(dt).all() or np.any(dt <= 0) or np.any(dt > 0.12):
                raise ValueError("Non-continuous timestamps in first 50 rows. Select a continuous recording.")
            break
    return data
