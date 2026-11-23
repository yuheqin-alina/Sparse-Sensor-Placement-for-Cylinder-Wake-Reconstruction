"""Reproduce the ME 570 cylinder-wake midpoint baseline with NumPy and SciPy.

Run `python me570_cylinder_baseline.py` to download the public data and evaluate
on validation frames. Use `--data PATH` to point to an existing .mat file.
"""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat
from scipy.linalg import qr


DATA_URL = (
    "https://raw.githubusercontent.com/maziarraissi/PINNs/master/"
    "main/Data/cylinder_nektar_wake.mat"
)
RANK = 8
RANDOM_MASKS = 40
SEED = 570


def read_cross_stream_velocity(path):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as scratch:
            download = Path(scratch) / path.name
            with urlopen(DATA_URL, timeout=60) as response, download.open("wb") as output:
                shutil.copyfileobj(response, output)
            download.replace(path)

    velocity = loadmat(path, variable_names=("U_star",))["U_star"]
    if velocity.shape != (5000, 2, 200):
        raise ValueError(f"Expected U_star with shape (5000, 2, 200); got {velocity.shape}.")
    return velocity[:, 1, :]


def reconstruction_metrics(basis, centered_fields, sensors):
    measurement_matrix = basis[sensors, :]
    measurements = centered_fields[sensors, :]
    coefficients = np.linalg.lstsq(measurement_matrix, measurements, rcond=None)[0]
    residual = basis @ coefficients - centered_fields
    rmse = np.sqrt(np.mean(residual**2))
    return rmse, np.linalg.cond(measurement_matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cylinder_nektar_wake.mat"))
    args = parser.parse_args()

    fields = read_cross_stream_velocity(args.data)
    train = fields[:, :100]
    validation = fields[:, 100:150]
    # Frames 150:200 are reserved for the project's final test.
    train_mean = train.mean(axis=1, keepdims=True)
    basis = np.linalg.svd(train - train_mean, full_matrices=False)[0][:, :RANK]
    centered_validation = validation - train_mean

    _, _, pivots = qr(basis.T, mode="economic", pivoting=True)
    qr_sensors = pivots[:RANK]
    qr_rmse, qr_condition = reconstruction_metrics(basis, centered_validation, qr_sensors)

    rng = np.random.default_rng(SEED)
    random_results = []
    for _ in range(RANDOM_MASKS):
        sensors = rng.choice(fields.shape[0], size=RANK, replace=False)
        random_results.append(reconstruction_metrics(basis, centered_validation, sensors))
    quartiles = np.percentile(random_results, (25, 50, 75), axis=0)
    full_field_projection = basis @ (basis.T @ centered_validation)
    projection_rmse = np.sqrt(np.mean((full_field_projection - centered_validation)**2))

    print(f"Validation frames 100:150, {RANK} sensors, {RANK} POD modes, noiseless")
    print(f"QR: RMSE={qr_rmse:.6f}, condition number={qr_condition:.2f}")
    print(f"QR sensor indices (zero-based): {qr_sensors.tolist()}")
    for column, label in enumerate(("RMSE", "condition number")):
        lower, median, upper = quartiles[:, column]
        print(
            f"Random ({RANDOM_MASKS} masks, seed {SEED}): {label} median={median:.6g}, "
            f"quartiles=[{lower:.6g}, {upper:.6g}]"
        )
    print(f"Full-field rank-{RANK} projection RMSE={projection_rmse:.6f}")


if __name__ == "__main__":
    main()
