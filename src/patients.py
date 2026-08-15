"""Virtual patient generation.

Parameter ranges are consistent with pulse2percept's default calibration
literature for the AxonMapModel / ArgusII implant.
"""

import numpy as np


def generate_virtual_patients(n_patients: int, seed: int = 0) -> list[dict]:
    """Sample patient-specific implant/model parameters.

    Args:
        n_patients: number of virtual patients to generate.
        seed: random seed for reproducibility.

    Returns:
        List of dicts, one per patient, with keys:
        id, rho, axlambda, loc_od_x, loc_od_y, implant_rot.
    """
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n_patients):
        patients.append({
            "id": i,
            "rho": rng.uniform(100, 400),        # spatial decay constant (um)
            "axlambda": rng.uniform(500, 2000),  # axonal decay constant (um)
            "loc_od_x": rng.uniform(-20, 20),    # optic disc location, x (deg)
            "loc_od_y": rng.uniform(-5, 5),      # optic disc location, y (deg)
            "implant_rot": rng.uniform(-30, 30), # implant rotation (deg)
        })
    return patients


def split_patients(patients: list[dict], n_pretrain: int, n_holdout: int):
    """Split patients into pretraining pool and held-out evaluation set."""
    pretrain = patients[:n_pretrain]
    holdout = patients[n_pretrain:n_pretrain + n_holdout]
    return pretrain, holdout
