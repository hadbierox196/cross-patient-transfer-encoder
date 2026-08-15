"""pulse2percept simulator wrapper.

Builds a patient-specific implant/model pair and renders percepts, resized
to a fixed output resolution so they can be compared against target images
of arbitrary simulator grid resolution.
"""

import numpy as np
from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel
from skimage.transform import resize as sk_resize

from . import config


def build_model_for_patient(patient: dict):
    """Construct an ArgusII implant + AxonMapModel for one virtual patient."""
    implant = ArgusII(rot=patient["implant_rot"])
    model = AxonMapModel(
        rho=patient["rho"],
        axlambda=patient["axlambda"],
        loc_od=(patient["loc_od_x"], patient["loc_od_y"]),
        xrange=(-15, 15),
        yrange=(-15, 15),
        xystep=0.5,
    )
    model.build()
    return implant, model


def render_percept(implant, model, electrode_amplitudes: dict,
                    out_size: int = config.IMG_SIZE) -> np.ndarray:
    """Render a percept for given electrode amplitudes, resized to out_size x out_size.

    NOTE: the simulator's native output grid resolution depends on the model's
    xrange/yrange/xystep and will generally NOT match out_size directly — this
    function resizes to guarantee shape compatibility with target images
    (required for SSIM comparison). This fixes a shape-mismatch bug present in
    early iterations of this pipeline.
    """
    implant.stim = electrode_amplitudes
    percept = model.predict_percept(implant)
    frame = np.asarray(percept.data[..., 0])
    return sk_resize(frame, (out_size, out_size), anti_aliasing=True, preserve_range=True)


def compute_electrode_templates(implant, model, out_size: int = config.IMG_SIZE) -> np.ndarray:
    """Render each electrode alone at unit amplitude to build a linear template matrix.

    Returns an (n_pixels, n_electrodes) matrix mapping electrode amplitude
    vectors to pixel space via linear combination. Computed once per patient
    and reused for all NNLS target computations for that patient.
    """
    templates = []
    for name in implant.electrode_names:
        stim_dict = {n: (1.0 if n == name else 0.0) for n in implant.electrode_names}
        frame = render_percept(implant, model, stim_dict, out_size=out_size)
        templates.append(frame.flatten())
    return np.stack(templates, axis=1)
