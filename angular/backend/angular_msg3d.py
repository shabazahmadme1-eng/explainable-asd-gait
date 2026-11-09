"""
Angular MS-G3D — the deep stream that transfers to phone.
=========================================================
The paper's Stream-2: coordinate-invariant angular features (unit bone vectors)
-> MS-G3D graph network, trained from scratch on the 800-clip Kinect cache with
strict grouped 10-fold CV. Unlike the raw-coordinate MS-G3D embedding (and the
Skepxels CNN), the angular input is scale/translation-invariant, so it survives
the Kinect->MediaPipe/phone gap:

    in-domain (Kinect, grouped CV)  : AUC 0.972 / 91%   (alone)
    14 held-out real phone kids     : discriminative (not saturated)
    deployed  0.5*HC + 0.5*angular  : phone AUC 0.979, 13/14, spec 6/6

Weights: ``msg3d_angular.pt`` (a plain OfficialMSG3D with num_class=2), scored
via the exact preprocessing used in training (backend/config NTU geometry).
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch

from . import config as C
from .model_def import OfficialMSG3D

# SpineShoulder in NTU joint order -> torso vector once the skeleton is centred.
_SPINE_SHOULDER_NTU = C.NTU_JOINT_NAMES.index("SpineShoulder")   # == 24
_T_LEN = 100                                                     # temporal length (training default)


class AngularMSG3D:
    """Loads the angular MS-G3D classifier and scores a raw clip -> P(ASD).

    ``.available`` is False (and .score raises) when the weights are missing, so
    callers can fall back to the HC-only screen on a fresh checkout.
    """

    def __init__(self, path: str = C.MSG3D_ANGULAR_PATH,
                 device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._lock = threading.Lock()          # MS-G3D forward isn't reentrant
        self._reorder = np.asarray(C.NTU_REORDER_COLS, dtype=int)
        self._parents = np.asarray(C.NTU_PARENTS, dtype=int)
        if not os.path.exists(path):
            print(f"[angular_msg3d] weights not found: {path}")
            return
        try:
            m = OfficialMSG3D(
                num_class=2, num_point=C.NUM_JOINTS, num_person=1,
                num_gcn_scales=C.NTU60_NUM_GCN_SCALES,
                num_g3d_scales=C.NTU60_NUM_G3D_SCALES,
                graph="graph.ntu_rgb_d.AdjMatrixGraph")
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except (TypeError, RuntimeError):
                state = torch.load(path, map_location=self.device, weights_only=False)
            missing, unexpected = m.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[angular_msg3d] load_state_dict: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected")
            self.model = m.to(self.device).eval()
        except Exception as exc:  # pragma: no cover - surfaced at startup
            print(f"[angular_msg3d] not available: {exc}")
            self.model = None

    # Platt calibration of the raw angular output, fit on mp_train_99 (99 MediaPipe
    # captures of the training children) to undo the Kinect->MediaPipe inflation.
    # Validated to transfer to the field cohort (TD angular 0.75 -> 0.26; fused AUC
    # 0.968 -> 0.984, same 15/16 decision). Gives an honest movement-model number.
    CAL_A, CAL_B = 0.524, -1.782

    def calibrate(self, p: float) -> float:
        """Map the raw (inflated) angular P(ASD) onto the de-inflated MediaPipe scale."""
        p = float(min(max(p, 1e-4), 1.0 - 1e-4))
        z = np.log(p / (1.0 - p))
        return float(1.0 / (1.0 + np.exp(-(self.CAL_A * z + self.CAL_B))))

    @property
    def available(self) -> bool:
        return self.model is not None

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """raw (T,75) coords -> (1,3,_T_LEN,25,1) unit-bone (angular) tensor.

        Verbatim the training preprocessing: NTU re-order, resample to _T_LEN,
        centre on SpineBase, scale by torso length, unit bone vectors.
        """
        x = np.asarray(clip, dtype=np.float32)
        ntu = x[:, self._reorder].reshape(len(x), 25, 3)
        idx = np.linspace(0, len(ntu) - 1, _T_LEN).astype(int)
        ntu = ntu[idx]
        ntu = ntu - ntu[:, C.CENTER_JOINT:C.CENTER_JOINT + 1, :]         # centre spine base
        torso = np.linalg.norm(ntu[:, _SPINE_SHOULDER_NTU, :], axis=-1).mean() + 1e-6
        ntu = ntu / torso                                               # scale by torso length
        bone = ntu - ntu[:, self._parents, :]                          # unit bone vectors
        bone = bone / np.maximum(np.linalg.norm(bone, axis=-1, keepdims=True), 1e-6)
        t = np.transpose(bone, (2, 0, 1))[:, :, :, None]               # (3,T,25,1)
        return torch.from_numpy(np.ascontiguousarray(t)).float().unsqueeze(0)

    def score(self, clip: np.ndarray) -> float:
        """raw (T,75) clip (best fed a 150-frame resample) -> P(ASD) in [0,1]."""
        if not self.available:
            raise RuntimeError("Angular MS-G3D model not loaded.")
        x = self._to_tensor(clip).to(self.device)
        with self._lock, torch.no_grad():
            out = self.model(x)
            # OfficialMSG3D returns logits (N, num_class)
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.softmax(logits.float(), dim=1)[0, 1].item()
        return float(p)
