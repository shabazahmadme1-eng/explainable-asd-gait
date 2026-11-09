"""
Test-time domain alignment for the MS-G3D embedding stream.

Problem (measured): the embedding net only ``center_skeleton``s its input, so it
is sensitive to the *raw* per-joint coordinate distribution. MediaPipe skeletons
sit ~4.3 sigma outside the Kinect training distribution (per-joint diagnostics:
lower body over-extended, depth axis shifted), which collapses the embedding to
a near-constant ~0.92 regardless of class.

Fix: map each input channel into the Kinect distribution the net expects,

    x' = (x - mp_mu) / mp_sd * (k_sd * shrink) + k_mu

where (k_mu, k_sd) is the Kinect reference *baked into the model's data_bn
buffers* (no external data needed for the target) and (mp_mu, mp_sd) is a
MediaPipe reference estimated from a corpus of keypoint clips
(see scripts/build_mp_reference.py). This is equivalent to swapping data_bn's
running statistics for MediaPipe statistics at test time (AdaBN).

Joint and bone streams are aligned independently, each against its own data_bn
reference. Only the embedding path is aligned — the handcrafted-feature path
keeps the raw skeleton (its features are scale/orientation-invariant and barely
shift across domains).

`shrink` (default 1.0) controls how much MediaPipe variance is mapped onto the
Kinect variance: 1.0 = full match (best class separation in POC), <1.0 pulls
inputs toward the Kinect mean (safer / lower OOD, slightly less signal).
"""
from __future__ import annotations

import os

import numpy as np


def kinect_reference_from_model(model) -> dict:
    """Read the Kinect per-channel reference (25x3 mean/std) from data_bn buffers."""
    def ref(bn):
        mu = bn.running_mean.detach().cpu().numpy().astype(np.float64).reshape(25, 3)
        var = bn.running_var.detach().cpu().numpy().astype(np.float64).reshape(25, 3)
        return mu, np.sqrt(var + 1e-5)

    kj_mu, kj_sd = ref(model.stream1.data_bn)
    out = {"joint_mu": kj_mu, "joint_sd": kj_sd}
    if getattr(model, "use_bone_stream", False) and hasattr(model, "stream2"):
        kb_mu, kb_sd = ref(model.stream2.data_bn)
        out["bone_mu"], out["bone_sd"] = kb_mu, kb_sd
    return out


def load_mp_reference(path: str | None) -> dict | None:
    """Load a MediaPipe skeleton reference (.npz) or return None if absent."""
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    ref = {"joint_mu": d["joint_mu"].astype(np.float64),
           "joint_sd": d["joint_sd"].astype(np.float64)}
    if "bone_mu" in d:
        ref["bone_mu"] = d["bone_mu"].astype(np.float64)
        ref["bone_sd"] = d["bone_sd"].astype(np.float64)
    ref["n_frames"] = int(d["n_frames"]) if "n_frames" in d else None
    ref["n_clips"] = int(d["n_clips"]) if "n_clips" in d else None
    return ref


class SkeletonAligner:
    """Per-channel affine alignment of a centered NTU skeleton (and its bones)."""

    def __init__(self, kinect_ref: dict, mp_ref: dict | None, shrink: float = 1.0):
        self.k = kinect_ref
        self.mp = mp_ref
        self.shrink = float(shrink)

    @property
    def ready(self) -> bool:
        return self.mp is not None and self.k is not None

    def _affine(self, arr_tvc: np.ndarray, mmu, msd, kmu, ksd) -> np.ndarray:
        """arr_tvc: (T, 25, 3) -> aligned (T, 25, 3)."""
        out = arr_tvc.astype(np.float64).copy()
        for c in range(3):
            out[:, :, c] = ((out[:, :, c] - mmu[:, c]) / (msd[:, c] + 1e-9)
                            * (ksd[:, c] * self.shrink) + kmu[:, c])
        return out

    def align_joint(self, ntu: np.ndarray) -> np.ndarray:
        """ntu (3, T, 25, 1) -> aligned (3, T, 25, 1)."""
        if not self.ready:
            return ntu
        x = ntu[:, :, :, 0].transpose(1, 2, 0)               # (T,25,3)
        x = self._affine(x, self.mp["joint_mu"], self.mp["joint_sd"],
                         self.k["joint_mu"], self.k["joint_sd"])
        return x.transpose(2, 0, 1)[:, :, :, None].astype(np.float32)

    def align_bone(self, bone: np.ndarray) -> np.ndarray:
        """bone (3, T, 25, 1) computed from the RAW joints -> aligned (3, T, 25, 1)."""
        if not self.ready or "bone_mu" not in self.k or "bone_mu" not in self.mp:
            return bone
        x = bone[:, :, :, 0].transpose(1, 2, 0)
        x = self._affine(x, self.mp["bone_mu"], self.mp["bone_sd"],
                         self.k["bone_mu"], self.k["bone_sd"])
        return x.transpose(2, 0, 1)[:, :, :, None].astype(np.float32)
