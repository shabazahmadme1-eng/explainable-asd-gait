"""
TwoStreamOfficialMSG3D — the network whose fine-tuned weights live in
msg3d_v40.pth. Architecture is identical to training; the only change is that
the ImageNet-style "load NTU pretrained weights" step is skipped, because we
load the *full* fine-tuned state dict right after construction.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from . import config as C

# Make the cloned MS-G3D repo importable (provides model.msg3d.Model).
if C.MSG3D_REPO not in sys.path:
    sys.path.insert(0, C.MSG3D_REPO)

try:
    from model.msg3d import Model as OfficialMSG3D
except Exception as exc:  # pragma: no cover - surfaced clearly at startup
    raise ImportError(
        f"Could not import MS-G3D from {C.MSG3D_REPO!r}. Clone it with:\n"
        f"  git clone https://github.com/kenziyuliu/ms-g3d.git "
        f"{C.MSG3D_REPO}\n(original error: {exc})"
    ) from exc


class TwoStreamOfficialMSG3D(nn.Module):
    def __init__(self, num_classes: int = 2, use_bone_stream: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        self.use_bone_stream = use_bone_stream
        model_args = dict(
            num_class=C.NTU60_NUM_CLASSES, num_point=C.NUM_JOINTS, num_person=1,
            num_gcn_scales=C.NTU60_NUM_GCN_SCALES,
            num_g3d_scales=C.NTU60_NUM_G3D_SCALES,
            graph="graph.ntu_rgb_d.AdjMatrixGraph",
        )
        self.stream1 = OfficialMSG3D(**model_args)
        self.stream1.fc = nn.Identity()
        if use_bone_stream:
            self.stream2 = OfficialMSG3D(**model_args)
            self.stream2.fc = nn.Identity()
        with torch.no_grad():
            actual_dim = self.stream1(
                torch.zeros(1, 3, C.TARGET_FRAMES, 25, 1)).shape[1]
        self.emb_dim = actual_dim * 2 if use_bone_stream else actual_dim
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.emb_dim, num_classes)

    def forward(self, x_j, x_b=None):
        feat1 = self.stream1(x_j)
        if self.use_bone_stream and x_b is not None:
            features = torch.cat([feat1, self.stream2(x_b)], dim=1)
        else:
            features = feat1
        return self.classifier(self.dropout(features)), features

    def get_embeddings(self, x_j, x_b=None):
        with torch.no_grad():
            return self.forward(x_j, x_b)[1]


def load_deploy_model(pth_path: str = C.PTH_PATH,
                      device: torch.device | None = None) -> TwoStreamOfficialMSG3D:
    """Build the two-stream model and load the fine-tuned deployment weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    model = TwoStreamOfficialMSG3D(
        num_classes=2, use_bone_stream=C.USE_BONE_STREAM, dropout=0.5).to(device)
    try:
        state = torch.load(pth_path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        state = torch.load(pth_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Should be empty for the v40 checkpoint; warn loudly if not.
        print(f"[model_def] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    return model
