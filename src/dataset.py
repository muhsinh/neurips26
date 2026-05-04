"""WESAD windowed dataset.

Each sample is a dict {ACC, BVP, EDA, TEMP, label}. ACC has 3 channels, others 1.
PyTorch tensor shape per modality: (B, T, C). Models accept dict-of-tensors.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ["ACC", "BVP", "EDA", "TEMP"]
MOD_CHANNELS = {"ACC": 3, "BVP": 1, "EDA": 1, "TEMP": 1}


class SubjectWindows:
    """Lazy holder for one subject's preprocessed windows. Loads .npz on demand."""
    def __init__(self, npz_path: Path):
        self.path = Path(npz_path)
        self.subject = self.path.stem
        self._cache = None

    def _load(self):
        if self._cache is None:
            data = np.load(self.path)
            self._cache = {
                "labels": data["labels"].astype(np.int64),
                **{m: data[f"X_{m}"].astype(np.float32) for m in MODALITIES},
            }
        return self._cache

    @property
    def labels(self) -> np.ndarray:
        return self._load()["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, mod: str) -> np.ndarray:
        return self._load()[mod]


class WindowDataset(Dataset):
    """Concat dataset over multiple subjects' windows."""
    def __init__(self, subjects: list[SubjectWindows]):
        self.subjects = subjects
        sizes = [len(s) for s in subjects]
        self.cum = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.cum[-1])

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict:
        subj_idx = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = idx - int(self.cum[subj_idx])
        s = self.subjects[subj_idx]
        d = s._load()
        item = {m: torch.from_numpy(d[m][local]) for m in MODALITIES}
        item["label"] = int(d["labels"][local])
        item["subject"] = s.subject
        return item


def collate(batch: list[dict]) -> dict:
    out = {}
    for m in MODALITIES:
        out[m] = torch.stack([b[m] for b in batch], dim=0)
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out["subject"] = [b["subject"] for b in batch]
    return out


def load_all_subjects(processed_dir: Path | str) -> dict[str, SubjectWindows]:
    """Return dict {subject_id: SubjectWindows}."""
    processed_dir = Path(processed_dir)
    out = {}
    for npz in sorted(processed_dir.glob("S*.npz")):
        out[npz.stem] = SubjectWindows(npz)
    return out


def loso_split(all_subjects: dict, test_subject: str) -> tuple[list, list]:
    train = [s for sid, s in all_subjects.items() if sid != test_subject]
    test = [all_subjects[test_subject]]
    return train, test
