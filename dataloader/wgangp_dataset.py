from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from dataloader.eeg_dataset import EEGWindowDataset


@dataclass(frozen=True)
class SingleClassStats:
    class_index: int
    class_name: str
    total_windows: int
    selected_windows: int


class SingleClassEEGWindowDataset(Dataset):
    """View of an EEGWindowDataset containing only windows positive for one class.

    Returns only the EEG tensor x (shape: (C, T)).

    Selection logic:
      - include_other_labels=True: window is selected if it contains the class.
      - include_other_labels=False: window is selected only if it contains *only* that class.

    Notes:
      - This dataset intentionally uses the base dataset's prebuilt index (`_index`) for
        fast filtering without EDF I/O.
    """

    def __init__(
        self,
        base: EEGWindowDataset,
        *,
        class_index: int,
        class_name: str | None = None,
        include_other_labels: bool = True,
    ) -> None:
        self.base = base
        self.class_index = int(class_index)
        if self.class_index < 0 or self.class_index >= len(self.base.label_names):
            raise ValueError(
                f"class_index must be in [0, {len(self.base.label_names) - 1}], got {self.class_index}"
            )

        resolved_name = (
            str(class_name).strip().lower() if class_name is not None else self.base.label_names[self.class_index]
        )
        self.class_name = resolved_name
        self.include_other_labels = bool(include_other_labels)

        bit = 1 << self.class_index
        self._indices: List[int] = []

        for i, win in enumerate(self.base._index):  
            mask = int(win.label_mask)
            if mask & bit:
                if self.include_other_labels:
                    self._indices.append(i)
                else:
                    if mask == bit:
                        self._indices.append(i)

        self.stats = SingleClassStats(
            class_index=self.class_index,
            class_name=self.class_name,
            total_windows=len(self.base),
            selected_windows=len(self._indices),
        )

        if not self._indices:
            raise RuntimeError(
                f"No windows found for class '{self.class_name}' (index={self.class_index}). "
                "Try relaxing overlap thresholds or using include_other_labels=True."
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        base_idx = self._indices[int(idx)]
        x, _y = self.base[base_idx]
        return x
