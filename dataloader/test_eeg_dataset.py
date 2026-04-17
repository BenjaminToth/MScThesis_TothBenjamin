from __future__ import annotations

import argparse
from pathlib import Path

from dataloader.eeg_dataset import make_split_dataloaders


def _print_loader_shapes(name: str, loader) -> None:
    dataset = loader.dataset
    print(f"[{name}] files={len(dataset.edf_paths)} windows={len(dataset)}")

    x, y = next(iter(loader))
    print(f"[{name}] x shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"[{name}] y shape={tuple(y.shape)} dtype={y.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/dev/test EEG DataLoaders and print batch shapes."
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default="data/01_tcp_ar_split",
        help="Root directory containing train/dev/test split folders.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument(
        "--min-overlap-sec",
        type=float,
        default=0.5,
        help="Label a window only if artifact overlap >= this many seconds (0 disables this check).",
    )
    parser.add_argument(
        "--min-overlap-frac-artifact",
        type=float,
        default=0.5,
        help="Label a window only if artifact overlap >= this fraction of the artifact duration (0 disables this check).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-window channel-wise z-score normalization.",
    )

    args = parser.parse_args()

    split_root = Path(args.split_root)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    loaders = make_split_dataloaders(
        split_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_kwargs={
            "sfreq": args.sfreq,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "min_overlap_sec": args.min_overlap_sec,
            "min_overlap_frac_of_artifact": args.min_overlap_frac_artifact,
            "normalize": not args.no_normalize,
        },
    )

    for split_name in ("train", "dev", "test"):
        _print_loader_shapes(split_name, loaders[split_name])


if __name__ == "__main__":
    main()
