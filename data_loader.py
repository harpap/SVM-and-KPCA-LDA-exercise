#!/usr/bin/env python3
"""
Load CIFAR-10 from the `cifar-10-batches-py` folder.

- Loads data_batch_1..5 as training set
- Loads test_batch as test set
- Loads batches.meta for label names
- Returns images as float32 in [0, 1] with shape (N, 32, 32, 3)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


def _load_batch(batch_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CIFAR-10 batch file.

    Returns
    -------
    images : np.ndarray, shape (N, 32, 32, 3), dtype float32 in [0, 1]
    labels : np.ndarray, shape (N,), dtype int64
    """
    with open(batch_path, "rb") as f:
        # CIFAR-10 python version was pickled with Python 2
        batch = pickle.load(f, encoding="latin1")

    data = batch["data"]  # shape (N, 3072)
    labels = np.array(batch["labels"], dtype=np.int64)

    # reshape to (N, 3, 32, 32) then to (N, 32, 32, 3)
    images = data.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    # scale to [0, 1]
    images = images.astype("float32") / 255.0

    return images, labels


def load_cifar10(folder: str) -> Dict[str, Any]:
    """
    Load the full CIFAR-10 dataset from `folder`.

    Parameters
    ----------
    folder : str
        Path to the cifar-10-batches-py directory.

    Returns
    -------
    A dict with:
        "x_train"      : (50000, 32, 32, 3)
        "y_train"      : (50000,)
        "x_test"       : (10000, 32, 32, 3)
        "y_test"       : (10000,)
        "label_names"  : list of 10 class names (strings)
    """
    folder_path = Path(folder)

    # --- load training batches ---
    train_images_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_file = folder_path / f"data_batch_{i}"
        imgs, lbls = _load_batch(batch_file)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)

    x_train = np.concatenate(train_images_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)

    # --- load test batch ---
    test_file = folder_path / "test_batch"
    x_test, y_test = _load_batch(test_file)

    # --- load meta (label names) ---
    meta_file = folder_path / "batches.meta"
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="latin1")

    # label_names come as bytes in some versions; decode to str
    label_names = meta["label_names"]
    label_names = [ln.decode("utf-8") if isinstance(ln, bytes) else ln
                   for ln in label_names]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "label_names": label_names,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 from cifar-10-batches-py folder."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cifar-10-batches-py",
        help="Path to cifar-10-batches-py folder",
    )
    args = parser.parse_args()

    dataset = load_cifar10(args.data_dir)

    print("x_train:", dataset["x_train"].shape)
    print("y_train:", dataset["y_train"].shape)
    print("x_test: ", dataset["x_test"].shape)
    print("y_test: ", dataset["y_test"].shape)
    print("label_names:", dataset["label_names"])
