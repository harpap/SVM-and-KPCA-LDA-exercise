#!/usr/bin/env python3
"""
Dataset loader utilities for the SVM / KPCA / LDA exercise.

Supports:
- CIFAR-10 (python version, cifar-10-batches-py)
- Breast cancer (UCI / LIBSVM format, with optional scaled version)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np


# ----------------------------------------------------------------------
# CIFAR-10
# ----------------------------------------------------------------------
def _load_cifar_batch(batch_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
        imgs, lbls = _load_cifar_batch(batch_file)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)

    x_train = np.concatenate(train_images_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)

    # --- load test batch ---
    test_file = folder_path / "test_batch"
    x_test, y_test = _load_cifar_batch(test_file)

    # --- load meta (label names) ---
    meta_file = folder_path / "batches.meta"
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="latin1")

    # label_names come as bytes in some versions; decode to str
    label_names = meta["label_names"]
    label_names = [
        ln.decode("utf-8") if isinstance(ln, bytes) else ln
        for ln in label_names
    ]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "label_names": label_names,
    }


# ----------------------------------------------------------------------
# Breast cancer (UCI / LIBSVM format)
# ----------------------------------------------------------------------
def _find_existing_file(folder: Path, base_name: str) -> Path:
    """
    Try a few common extensions and return the first existing path.
    """
    candidates = [base_name, base_name + ".txt", base_name + ".data"]
    for name in candidates:
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} inside {folder}"
    )


def _load_libsvm_dense(path: Path, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a LIBSVM-formatted file into dense numpy arrays.

    Each line: <label> index1:value1 index2:value2 ...
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # first token is the label (2 for benign, 4 for malignant)
            y_val = int(float(parts[0]))

            features = np.zeros(n_features, dtype=float)
            for token in parts[1:]:
                idx_str, val_str = token.split(":")
                idx = int(idx_str) - 1  # indices in file start from 1
                if 0 <= idx < n_features:
                    features[idx] = float(val_str)

            X_list.append(features)
            y_list.append(y_val)

    X = np.vstack(X_list)
    y_raw = np.array(y_list, dtype=int)

    return X, y_raw


def load_breast_cancer(folder: str, scaled: bool = False) -> Dict[str, Any]:
    """
    Load the breast-cancer dataset (UCI / Wisconsin) from `folder`.

    Parameters
    ----------
    folder : str
        Folder containing breast-cancer(.txt) and/or breast-cancer_scale(.txt).
    scaled : bool, default False
        If True, use the *scaled* version (features scaled to [-1, 1]).

    Returns
    -------
    A dict with:
        "x"             : (683, 10) feature matrix (float64)
        "y"             : (683,) labels in {0, 1}
        "y_raw"         : (683,) original labels {2, 4}
        "feature_names" : list of 10 feature names
        "class_names"   : {0: "benign", 1: "malignant"}
    """
    folder_path = Path(folder)

    base_name = "breast-cancer_scale" if scaled else "breast-cancer"
    data_path = _find_existing_file(folder_path, base_name)

    n_features = 10
    X, y_raw = _load_libsvm_dense(data_path, n_features=n_features)

    # Map original labels {2, 4} -> {0, 1}
    # 2 = benign, 4 = malignant
    y = np.where(y_raw == 4, 1, 0)

    feature_names = [
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
    ]

    class_names = {0: "benign", 1: "malignant"}

    return {
        "x": X,
        "y": y,
        "y_raw": y_raw,
        "feature_names": feature_names,
        "class_names": class_names,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 or breast-cancer datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "breast-cancer"],
        default="cifar10",
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset folder "
             "(e.g. cifar-10-batches-py or breast-cancer).",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="Use scaled version for breast-cancer (breast-cancer_scale). "
             "Ignored for CIFAR-10.",
    )

    args = parser.parse_args()

    if args.dataset == "cifar10":
        dataset = load_cifar10(args.data_dir)
        print("Loaded CIFAR-10")
        print("x_train:", dataset["x_train"].shape)
        print("y_train:", dataset["y_train"].shape)
        print("x_test: ", dataset["x_test"].shape)
        print("y_test: ", dataset["y_test"].shape)
        print("label_names:", dataset["label_names"])
    else:
        dataset = load_breast_cancer(args.data_dir, scaled=args.scaled)
        print("Loaded breast-cancer dataset")
        print("x:", dataset["x"].shape)
        print("y:", dataset["y"].shape)
        print("y_raw classes:", np.unique(dataset["y_raw"]))
        print("feature_names:", dataset["feature_names"])
        print("class_names:", dataset["class_names"])
