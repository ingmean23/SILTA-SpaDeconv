"""Small read-only H5AD helpers used by the isolated benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _values(node: h5py.Dataset) -> list[Any]:
    return [_decode(value) for value in node[:]]


def _index(group: h5py.Group) -> list[str]:
    return [str(value) for value in _values(group["_index"])]


@dataclass(frozen=True)
class H5ADInfo:
    path: Path
    shape: tuple[int, int]
    obs_index: list[str]
    var_names: list[str]


def inspect_h5ad(path: str | Path) -> H5ADInfo:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        node = handle["X"]
        shape = tuple(int(v) for v in (
            node.shape if isinstance(node, h5py.Dataset) else node.attrs["shape"]
        ))
        obs_index = _index(handle["obs"])
        var_names = _index(handle["var"])
    if shape != (len(obs_index), len(var_names)):
        raise ValueError(f"H5AD shape/index mismatch: {path}")
    return H5ADInfo(path, shape, obs_index, var_names)


def read_selected_columns(path: str | Path, columns: list[int]):
    columns = [int(value) for value in columns]
    with h5py.File(path, "r") as handle:
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[:, columns])
        encoding = str(_decode(node.attrs.get("encoding-type", "")))
        shape = tuple(int(value) for value in node.attrs["shape"])
        matrix = sparse.csr_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]),
            shape=shape,
        ) if encoding == "csr_matrix" else sparse.csc_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]),
            shape=shape,
        ) if encoding == "csc_matrix" else None
        if matrix is None:
            raise ValueError(f"Unsupported H5AD X encoding: {encoding}")
        return matrix[:, columns].tocsr()


def read_row_sums(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[:], dtype=np.float64).sum(axis=1)
        shape = tuple(int(value) for value in node.attrs["shape"])
        encoding = str(_decode(node.attrs.get("encoding-type", "")))
        values = np.asarray(node["data"][:], dtype=np.float64)
        if encoding == "csr_matrix":
            return np.asarray(sparse.csr_matrix(
                (values, node["indices"][:], node["indptr"][:]), shape=shape,
            ).sum(axis=1)).ravel()
        if encoding == "csc_matrix":
            return np.asarray(sparse.csc_matrix(
                (values, node["indices"][:], node["indptr"][:]), shape=shape,
            ).sum(axis=1)).ravel()
        raise ValueError(f"Unsupported H5AD X encoding: {encoding}")

