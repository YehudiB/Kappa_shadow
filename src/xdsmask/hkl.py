from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

INT_ITEMS = {"h", "k", "l", "iseg"}

DEFAULT_ITEMS = [
    "h", "k", "l", "iobs", "sigma", "xcal", "ycal", "zcal", "rlp",
    "peak", "corr", "maxc", "xobs", "yobs", "zobs", "alf0", "bet0",
    "alf1", "bet1", "psi", "iseg",
]


def _parse_header_lines(lines):
    header = {}
    items = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith("!"):
            break
        body = line[1:].strip()
        if body.upper().startswith("END_OF_HEADER"):
            break
        if "=" in body:
            tokens = body.split()
            i = 0
            while i < len(tokens):
                key, _, val = tokens[i].partition("=")
                if "=" in tokens[i]:
                    if not val and i + 1 < len(tokens):
                        i += 1
                        val = tokens[i]
                    header[key.lower()] = val
                i += 1
        elif "," in body:
            items.extend(t.strip().lower() for t in body.split(",") if t.strip())
    return header, items


def _records_dtype(items):
    return np.dtype([(name, np.int32 if name in INT_ITEMS else np.float64) for name in items])


@dataclass
class IntegrateHKL:
    header: dict
    items: list
    header_lines: list
    records: np.ndarray

    @classmethod
    def read(cls, path):
        path = Path(path)
        lines = path.read_text().splitlines()
        header_lines = [raw for raw in lines if raw.strip().startswith("!")]
        header, items = _parse_header_lines(header_lines)
        if not items:
            n_items = int(float(header.get("number_of_items_in_each_data_record", len(DEFAULT_ITEMS))))
            items = DEFAULT_ITEMS[:n_items]
        data = np.loadtxt(path, comments="!", ndmin=2)
        records = np.empty(data.shape[0], dtype=_records_dtype(items))
        for j, name in enumerate(items):
            records[name] = data[:, j]
        return cls(header=header, items=items, header_lines=header_lines, records=records)

    def write(self, path, records=None):
        records = self.records if records is None else records
        path = Path(path)
        with path.open("w") as fh:
            for line in self.header_lines:
                fh.write(line.rstrip("\n") + "\n")
            for rec in records:
                parts = []
                for name in self.items:
                    if name in INT_ITEMS:
                        parts.append(f"{int(rec[name]):5d}")
                    else:
                        parts.append(f"{float(rec[name]):13.5E}")
                fh.write(" ".join(parts) + "\n")


def write_filter_hkl(path, records, masked, dx=0.5, dy=0.5, dz=0.5):
    path = Path(path)
    with path.open("w") as fh:
        for rec, m in zip(records, masked):
            if not m:
                continue
            fh.write(
                f"{int(rec['h']):5d}{int(rec['k']):5d}{int(rec['l']):5d}"
                f" {float(rec['xcal']):12.4f} {float(rec['ycal']):12.4f} {float(rec['zcal']):12.4f}"
                f" {dx:6.3f} {dy:6.3f} {dz:6.3f}\n"
            )
