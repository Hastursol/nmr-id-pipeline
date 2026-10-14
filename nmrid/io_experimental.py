"""Readers for measured 1H spectra (Bruker, CSV) onto the shared ppm grid.

The Bruker reader plays the role of ``readBruker.py`` in the pSCNN code, but is
written differently: the ppm axis comes from the processing parameters
(``OFFSET``, ``SW_p``, ``SF``, ``SI`` in ``procs``) of the processed spectrum,
or from nmrglue's unit-conversion object for a raw FID, and the spectrum is
then integrated into the same 2,000 bins used for the database spectra
(``spectra.bin_continuous``) instead of being kept at instrument resolution.

Accepted Bruker layouts (``path`` can be any of them):
    <sample>/1/pdata/1/1r    (the layout of the pSCNN data on Zenodo)
    <experiment>/pdata/1/1r
    <experiment>/pdata/1     (the processed-data folder itself)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectra import Grid, bin_continuous, exclude_regions


def _find_bruker_dirs(path: Path) -> tuple[Path | None, Path | None]:
    """Return (experiment dir, processed-data dir) for the supported layouts."""
    path = Path(path)
    if (path / "1r").exists() and (path / "procs").exists():
        return path.parent.parent, path
    for exp in (path, path / "1"):
        if (exp / "pdata" / "1" / "1r").exists():
            return exp, exp / "pdata" / "1"
        if (exp / "fid").exists() and (exp / "acqus").exists():
            return exp, None
    return None, None


def read_bruker(path: str | Path, use_raw_fid: bool = False) -> dict:
    """Read a Bruker 1D 1H experiment; returns ``{"name", "ppm", "intensity"}``."""
    import nmrglue as ng

    exp_dir, pdata_dir = _find_bruker_dirs(Path(path))
    if exp_dir is None:
        raise FileNotFoundError(f"No Bruker experiment found under {path}")

    if pdata_dir is not None and not use_raw_fid:
        dic, data = ng.bruker.read_pdata(str(pdata_dir), scale_data=True)
        procs = dic["procs"]
        si = int(procs["SI"])
        ppm = float(procs["OFFSET"]) - np.arange(si) * (float(procs["SW_p"]) / float(procs["SF"])) / si
        intensity = np.asarray(data, dtype=float).real
    else:
        dic, fid = ng.bruker.read(str(exp_dir))
        data = ng.bruker.remove_digital_filter(dic, fid)
        data = ng.proc_base.zf_size(data, 2 * data.shape[-1])
        data = ng.proc_base.fft(data)
        data = ng.proc_autophase.autops(data, "acme", disp=False)
        data = ng.proc_bl.cbf(data)
        udic = ng.bruker.guess_udic(dic, data)
        ppm = ng.fileiobase.uc_from_udic(udic).ppm_scale()
        intensity = np.asarray(data).real
    name = Path(path).resolve().name
    return {"name": name, "ppm": np.asarray(ppm, float), "intensity": intensity}


def read_csv_spectrum(path: str | Path) -> dict:
    """Two columns, ppm and intensity; a header row and any delimiter are tolerated."""
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return {"name": Path(path).stem, "ppm": df.iloc[:, 0].to_numpy(float),
            "intensity": df.iloc[:, 1].to_numpy(float)}


def read_experimental(path: str | Path, grid: Grid, fmt: str = "auto",
                      exclude: list | None = None, use_raw_fid: bool = False) -> dict:
    """Read a measured spectrum and bin it onto ``grid``.

    ``exclude`` is a list of [low, high] ppm windows to blank out (solvent, water).
    """
    path = Path(path)
    if fmt == "auto":
        fmt = "csv" if path.is_file() and path.suffix.lower() in (".csv", ".txt", ".tsv", ".dat") else "bruker"
    raw = read_csv_spectrum(path) if fmt == "csv" else read_bruker(path, use_raw_fid=use_raw_fid)
    spec = bin_continuous(raw["ppm"], raw["intensity"], grid)
    if exclude:
        spec = exclude_regions(spec, grid, exclude)
    raw["spectrum"] = spec
    return raw
