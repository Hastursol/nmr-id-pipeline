"""The Bruker reader recovers peak positions from a processed (pdata) spectrum."""
import numpy as np
import pytest

from nmrid.io_experimental import read_experimental
from nmrid.spectra import Grid

ng = pytest.importorskip("nmrglue")


def test_bruker_pdata_ppm_axis(tmp_path):
    si, sf, sw_hz, offset = 32768, 400.13, 16 * 400.13, 14.0  # 14 ppm down to -2 ppm
    ppm = offset - np.arange(si) * (sw_hz / sf) / si
    data = (1 / (1 + ((ppm - 3.50) / 0.002) ** 2) + 0.5 / (1 + ((ppm - 7.10) / 0.002) ** 2)).astype("float32")
    pdata = tmp_path / "sample" / "1" / "pdata" / "1"
    pdata.mkdir(parents=True)
    procs = {"SI": si, "SF": sf, "SW_p": sw_hz, "OFFSET": offset, "BYTORDP": 0, "DTYPP": 0,
             "NC_proc": 0, "XDIM": si, "_comments": [],
             "_coreheader": ["##TITLE= Parameter file", "##JCAMPDX= 5.0"]}
    ng.bruker.write_pdata(str(pdata), {"procs": procs}, data, write_procs=True, overwrite=True)
    (tmp_path / "sample" / "1" / "acqus").write_text("##TITLE= test\n##$SW= 16\n##END=\n")

    grid = Grid()
    spec = read_experimental(tmp_path / "sample", grid)["spectrum"]
    top = sorted(grid.centers[np.argsort(spec)[::-1][:2]])
    assert abs(top[0] - 3.50) < 2 * grid.width and abs(top[1] - 7.10) < 2 * grid.width
