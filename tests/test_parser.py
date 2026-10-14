import pytest

from nmrid.io_nmrshiftdb import (_multiplicity_partners, iter_sd_records, load_nmrshiftdb,
                                 parse_nmredata_1h)
from nmrid.spectra import Grid

Chem = pytest.importorskip("rdkit.Chem")


def _record(molblock: str, fields: dict) -> str:
    body = "".join(f"> <{k}>\n{v}\n\n" for k, v in fields.items())
    return f"{molblock}\n{body}$$$$\n"


def test_classic_format_counts_protons_on_heavy_atoms(tmp_path):
    mb = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))  # C1 (CH3), C2 (CH2), O3 (OH)
    sd = _record(mb, {"nmrshiftdb2 ID": "1", "Spectrum 1H 0": "1.22;0.0;1|3.69;0.0;2|2.61;0.0;3|"})
    path = tmp_path / "ethanol.sd"
    path.write_text(sd)
    comps, stats = load_nmrshiftdb(path, Grid(), progress=False)
    assert stats["compounds"] == 1
    res = comps[0]["spectra"][0]["resonances"]
    assert [(r.shift, r.n_h) for r in res] == [(1.22, 3.0), (2.61, 1.0), (3.69, 2.0)]


def test_classic_format_with_explicit_hydrogens(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    mb = Chem.MolToMolBlock(mol)
    entries = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            heavy = atom.GetNeighbors()[0].GetIdx()
            shift = {0: 1.22, 1: 3.69, 2: 2.61}[heavy]
            entries.append(f"{shift};0.0;{atom.GetIdx() + 1}")
    path = tmp_path / "ethanol_h.sd"
    path.write_text(_record(mb, {"Spectrum 1H 0": "|".join(entries)}))
    comps, _ = load_nmrshiftdb(path, Grid(), progress=False)
    res = comps[0]["spectra"][0]["resonances"]
    assert [(r.shift, r.n_h) for r in res] == [(1.22, 3.0), (2.61, 1.0), (3.69, 2.0)]


def test_multiple_spectra_of_one_compound_are_grouped(tmp_path):
    mb = Chem.MolToMolBlock(Chem.MolFromSmiles("CC(=O)C"))
    sd = _record(mb, {"Spectrum 1H 0": "2.17;0.0;1|2.17;0.0;4"}) + \
        _record(mb, {"Spectrum 1H 1": "2.09;0.0;1|2.09;0.0;4"})
    path = tmp_path / "acetone.sd"
    path.write_text(sd)
    comps, stats = load_nmrshiftdb(path, Grid(), min_peaks=1, progress=False)
    assert stats["compounds"] == 1 and len(comps[0]["spectra"]) == 2
    assert comps[0]["spectra"][0]["resonances"][0].n_h == 6.0


def test_nmredata_block():
    block = "Larmor=400.13\\\n1.22, S=t, N=3, L=H1, J=7.0(H2)\\\n3.69, S=q, N=2, L=H2, J=7.0(H1)\\\n2.61, S=br s, N=1, L=H3\\"
    res, larmor = parse_nmredata_1h(block)
    assert larmor == pytest.approx(400.13)
    assert [r.shift for r in res] == [1.22, 2.61, 3.69]
    assert res[0].couplings == ((7.0, 2),) and res[0].n_h == 3
    assert res[2].couplings == ((7.0, 3),)
    assert res[1].broad and res[1].couplings == ()


def test_multiplicity_codes():
    assert _multiplicity_partners("dt") == [1, 2]
    assert _multiplicity_partners("s") == []
    assert _multiplicity_partners("m") is None
    assert _multiplicity_partners("quint") == [4]


def test_iter_sd_records_counts(tmp_path):
    mb = Chem.MolToMolBlock(Chem.MolFromSmiles("C"))
    path = tmp_path / "m.sd"
    path.write_text(_record(mb, {"a": "1"}) * 3)
    assert sum(1 for _ in iter_sd_records(path)) == 3
