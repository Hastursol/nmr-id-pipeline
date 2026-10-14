"""Reader for nmrshiftdb2 SD files.

Two spectrum encodings are understood:

1. The classic nmrshiftdb2 export (``nmrshiftdb2withsignals.sd``), in which each
   spectrum is a data field such as ``> <Spectrum 1H 0>`` whose value is a
   ``|``-separated list of ``shift;intensity[multiplicity];atom`` entries.
   The atom number is a 1-based index into the molfile.  The number of protons
   behind each entry is looked up with RDKit (an H atom counts 1, a heavy atom
   counts its attached hydrogens); without RDKit every entry counts 1.

2. NMReDATA tags (``> <NMREDATA_1D_1H>``), used by the nmredata exports, whose
   signal lines look like ``7.26, S=d, N=2, L=H3, J=8.1(H4)``.  These carry the
   multiplicity and coupling constants, so multiplets are rendered.

The SD file is streamed record by record, so the full database never has to fit
in memory.  Run ``python -m nmrid inspect`` on the downloaded file first: it
prints the parsed peak lists of the first records so the format assumptions can
be checked by eye before any training.
"""
from __future__ import annotations

import gzip
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

from .spectra import Resonance

try:  # RDKit is optional: used for proton counts and compound identity
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:  # pragma: no cover
    HAVE_RDKIT = False

_FIELD = re.compile(r"^>\s*<([^>]+)>")
_CLASSIC_1H = re.compile(r"^Spectrum\s+1H\b", re.IGNORECASE)
_NMREDATA_1H = re.compile(r"^NMREDATA_1D_1H$", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------- SD I/O
def iter_sd_records(path: str | Path):
    """Yield ``(molblock, fields)`` for each record of an SD file (.sd, .sdf, .gz)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        mol_lines: list[str] = []
        fields: "OrderedDict[str, list[str]]" = OrderedDict()
        current: str | None = None
        in_mol = True
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.strip() == "$$$$":
                yield "\n".join(mol_lines), {k: "\n".join(v).strip() for k, v in fields.items()}
                mol_lines, fields, current, in_mol = [], OrderedDict(), None, True
                continue
            if in_mol:
                mol_lines.append(line)
                if line.startswith("M  END"):
                    in_mol = False
                continue
            m = _FIELD.match(line)
            if m:
                current = m.group(1).strip()
                fields[current] = []
            elif current is not None:
                if line.strip() == "":
                    current = None
                else:
                    fields[current].append(line)
        if mol_lines and any(s.strip() for s in mol_lines):
            yield "\n".join(mol_lines), {k: "\n".join(v).strip() for k, v in fields.items()}


# ---------------------------------------------------------------- molecules
def _mol_for_indices(molblock: str):
    """Molecule with the molfile's atom order intact (no H removal, no sanitisation)."""
    if not HAVE_RDKIT:
        return None
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
    return mol


def protons_on_atom(mol, atom_number: int | None) -> int:
    if mol is None or atom_number is None or not 1 <= atom_number <= mol.GetNumAtoms():
        return 1
    atom = mol.GetAtomWithIdx(atom_number - 1)
    if atom.GetAtomicNum() == 1:
        return 1
    try:
        return max(1, int(atom.GetTotalNumHs(includeNeighbors=True)))
    except Exception:
        return 1


def compound_identity(molblock: str, fields: dict) -> tuple[str, str, str]:
    """Return (key, display name, SMILES).  The key groups spectra of one compound."""
    title = molblock.split("\n", 1)[0].strip()
    record_id = (
        fields.get("nmrshiftdb2 ID") or fields.get("nmrshiftdb2 id") or fields.get("ID") or title
    )
    name = fields.get("Name") or fields.get("NAME") or title or str(record_id)
    smiles = ""
    if HAVE_RDKIT:
        try:
            mol = Chem.MolFromMolBlock(molblock)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                key = Chem.MolToInchiKey(mol) or smiles
                if key:
                    return key, name, smiles
        except Exception:
            pass
    return f"record:{record_id}", name, smiles


# ------------------------------------------------------------------ parsers
def parse_classic_1h(value: str, mol=None) -> list[Resonance]:
    """Parse ``shift;intensity[mult];atom|...`` into resonances (grouped by shift)."""
    text = value.replace("\n", "")
    seen: set = set()
    by_shift: dict = defaultdict(float)
    for entry in text.split("|"):
        parts = [p.strip() for p in entry.split(";")]
        if not parts or not parts[0]:
            continue
        try:
            shift = float(parts[0])
        except ValueError:
            continue
        atom = None
        if len(parts) >= 3:
            try:
                atom = int(float(parts[-1]))
            except ValueError:
                atom = None
        key = (round(shift, 3), atom)
        if atom is not None and key in seen:  # same atom listed twice at one shift
            continue
        seen.add(key)
        by_shift[round(shift, 3)] += protons_on_atom(mol, atom)
    return [Resonance(s, float(n)) for s, n in sorted(by_shift.items())]


_MULT_CODES = [("quint", 4), ("sext", 5), ("sept", 6), ("hept", 6), ("p", 4), ("h", 5),
               ("d", 1), ("t", 2), ("q", 3)]


def _multiplicity_partners(mult: str) -> list[int] | None:
    """'dt' -> [1, 2]; 's' -> []; 'm' / 'br' -> None (no resolvable pattern)."""
    m = mult.lower().replace(" ", "").replace(".", "")
    if m in ("", "s"):
        return []
    if m in ("br", "brs", "bs") or "m" in m:
        return None
    m = m.replace("br", "")
    out: list[int] = []
    while m:
        if m[0] == "s":
            m = m[1:]
            continue
        for code, n in _MULT_CODES:
            if m.startswith(code):
                out.append(n)
                m = m[len(code):]
                break
        else:
            return None
    return out


def parse_nmredata_1h(value: str) -> tuple[list[Resonance], float | None]:
    """Parse an NMREDATA_1D_1H block.  Returns (resonances, Larmor frequency or None)."""
    larmor = None
    resonances: list[Resonance] = []
    for raw in value.splitlines():
        line = raw.strip().rstrip("\\").strip()
        if not line or line.startswith(";"):
            continue
        head = line.split(",", 1)[0].strip()
        if "=" in head:  # header line, e.g. Larmor=400.13
            k, _, v = head.partition("=")
            if k.strip().lower() == "larmor":
                nums = _NUMBER.findall(v)
                larmor = float(nums[0]) if nums else None
            continue
        nums = _NUMBER.findall(head.replace(" ", ""))
        if not nums:
            continue
        if "-" in head[1:] and len(nums) >= 2:  # a range such as 7.20-7.35
            shift = 0.5 * (abs(float(nums[0])) + abs(float(nums[1])))
        else:
            shift = float(nums[0])
        attrs = {}
        for m in re.finditer(r"([A-Za-z]+)\s*=\s*(.*?)(?=,\s*[A-Za-z]+\s*=|$)", line):
            attrs[m.group(1).upper()] = m.group(2).strip()
        n_h = 1.0
        if "N" in attrs and _NUMBER.findall(attrs["N"]):
            n_h = float(_NUMBER.findall(attrs["N"])[0])
        elif "E" in attrs and _NUMBER.findall(attrs["E"]):
            n_h = float(_NUMBER.findall(attrs["E"])[0])
        mult = attrs.get("S", "")
        partners = _multiplicity_partners(mult)
        j_values = [float(x) for x in _NUMBER.findall(re.sub(r"\([^)]*\)", "", attrs.get("J", "")))]
        j_values = [abs(j) for j in j_values if abs(j) > 0]
        couplings: tuple = ()
        if partners and len(partners) == len(j_values):
            couplings = tuple(zip(j_values, partners))
        broad = partners is None or "br" in mult.lower()
        resonances.append(Resonance(shift, n_h, couplings, broad=broad))
    return sorted(resonances, key=lambda r: r.shift), larmor


def _spectrum_meta(fields: dict, spectrum_field: str) -> dict:
    """Collect solvent / temperature / field fields that belong to a spectrum."""
    suffix = spectrum_field.split()[-1]
    meta = {}
    for k, v in fields.items():
        lk = k.lower()
        if any(t in lk for t in ("solvent", "temperature", "field strength", "frequency")):
            if k.endswith(suffix) or not re.search(r"\d+$", k):
                meta[k] = v
    return meta


def spectra_from_record(molblock: str, fields: dict) -> list[dict]:
    mol = None
    spectra = []
    for name, value in fields.items():
        if _CLASSIC_1H.match(name):
            if mol is None:
                mol = _mol_for_indices(molblock)
            res = parse_classic_1h(value, mol)
            meta = _spectrum_meta(fields, name)
            larmor = None
            for k, v in meta.items():
                if "field" in k.lower() or "frequency" in k.lower():
                    nums = _NUMBER.findall(v)
                    larmor = float(nums[0]) if nums else None
            spectra.append({"resonances": res, "larmor": larmor or 400.0, "meta": {"field": name, **meta}})
        elif _NMREDATA_1H.match(name.strip()):
            res, larmor = parse_nmredata_1h(value)
            spectra.append({"resonances": res, "larmor": larmor or 400.0, "meta": {"field": name}})
    return spectra


# ------------------------------------------------------------------ loading
def load_nmrshiftdb(path: str | Path, grid, min_peaks: int = 2, max_records: int | None = None,
                    progress: bool = True) -> tuple[list[dict], dict]:
    """Group all usable 1H spectra by compound.

    Returns (compounds, stats).  Each compound is
    ``{"key", "name", "smiles", "spectra": [{"resonances", "larmor", "meta"}, ...]}``.
    """
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None
    stats = defaultdict(int)
    compounds: "OrderedDict[str, dict]" = OrderedDict()
    records = iter_sd_records(path)
    if progress and tqdm is not None:
        records = tqdm(records, desc="Reading SD records", unit="rec")
    for i, (molblock, fields) in enumerate(records):
        if max_records is not None and i >= max_records:
            break
        stats["records"] += 1
        spectra = spectra_from_record(molblock, fields)
        if not spectra:
            continue
        stats["records_with_1h"] += 1
        key, name, smiles = compound_identity(molblock, fields)
        for spec in spectra:
            stats["spectra_1h"] += 1
            spec["resonances"] = [r for r in spec["resonances"]
                                  if grid.ppm_min <= r.shift <= grid.ppm_max]
            if len(spec["resonances"]) < min_peaks:
                stats["spectra_too_few_peaks"] += 1
                continue
            entry = compounds.setdefault(key, {"key": key, "name": name, "smiles": smiles, "spectra": []})
            entry["spectra"].append(spec)
            stats["spectra_kept"] += 1
    out = list(compounds.values())
    stats["compounds"] = len(out)
    stats["compounds_with_multiple_spectra"] = sum(len(c["spectra"]) > 1 for c in out)
    stats["rdkit"] = HAVE_RDKIT
    return out, dict(stats)


def describe_record(molblock: str, fields: dict) -> str:
    key, name, smiles = compound_identity(molblock, fields)
    lines = [f"{name}  |  {smiles or '(no SMILES)'}  |  key={key}"]
    lines.append("  fields: " + ", ".join(fields.keys()))
    for spec in spectra_from_record(molblock, fields):
        lines.append(f"  {spec['meta'].get('field')}: larmor={spec['larmor']}  meta={spec['meta']}")
        for r in spec["resonances"]:
            j = ", ".join(f"J={jj:.1f}Hz x{n}" for jj, n in r.couplings)
            lines.append(f"    {r.shift:7.3f} ppm  nH={r.n_h:g}  {j}{'  broad' if r.broad else ''}")
    return "\n".join(lines)
