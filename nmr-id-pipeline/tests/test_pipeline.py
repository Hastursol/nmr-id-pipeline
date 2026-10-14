"""End-to-end smoke test on a tiny synthetic library (about a minute on a laptop CPU)."""
import numpy as np
import pandas as pd

from nmrid.cli import main
from nmrid.config import load_config


def test_end_to_end(tmp_path, capsys):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(f"""
run_name: tiny
paths: {{processed: {tmp_path}/processed, results: {tmp_path}/results}}
data: {{source: synthetic, n_compounds: 12, replicates: 10}}
rf: {{n_estimators: 10, n_jobs: 1}}
cnn: {{epochs: 1, batch_size: 32}}
siamese: {{epochs: 1, batch_size: 32}}
experiments: {{noise_levels: [0.0, 0.05], shift_ppm: [0.0, 0.02], train_fractions: [0.5, 1.0]}}
""")
    main(["prepare", "-c", str(cfg_path)])
    main(["experiment", "-c", str(cfg_path), "--rq", "1", "2", "4", "openset"])
    out = tmp_path / "results" / "tiny"
    rq1 = pd.read_csv(out / "tables" / "rq1_results.csv")
    assert set(rq1["model"]) == {"cosine", "rf", "cnn", "siamese"}
    assert rq1["top1"].between(0, 1).all()
    for name in ("rq1_comparison", "rq2_noise", "rq2_shift", "rq4_learning_curves", "example_spectra"):
        assert (out / "figures" / f"{name}.png").exists()
    assert (out / "tables" / "rq1_table.tex").exists()
    assert (out / "tables" / "openset_results.csv").exists()

    # library search with a CSV query made from a reference spectrum
    from nmrid.dataset import NMRDataset

    ds = NMRDataset.load(tmp_path / "processed" / "tiny_dataset.npz")
    q = tmp_path / "query.csv"
    pd.DataFrame({"ppm": ds.grid.centers[::-1], "i": ds.refs[3][::-1]}).to_csv(q, index=False)
    main(["search", "-c", str(cfg_path), "--query", str(q), "--model", "cosine", "--top", "3",
          "--out", str(tmp_path / "hits.csv")])
    hits = pd.read_csv(tmp_path / "hits.csv")
    assert hits.loc[0, "key"] == ds.keys[3]


def test_config_override():
    from nmrid.config import parse_override

    cfg = load_config(overrides=parse_override("cnn.epochs=3"))
    assert cfg["cnn"]["epochs"] == 3 and cfg["cnn"]["lr"] == 1e-3
    assert np.isclose(sum(cfg["data"]["split"]), 1.0)
