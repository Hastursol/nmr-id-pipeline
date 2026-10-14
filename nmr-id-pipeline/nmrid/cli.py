"""Command-line interface:  python -m nmrid <command> [options]

    inspect     print the parsed 1H peak lists of the first SD records (check the parser)
    prepare     parse nmrshiftdb2 (or generate synthetic data) and build the dataset
    train       train one model (rf | cnn | siamese)
    experiment  run RQ1-RQ4 and/or the open-set experiment
    search      rank the library compounds for a measured spectrum (Bruker folder or CSV)

Any configuration value can be overridden with --set, e.g. --set cnn.epochs=5
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def _cfg(args):
    from .config import deep_update, load_config, parse_override

    overrides: dict = {}
    for item in getattr(args, "set", None) or []:
        overrides = deep_update(overrides, parse_override(item))
    return load_config(args.config, overrides)


def cmd_inspect(args):
    from .io_nmrshiftdb import HAVE_RDKIT, describe_record, iter_sd_records

    print(f"RDKit available: {HAVE_RDKIT}\n")
    shown = 0
    for molblock, fields in iter_sd_records(args.sdf):
        text = describe_record(molblock, fields)
        if args.only_1h and "ppm" not in text:
            continue
        print(text, "\n")
        shown += 1
        if shown >= args.n:
            break


def cmd_prepare(args):
    from .config import dataset_path, run_dir
    from .dataset import build_dataset, load_compounds

    cfg = _cfg(args)
    t0 = time.perf_counter()
    compounds, stats = load_compounds(cfg)
    print("source statistics:", json.dumps(stats, indent=2))
    ds = build_dataset(compounds, cfg)
    path = dataset_path(cfg)
    ds.save(path)
    with open(run_dir(cfg) / "dataset_info.json", "w") as fh:
        json.dump({"source_stats": stats, **ds.info}, fh, indent=2)
    print(f"dataset: {ds.info}")
    print(f"saved {path} ({time.perf_counter() - t0:.0f}s)")


def cmd_train(args):
    from .experiments import Runner

    runner = Runner(_cfg(args))
    getattr(runner, args.model)(seed=args.seed)
    print("done")


def cmd_experiment(args):
    from .experiments import Runner

    Runner(_cfg(args)).run(args.rq)


def cmd_search(args):
    from .experiments import Runner
    from .io_experimental import read_experimental
    from .search import rank_library

    runner = Runner(_cfg(args), verbose=False)
    exclude = [args.exclude[i: i + 2] for i in range(0, len(args.exclude or []), 2)]
    query = read_experimental(args.query, runner.ds.grid, fmt=args.format, exclude=exclude)
    result = rank_library(query["spectrum"], runner, args.model, args.seed, args.top)
    print(f"query: {query['name']}   model: {args.model}")
    print(result.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)
        np.save(str(args.out).rsplit(".", 1)[0] + "_binned.npy", query["spectrum"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m nmrid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp):
        sp.add_argument("-c", "--config", default="configs/default.yaml")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
        return sp

    sp = sub.add_parser("inspect", help="show parsed peak lists from an SD file")
    sp.add_argument("--sdf", default="data/raw/nmrshiftdb2withsignals.sd")
    sp.add_argument("-n", type=int, default=5)
    sp.add_argument("--all-records", dest="only_1h", action="store_false",
                    help="also show records without a 1H spectrum")
    sp.set_defaults(func=cmd_inspect)

    with_config(sub.add_parser("prepare", help="build the dataset")).set_defaults(func=cmd_prepare)

    sp = with_config(sub.add_parser("train", help="train one model"))
    sp.add_argument("--model", choices=["rf", "cnn", "siamese"], required=True)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_train)

    sp = with_config(sub.add_parser("experiment", help="run experiments"))
    sp.add_argument("--rq", nargs="+", default=["all"], choices=["1", "2", "3", "4", "openset", "all"])
    sp.set_defaults(func=cmd_experiment)

    sp = with_config(sub.add_parser("search", help="identify a measured spectrum"))
    sp.add_argument("--query", required=True, help="Bruker experiment folder or CSV (ppm,intensity)")
    sp.add_argument("--format", default="auto", choices=["auto", "bruker", "csv"])
    sp.add_argument("--model", default="siamese", choices=["cosine", "rf", "cnn", "siamese"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--exclude", type=float, nargs="*", metavar="PPM",
                    help="pairs of ppm limits to blank, e.g. --exclude 7.20 7.32 1.50 1.62")
    sp.add_argument("--out", help="write the ranking to this CSV")
    sp.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
