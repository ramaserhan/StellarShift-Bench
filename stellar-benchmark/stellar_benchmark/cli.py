"""Command-line interface for reproducible benchmark experiments."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .config import (
    load_cross_survey_config,
    load_desi_reliability_config,
    load_desi_snr_config,
    load_synthetic_demo_config,
)
from .cross_survey_experiment import run_cross_survey_experiment
from .data.desi import extract_desi_arm_spectra
from .desi_experiment import run_desi_snr_experiment
from .demo import run_synthetic_demo
from .reliability_experiment import run_desi_reliability_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stellar-benchmark",
        description="Run reproducible stellar domain-shift benchmark experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="run an experiment from a YAML configuration"
    )
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override output_dir from the YAML file",
    )
    extract_parser = subparsers.add_parser(
        "desi-extract",
        help="extract aligned DESI B-, R-, or Z-arm spectra from FITS files",
    )
    extract_parser.add_argument(
        "--input-glob",
        required=True,
        help="quoted glob for downloaded DESI FITS files",
    )
    extract_parser.add_argument("--output", required=True, type=Path)
    extract_parser.add_argument(
        "--arm",
        choices=("B", "R", "Z"),
        default="R",
        help="DESI spectrograph arm to extract (default: R)",
    )

    desi_parser = subparsers.add_parser(
        "desi-run",
        help="run the real-DESI controlled S/N benchmark",
    )
    desi_parser.add_argument("--config", required=True, type=Path)
    desi_parser.add_argument(
        "--input",
        type=Path,
        help="override input_npz from the YAML file",
    )
    desi_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override output_dir from the YAML file",
    )
    reliability_parser = subparsers.add_parser(
        "desi-reliability-run",
        help="run calibration, OOD, bootstrap, and adaptation audits on DESI",
    )
    reliability_parser.add_argument("--config", required=True, type=Path)
    reliability_parser.add_argument("--input", type=Path)
    reliability_parser.add_argument("--output-dir", type=Path)

    cross_parser = subparsers.add_parser(
        "cross-survey-run",
        help="run a genuine source-survey to target-survey optical transfer",
    )
    cross_parser.add_argument("--config", required=True, type=Path)
    cross_parser.add_argument("--source", type=Path)
    cross_parser.add_argument("--target", type=Path)
    cross_parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_synthetic_demo_config(args.config)
        if args.output_dir is not None:
            config.output_dir = str(args.output_dir)
        artifacts = run_synthetic_demo(config)
        print(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        print(f"Saved {len(artifacts)} artifacts to {Path(config.output_dir).resolve()}")
        return 0
    if args.command == "desi-extract":
        files = [Path(path) for path in glob.glob(args.input_glob)]
        summary = extract_desi_arm_spectra(files, args.output, arm=args.arm)
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "desi-run":
        config = load_desi_snr_config(args.config)
        if args.input is not None:
            config.input_npz = str(args.input)
        if args.output_dir is not None:
            config.output_dir = str(args.output_dir)
        artifacts = run_desi_snr_experiment(config)
        print(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        print(f"Saved {len(artifacts)} artifacts to {Path(config.output_dir).resolve()}")
        return 0
    if args.command == "desi-reliability-run":
        config = load_desi_reliability_config(args.config)
        if args.input is not None:
            config.input_npz = str(args.input)
        if args.output_dir is not None:
            config.output_dir = str(args.output_dir)
        artifacts = run_desi_reliability_experiment(config)
        print(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        print(f"Saved {len(artifacts)} artifacts to {Path(config.output_dir).resolve()}")
        return 0
    if args.command == "cross-survey-run":
        config = load_cross_survey_config(args.config)
        if args.source is not None:
            config.source_npz = str(args.source)
        if args.target is not None:
            config.target_npz = str(args.target)
        if args.output_dir is not None:
            config.output_dir = str(args.output_dir)
        artifacts = run_cross_survey_experiment(config)
        print(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        print(f"Saved {len(artifacts)} artifacts to {Path(config.output_dir).resolve()}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
