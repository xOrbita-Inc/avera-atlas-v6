from __future__ import annotations

import argparse
from pathlib import Path

from .canonical_scenarios import run_all_canonical_scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sandbox canonical regression scenarios."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic scenario seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )

    args = parser.parse_args()

    report = run_all_canonical_scenarios(seed=args.seed)
    report_json = report.to_json()

    print(report_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            report_json + "\n",
            encoding="utf-8",
        )

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())