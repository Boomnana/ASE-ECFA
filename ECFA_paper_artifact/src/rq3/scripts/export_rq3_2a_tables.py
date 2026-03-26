from __future__ import annotations

import sys
from pathlib import Path


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from rq3fresh.reporting import export_main_tables


def main() -> None:
    ap = argparse.ArgumentParser(description='Export RQ3-2A tables from an existing run directory.')
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()
    paths = export_main_tables(args.run_dir)
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()
