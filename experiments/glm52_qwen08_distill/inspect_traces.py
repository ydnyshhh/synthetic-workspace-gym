from __future__ import annotations

import sys

from build_dataset import main


if __name__ == "__main__":
    args = ["--dry-run", "--write-raw", "--write-sequential", *sys.argv[1:]]
    raise SystemExit(main(args))
