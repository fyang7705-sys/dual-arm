"""Offline and online entry points for the ACT pipeline."""

from __future__ import annotations

from starter.cli import main as run


def main(argv: list[str] | None = None) -> None:
    run("act", argv)


if __name__ == "__main__":
    main()
