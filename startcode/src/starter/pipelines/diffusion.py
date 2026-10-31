"""Offline and online entry points for the diffusion pipeline."""

from __future__ import annotations

from starter.cli import main as run


def main(argv: list[str] | None = None) -> None:
    run("diffusion", argv)


if __name__ == "__main__":
    main()
