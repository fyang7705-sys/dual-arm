"""Tiny YAML reader for the flat configs in this starter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_simple_yaml(text)


def require(config: dict[str, Any], key: str) -> Any:
    """Return a config value. Missing keys are an error, not a second default."""

    if key not in config:
        raise KeyError(f"Config is missing {key}.")
    return config[key]


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the subset of YAML used by configs/*.yaml.

    Supports nested maps, lists of scalars, bools, ints, floats, and strings.
    """

    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))

    root: dict[str, Any] = {}

    def parse_block(index: int, indent: int) -> tuple[dict[str, Any], int]:
        node: dict[str, Any] = {}
        while index < len(lines):
            level, content = lines[index]
            if level < indent:
                break
            if level > indent:
                raise ValueError(f"Unexpected indent in config: {content}")
            if content.startswith("- "):
                raise ValueError(f"List item is not inside a key: {content}")
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            index += 1
            if rest == "" or rest == "|" or rest == ">":
                if index < len(lines) and lines[index][1].startswith("- "):
                    values, index = parse_list(index, level + 2)
                    node[key] = values
                else:
                    child, index = parse_block(index, level + 2)
                    node[key] = child
            else:
                node[key] = _scalar(_strip_inline_comment(rest))
        return node, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        values: list[Any] = []
        while index < len(lines):
            level, content = lines[index]
            if level < indent or not content.startswith("- "):
                break
            values.append(_scalar(_strip_inline_comment(content[2:].strip())))
            index += 1
        return values, index

    root, end = parse_block(0, 0)
    if end != len(lines):
        raise ValueError("Failed to parse the whole config.")
    return root


def _strip_inline_comment(token: str) -> str:
    """Drop an unquoted ``#`` comment. ``100  ## 600`` is the integer 100."""

    in_single = False
    in_double = False
    for index, char in enumerate(token):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and (index == 0 or token[index - 1].isspace()):
            return token[:index].strip()
    return token.strip()


def _scalar(token: str) -> Any:
    if token in {"null", "Null", "~"}:
        return None
    if token in {"true", "True"}:
        return True
    if token in {"false", "False"}:
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        if any(mark in token for mark in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token
