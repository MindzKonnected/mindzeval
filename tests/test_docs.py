"""Keep the examples in the docs honest: they must load with the real loader."""

from __future__ import annotations

import re
from pathlib import Path

from mindzeval import load_config

DOCS = Path(__file__).resolve().parents[1] / "docs" / "reference"


def _first_yaml_block(text: str, after: str) -> str:
    section = text[text.index(after) :]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match, f"no yaml block after {after!r}"
    return match.group(1)


def test_full_config_example_loads(tmp_path: Path) -> None:
    text = (DOCS / "config.md").read_text(encoding="utf-8")
    path = tmp_path / "mindzeval.yaml"
    path.write_text(_first_yaml_block(text, "## Full example"), encoding="utf-8")
    env = {"AGENT_TOKEN": "t", "DB_PASSWORD": "p", "OPENAI_API_KEY": "k"}
    config = load_config(path, environ=env)
    assert len(config.matrix_combinations) == 4
    assert config.safety.pii_columns == ["email", "customers.phone"]
