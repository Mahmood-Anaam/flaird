"""Dependency-free Markdown structure and repository-local link checks."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "README.md",
    ROOT / "src/flaird/modeling/README.md",
    ROOT / "demo/README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]


def test_local_markdown_links_resolve():
    for path in FILES:
        content = path.read_text()
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            resolved = path.parent / unquote(parsed.path)
            assert resolved.exists(), f"{path.relative_to(ROOT)}: missing {target}"


def test_fences_and_display_math_are_closed():
    for path in FILES:
        content = path.read_text()
        assert len(re.findall(r"^```", content, re.M)) % 2 == 0, str(path)
        # Strip fenced code so command examples are not treated as math.
        prose = re.sub(r"```.*?```", "", content, flags=re.S)
        assert len(re.findall(r"^\$\$\s*$", prose, re.M)) % 2 == 0, str(path)
        assert not re.search(r"^flowchart TD\s*$", content, re.M) or " --> " in content


def test_every_notebook_cell_is_audited():
    import json

    content = (ROOT / "docs/NOTEBOOK_EXECUTION_AUDIT.md").read_text()
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        section = content.split(f"## {path.name}\n", 1)[1].split("\n## ", 1)[0]
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            assert f'| {index} | {cell["cell_type"]} |' in section
