from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ts_rsi.types import ToolExecutionResult


class CandidateEditor:
    """Deterministic editor restricted to a candidate directory."""

    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root.resolve()
        self.candidate_root.mkdir(parents=True, exist_ok=True)

    def clone_from(self, source_root: Path) -> None:
        source = source_root.resolve()
        if self.candidate_root.exists():
            shutil.rmtree(self.candidate_root)
        shutil.copytree(source, self.candidate_root)

    def apply(self, tool_name: str, tool_input: dict[str, Any]) -> ToolExecutionResult:
        path = self._safe_path(str(tool_input["path"]))
        if tool_name == "str_replace":
            old = str(tool_input["old_str"])
            new = str(tool_input["new_str"])
            text = path.read_text(encoding="utf-8")
            occurrences = text.count(old)
            if occurrences != 1:
                return ToolExecutionResult(tool_name, False, f"Expected one match, found {occurrences}")
            path.write_text(text.replace(old, new), encoding="utf-8")
            return ToolExecutionResult(tool_name, True, "Replacement applied", (str(path),))
        if tool_name == "write_file":
            content = str(tool_input["content"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolExecutionResult(tool_name, True, "File written", (str(path),))
        if tool_name == "set_config":
            key = str(tool_input["key"])
            value = str(tool_input["value"])
            text = path.read_text(encoding="utf-8")
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                document = None
            if isinstance(document, dict):
                try:
                    document[key] = json.loads(value)
                except json.JSONDecodeError:
                    document[key] = value
                path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                return ToolExecutionResult(tool_name, True, "Config updated", (str(path),))
            prefix = f"{key}:"
            lines = text.splitlines(keepends=True)
            matches = [idx for idx, line in enumerate(lines) if line.startswith(prefix)]
            if len(matches) != 1:
                return ToolExecutionResult(tool_name, False, f"Expected one config key, found {len(matches)}")
            index = matches[0]
            ending = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"{key}: {value}{ending}"
            path.write_text("".join(lines), encoding="utf-8")
            return ToolExecutionResult(tool_name, True, "Config updated", (str(path),))
        return ToolExecutionResult(tool_name, False, f"Unsupported tool: {tool_name}")

    def _safe_path(self, relative_path: str) -> Path:
        path = (self.candidate_root / relative_path).resolve()
        if path != self.candidate_root and self.candidate_root not in path.parents:
            raise ValueError("Editor path escapes candidate root")
        return path
