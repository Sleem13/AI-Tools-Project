from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..utils.filesystem import should_ignore


@dataclass
class TreeNode:
    name: str
    path: Path
    is_dir: bool
    size: int = 0
    children: list[TreeNode] = field(default_factory=list)


class DirectoryTreeBuilder:
    def __init__(
        self,
        ignore_hidden: bool = True,
        ignore_cache: bool = True,
    ) -> None:
        self.ignore_hidden = ignore_hidden
        self.ignore_cache = ignore_cache

    def build(self, root: Path) -> TreeNode:
        root_node = TreeNode(
            name=root.name,
            path=root,
            is_dir=True,
        )

        if not root.is_dir():
            root_node.is_dir = False
            root_node.size = root.stat().st_size if root.exists() else 0
            return root_node

        total_size = 0
        try:
            entries = sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return root_node

        for entry in entries:
            if should_ignore(entry, self.ignore_hidden, self.ignore_cache):
                continue

            if entry.is_dir():
                child = self.build(entry)
            else:
                child = TreeNode(
                    name=entry.name,
                    path=entry,
                    is_dir=False,
                    size=entry.stat().st_size,
                )

            root_node.children.append(child)
            total_size += child.size

        root_node.size = total_size
        return root_node

    @staticmethod
    def render(tree: TreeNode, indent: str = "", is_last: bool = True) -> str:
        if indent == "":
            lines = [f"{tree.name}/"]
        else:
            prefix = "└── " if is_last else "├── "
            size_str = DirectoryTreeBuilder._format_size(tree.size)
            if tree.is_dir:
                lines = [f"{indent}{prefix}{tree.name}/  ({size_str})"]
            else:
                lines = [f"{indent}{prefix}{tree.name}  ({size_str})"]

        child_indent = indent + ("    " if is_last else "│   ")
        sorted_children = sorted(tree.children, key=lambda c: (not c.is_dir, c.name.lower()))
        for i, child in enumerate(sorted_children):
            child_is_last = i == len(sorted_children) - 1
            child_lines = DirectoryTreeBuilder.render(child, child_indent, child_is_last)
            lines.append(child_lines)

        return "\n".join(lines)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        size: float = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
