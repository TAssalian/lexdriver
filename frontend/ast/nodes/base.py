from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from frontend.lexer.tokens import Token


@dataclass
class Node:
    token: Token
    parent: Node | None = None
    first_child: Node | None = None
    last_child: Node | None = None
    prev_sibling: Node | None = None
    next_sibling: Node | None = None

    def add_child(self, child: Node) -> None:
        child.parent = self
        child.prev_sibling = self.last_child
        child.next_sibling = None

        if self.last_child is None:
            self.first_child = child
        else:
            self.last_child.next_sibling = child

        self.last_child = child

    def iter_children(self) -> Iterator[Node]:
        current = self.first_child
        while current is not None:
            yield current
            current = current.next_sibling
