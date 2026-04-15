from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from frontend.lexer.tokens import Token

if TYPE_CHECKING:
    from frontend.semantics.symbols import SymbolEntry, SymbolTable


@dataclass
class Node:
    token: Token
    parent: Node | None = None
    first_child: Node | None = None
    last_child: Node | None = None
    prev_sibling: Node | None = None
    next_sibling: Node | None = None
    symtab: SymbolTable | None = None # Allow visitors to re-enter scope without re-deriving it from parents
    symtab_entry: SymbolEntry | None = None # Know which symbol entry corresponds to this node for quick semantic checking
    

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

    def accept(self, visitor: Any) -> Any:
        method_name = f"visit_{self.__class__.__name__}" # Get's method name dynamically
        visit_func = getattr(visitor, method_name, None) # Check whether specific visitor subclass has that method or not.
        if visit_func is None:
            return visitor.visit_children(self)
        return visit_func(self)
