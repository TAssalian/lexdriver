from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from frontend.semantics.symbols.symbol_entry import SymbolEntry


@dataclass
class SymbolTable:
    name: str
    kind: str
    parent_scope: SymbolTable | None = None # To walk back up to parent scope to know whether function belongs to a class or if local var shadows class data member
    owner_entry: SymbolEntry | None = None
    entries: list[SymbolEntry] = field(default_factory=list)
    inherited_class_tables: list[SymbolTable] = field(default_factory=list)
    stack_frame_size: int = 0
    object_size: int = 0
    base_class_offsets: dict[str, int] = field(default_factory=dict) # offsets of a class' inherited class' in order to access its base class' members
    frame_offset_cursor: int = 0
    storage_size_resolver: Callable[[str | None, list[int | None], bool], int] | None = field(
        default=None,
        repr=False,
    )

    # Return matching entries in this symbol table and any inherited class tables.
    def lookup(self, 
        name: str | None = None, 
        kinds: set[str] | None = None, 
        visited: set[int] | None = None,
    ) -> list[SymbolEntry]:
        
        # Prevent infinite recursion if we have circular inheritance    
        if visited is None:
            visited = set()
        table_id = id(self)
        if table_id in visited:
            return []
        visited.add(table_id)

        matches = []
        for entry in self.entries:
            name_matches = name is None or entry.name == name
            kind_matches = kinds is None or entry.kind in kinds
            if name_matches and kind_matches:
                matches.append(entry)
        for symbol_table in self.inherited_class_tables:
            matches.extend(symbol_table.lookup(name, kinds=kinds, visited=visited))
        return matches

    def bind_frame_layout(
        self,
        storage_size_resolver: Callable[[str | None, list[int | None], bool], int],
        next_offset: int,
    ) -> None:
        self.storage_size_resolver = storage_size_resolver
        self.frame_offset_cursor = next_offset

    def allocate_compiler_entry(
        self,
        name: str,
        kind: str,
        entry_type: str,
        array_dimensions: list[int | None] | None = None,
    ) -> SymbolEntry:
        entry = SymbolEntry(
            name=name,
            kind=kind,
            type=entry_type,
            array_dimensions=list(array_dimensions or []),
        )
        entry.size = self.storage_size_resolver(entry.type, entry.array_dimensions, False)
        self.frame_offset_cursor -= entry.size
        entry.offset = self.frame_offset_cursor
        self.entries.append(entry)
        self.stack_frame_size = -self.frame_offset_cursor
        if self.owner_entry is not None:
            self.owner_entry.size = self.stack_frame_size
        return entry
