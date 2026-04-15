from __future__ import annotations

from frontend.ast.nodes import (
    AddOpNode,
    ClassDeclNode,
    FloatNumNode,
    FParamNode,
    FuncBodyNode,
    FuncDefNode,
    IdNode,
    IndexNode,
    IntNumNode,
    MinusNode,
    MultOpNode,
    NotNode,
    PlusNode,
    ProgNode,
    ProgramBlockNode,
    RelOpNode,
    StartNode,
    VarDeclNode,
    VariableNode,
)
from frontend.semantics.symbols import SymbolEntry, SymbolTable
from frontend.semantics.visitors.visitor import Visitor


class ComputeMemSizeVisitor(Visitor):
    INTEGER_SIZE = 4
    FLOAT_SIZE = 8
    VOID_SIZE = 0
    ADDRESS_SIZE = 4 # because of 32-bit memory model

    def __init__(self) -> None:
        self.global_table: SymbolTable | None = None
        self.current_scope: SymbolTable | None = None
        self.current_offset = 0
        self.literal_counter = 0
        self.temp_counter = 0
        self._computed_class_tables: set[int] = set() # classes we already have the size of
        self._active_class_tables: set[int] = set() # classes currently being computed in the call stack for inherited class members

    def visit_ProgNode(self, node: ProgNode):
        self.global_table = node.symtab
        self.current_scope = self.global_table
        self.visit_children(node)

    def visit_ClassDeclNode(self, node: ClassDeclNode):
        class_table = node.symtab
        table_id = id(class_table)
        if table_id in self._computed_class_tables:
            return None

        self._active_class_tables.add(table_id)
        previous_scope = self.current_scope

        self.current_scope = class_table
        self.current_offset = 0

        for inherits_node in node.inherits:
            parent_entry = self._lookup_class(inherits_node.id_node.token.lexeme)
            parent_table = parent_entry.inner_scope_table
            parent_table_id = id(parent_table)
            if parent_table_id not in self._computed_class_tables and parent_table_id not in self._active_class_tables:
                parent_entry.node.accept(self)
            self.current_offset += parent_table.size

        for member in node.members:
            if isinstance(member, VarDeclNode):
                member.accept(self)

        class_table.size = self.current_offset
        self._active_class_tables.remove(table_id)
        self._computed_class_tables.add(table_id)
        self.current_scope = previous_scope

    def visit_FuncDefNode(self, node: FuncDefNode):
        function_table = node.symtab
        previous_scope = self.current_scope

        self.current_scope = function_table
        self.current_offset = 0
        self.literal_counter = 0
        self.temp_counter = 0

        self._compute_ret_val_size(node.symtab_entry.type)
        self._compute_ret_addr_size()
        node.fparams_node.accept(self)
        node.func_body_node.accept(self)
        function_table.size = self.current_offset

        self.current_scope = previous_scope

    def visit_ProgramBlockNode(self, node: ProgramBlockNode):
        function_table = node.symtab
        previous_scope = self.current_scope

        self.current_scope = function_table
        self.current_offset = 0
        self.literal_counter = 0
        self.temp_counter = 0

        self._compute_ret_val_size(node.symtab_entry.type)
        self._compute_ret_addr_size()
        self.visit_children(node)
        function_table.size = self.current_offset

        self.current_scope = previous_scope

    def visit_FParamNode(self, node: FParamNode):
        entry = node.symtab_entry
        if entry.array_dimensions:
            entry.size = self.ADDRESS_SIZE
        else:
            entry.size = self._get_slot_size_of_entry(entry.type, entry.array_dimensions)
        entry.offset = self.current_offset
        self.current_offset += entry.size

    def visit_VarDeclNode(self, node: VarDeclNode):
        entry = node.symtab_entry
        entry.size = self._get_slot_size_of_entry(entry.type, entry.array_dimensions)
        entry.offset = self.current_offset
        self.current_offset += entry.size

    def visit_IntNumNode(self, node: IntNumNode):
        node.symtab_entry = self._make_synthetic_entry(
            name=f"__lit{self._next_literal_id()}",
            kind="literal",
            type_name=node.inferred_type,
            size=self._get_slot_size_of_entry(node.inferred_type, []),
            owner_class=None,
            node=node,
        )

    visit_FloatNumNode = visit_IntNumNode

    def visit_PlusNode(self, node: PlusNode):
        self.visit_children(node)
        self._compute_temp_for_node(node)

    visit_MinusNode = visit_PlusNode
    visit_NotNode = visit_PlusNode
    visit_AddOpNode = visit_PlusNode
    visit_MultOpNode = visit_PlusNode
    visit_RelOpNode = visit_PlusNode

    def _compute_ret_val_size(self, return_type: str | None) -> None:
        return_value_size = self._get_slot_size_of_entry(return_type, [])
        self._make_synthetic_entry(
            name="__ret_val",
            kind="return_value",
            type_name=return_type,
            size=return_value_size,
            owner_class=None,
            insertion_index=0,
        )

    def _compute_ret_addr_size(self) -> None:
        self._make_synthetic_entry(
            name="__ret_addr",
            kind="return_address",
            type_name="address",
            size=self.ADDRESS_SIZE,
            owner_class=None,
            insertion_index=1,
        )

    def _compute_temp_for_node(self, node) -> None:
        node.symtab_entry = self._make_synthetic_entry(
            name=f"__tmp{self._next_temp_id()}",
            kind="temp",
            type_name=node.inferred_type,
            size=self._get_slot_size_of_entry(node.inferred_type, []),
            owner_class=None,
            node=node,
        )

    def _make_synthetic_entry(
        self,
        name: str,
        kind: str,
        type_name: str | None,
        size: int,
        owner_class: str | None,
        node=None,
        insertion_index: int | None = None,
    ) -> SymbolEntry:
        
        entry = SymbolEntry(
            name=name,
            kind=kind,
            type=type_name,
            owner_class=owner_class,
            node=node,
            size=size,
            offset=self.current_offset,
        )
        
        self.current_offset += size
        if insertion_index is None:
            self.current_scope.entries.append(entry)
        else:
            self.current_scope.entries.insert(insertion_index, entry)
        return entry

    def _get_slot_size_of_entry(self, base_type: str | None, array_dimensions: list[int | None]) -> int:
        if base_type == "integer":
            size = self.INTEGER_SIZE
        elif base_type == "float":
            size = self.FLOAT_SIZE
        elif base_type == "void":
            size = self.VOID_SIZE
        else:
            class_entry = self._lookup_class(base_type)
            class_table = class_entry.inner_scope_table
            class_table_id = id(class_table)
            if class_table_id not in self._computed_class_tables and class_table_id not in self._active_class_tables:
                class_entry.node.accept(self)
            size = class_table.size

        total_size = size
        for dimension in array_dimensions:
            if dimension is None:
                continue
            total_size *= dimension
        return total_size

    def _lookup_class(self, class_name: str | None) -> SymbolEntry | None:
        return self.global_table.lookup(class_name, {"class"})[0]

    def _next_literal_id(self) -> int:
        self.literal_counter += 1
        return self.literal_counter

    def _next_temp_id(self) -> int:
        self.temp_counter += 1
        return self.temp_counter
