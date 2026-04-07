from __future__ import annotations

from frontend.ast.nodes import (
    AddOpNode,
    ClassDeclNode,
    FloatNumNode,
    FuncDefNode,
    IntNumNode,
    MinusNode,
    MultOpNode,
    NotNode,
    PlusNode,
    ProgNode,
    ProgramBlockNode,
    RelOpNode,
    StartNode,
)
from frontend.semantics.symbols import Diagnostic, SymbolEntry, SymbolTable
from frontend.semantics.visitors.visitor import Visitor


class ComputeMemSizeVisitor(Visitor):
    INT_SIZE = 4
    FLOAT_SIZE = 8
    ADDRESS_SIZE = 4
    RETURN_ADDRESS_SLOT = "retaddr"
    RETURN_VALUE_SLOT = "retval"
    COPIED_EXPRESSION_BOUNDARIES = {"AParamsNode", "IndexNode", "ReturnNode", "WriteNode"}
    NON_COPIED_BOUNDARIES = {"ArraySizeNode", "IfNode", "WhileNode"}
    OPERATOR_NODES = {"AddOpNode", "MinusNode", "MultOpNode", "NotNode", "PlusNode", "RelOpNode"}

    def __init__(self, global_table: SymbolTable | None = None) -> None:
        self.global_table = global_table
        self.current_scope: SymbolTable | None = None
        self.diagnostics: list[Diagnostic] = []
        self._computed_class_tables: set[int] = set()
        self._class_stack: set[int] = set()
        self._computed_function_tables: set[int] = set()
        self._finalized_function_tables: set[int] = set()

    def generic_visit(self, node):
        for child in node.iter_children():
            child.accept(self)
        return getattr(node, "symtab", None)

    def visit_StartNode(self, node: StartNode):
        if node.first_child is not None:
            node.first_child.accept(self)
        return node.symtab

    def visit_ProgNode(self, node: ProgNode):
        if self.global_table is None:
            self.global_table = node.symtab
        if self.global_table is None:
            return None

        for entry in self.global_table.lookup(kinds={"class"}):
            self._compute_class_size(entry)
            class_table = entry.inner_scope_table
            if class_table is None:
                continue
            for member_entry in class_table.entries:
                if member_entry.kind == "member_function":
                    self._compute_function_size(member_entry)
        for entry in self.global_table.lookup(kinds={"function"}):
            self._compute_function_size(entry)
        for child in node.iter_children():
            child.accept(self)
        for entry in self.global_table.lookup(kinds={"class"}):
            class_table = entry.inner_scope_table
            if class_table is None:
                continue
            for member_entry in class_table.entries:
                if member_entry.kind == "member_function":
                    self._finalize_function_size(member_entry)
        for entry in self.global_table.lookup(kinds={"function"}):
            self._finalize_function_size(entry)
        return self.global_table

    def visit_ClassDeclNode(self, node: ClassDeclNode):
        if node.symtab_entry is not None:
            self._compute_class_size(node.symtab_entry)
        return node.symtab

    def visit_FuncDefNode(self, node: FuncDefNode):
        if node.symtab_entry is not None:
            self._compute_function_size(node.symtab_entry)
        self._visit_in_function_scope(node.symtab, node.func_body_node)
        if node.symtab_entry is not None:
            self._finalize_function_size(node.symtab_entry)
        return node.symtab

    def visit_ProgramBlockNode(self, node: ProgramBlockNode):
        if node.symtab_entry is not None:
            self._compute_function_size(node.symtab_entry)
        self._visit_in_function_scope(node.symtab, *node.iter_children())
        if node.symtab_entry is not None:
            self._finalize_function_size(node.symtab_entry)
        return node.symtab

    def visit_IntNumNode(self, node: IntNumNode):
        self._materialize_literal(node, "integer")
        return None

    def visit_FloatNumNode(self, node: FloatNumNode):
        self._materialize_literal(node, "float")
        return None

    def visit_MinusNode(self, node: MinusNode):
        self._materialize_temp(node)
        self.generic_visit(node)
        return None

    def visit_PlusNode(self, node: PlusNode):
        self._materialize_temp(node)
        self.generic_visit(node)
        return None

    def visit_NotNode(self, node: NotNode):
        self._materialize_temp(node)
        self.generic_visit(node)
        return None

    def visit_AddOpNode(self, node: AddOpNode):
        children = list(node.iter_children())
        if children:
            children[0].accept(self)
        self._materialize_temp(node)
        for child in children[1:]:
            child.accept(self)
        return None

    def visit_MultOpNode(self, node: MultOpNode):
        children = list(node.iter_children())
        if children:
            children[0].accept(self)
        self._materialize_temp(node)
        for child in children[1:]:
            child.accept(self)
        return None

    def visit_RelOpNode(self, node: RelOpNode):
        children = list(node.iter_children())
        if children:
            children[0].accept(self)
        self._materialize_temp(node)
        for child in children[1:]:
            child.accept(self)
        return None

    def _compute_class_size(self, class_entry: SymbolEntry) -> int:
        class_table = class_entry.inner_scope_table
        if class_table is None:
            return 0

        table_id = id(class_table)
        if table_id in self._computed_class_tables:
            return class_table.object_size
        if table_id in self._class_stack:
            self._diagnostic(
                "error",
                "cyclic_class_layout",
                f"cannot compute storage layout for cyclic class '{class_table.name}'.",
                class_entry.node,
            )
            return 0

        self._class_stack.add(table_id)

        next_offset = 0
        for inherited_table in class_table.inherited_class_tables:
            class_table.base_class_offsets[inherited_table.name] = next_offset
            next_offset += self._compute_class_size(
                self.global_table.lookup(inherited_table.name, {"class"})[0]
            )

        for entry in class_table.entries:
            if entry.kind != "data_member":
                continue
            entry.size = self._size_of_storage(entry)
            entry.offset = next_offset
            next_offset += entry.size

        class_table.object_size = next_offset
        class_entry.size = next_offset

        self._class_stack.remove(table_id)
        self._computed_class_tables.add(table_id)
        return next_offset

    def _compute_function_size(self, function_entry: SymbolEntry) -> int:
        function_table = function_entry.inner_scope_table
        if function_table is None:
            return 0

        table_id = id(function_table)
        if table_id in self._computed_function_tables:
            return function_table.stack_frame_size

        next_offset = 0
        layout_order = ("param", "local_var")

        for kind in layout_order:
            for entry in function_table.entries:
                if entry.kind != kind:
                    continue
                entry.size = self._size_of_storage(entry)
                next_offset -= entry.size
                entry.offset = next_offset

        function_table.stack_frame_size = -next_offset
        function_entry.size = function_table.stack_frame_size
        function_table.bind_frame_layout(self._size_of_type, next_offset)

        self._computed_function_tables.add(table_id)
        return function_table.stack_frame_size

    def _finalize_function_size(self, function_entry: SymbolEntry) -> int:
        function_table = function_entry.inner_scope_table
        if function_table is None:
            return 0

        table_id = id(function_table)
        if table_id in self._finalized_function_tables:
            return function_table.stack_frame_size

        self._ensure_return_slots(function_entry, function_table)

        for kind in ("return_value", "return_address"):
            for entry in function_table.entries:
                if entry.kind != kind or entry.offset is not None:
                    continue
                entry.size = self._size_of_storage(entry)
                function_table.frame_offset_cursor -= entry.size
                entry.offset = function_table.frame_offset_cursor

        function_table.stack_frame_size = -function_table.frame_offset_cursor
        function_entry.size = function_table.stack_frame_size
        self._finalized_function_tables.add(table_id)
        return function_table.stack_frame_size

    def _size_of_storage(self, entry: SymbolEntry) -> int:
        if entry.kind == "return_address":
            return self.ADDRESS_SIZE

        if entry.kind == "return_value":
            return self._size_of_type(entry.type, entry.array_dimensions)

        if entry.kind in {"param", "local_var", "data_member", "temp", "literal"}:
            return self._size_of_type(entry.type, entry.array_dimensions, entry.kind == "param")

        return entry.size

    def _size_of_type(
        self,
        type_name: str | None,
        array_dimensions: list[int | None],
        is_param: bool = False,
    ) -> int:
        base_size = self._base_type_size(type_name)
        if base_size == 0:
            return 0

        if not array_dimensions:
            return base_size

        if any(dimension is None for dimension in array_dimensions):
            if is_param:
                return self.ADDRESS_SIZE
            self._diagnostic(
                "error",
                "non_constant_array_size",
                "array storage requires compile-time constant dimensions.",
                None,
            )
            return 0

        total_elements = 1
        for dimension in array_dimensions:
            total_elements *= dimension
        return base_size * total_elements

    def _base_type_size(self, type_name: str | None) -> int:
        if type_name == "integer":
            return self.INT_SIZE
        if type_name == "float":
            return self.FLOAT_SIZE
        if type_name in {None, "void", "address"}:
            return 0 if type_name != "address" else self.ADDRESS_SIZE

        matches = self.global_table.lookup(type_name, {"class"})
        if not matches:
            self._diagnostic(
                "error",
                "unknown_storage_type",
                f"cannot compute storage for unknown type '{type_name}'.",
                None,
            )
            return 0
        return self._compute_class_size(matches[0])

    def _visit_in_function_scope(self, scope: SymbolTable | None, *children) -> SymbolTable | None:
        previous_scope = self.current_scope
        self.current_scope = scope
        for child in children:
            if child is not None:
                child.accept(self)
        self.current_scope = previous_scope
        return scope

    def _materialize_temp(self, node) -> None:
        if self.current_scope is None or not self._needs_temp_slot(node):
            return

        base_type, array_dimensions = self._split_type(node.inferred_type)
        if base_type is None:
            return

        self.current_scope.allocate_compiler_entry("temp", "temp", base_type, array_dimensions)

    def _materialize_literal(self, node, literal_type: str) -> None:
        if self.current_scope is None or not self._needs_literal_slot(node):
            return

        self.current_scope.allocate_compiler_entry("litval", "literal", literal_type)

    def _needs_temp_slot(self, node) -> bool:
        current = node.parent
        while current is not None:
            name = current.__class__.__name__
            if name in self.NON_COPIED_BOUNDARIES:
                return False
            if name in self.COPIED_EXPRESSION_BOUNDARIES:
                return True
            if name in self.OPERATOR_NODES:
                return True
            if name == "StatementNode":
                return False
            current = current.parent
        return False

    def _needs_literal_slot(self, node) -> bool:
        current = node.parent
        while current is not None:
            name = current.__class__.__name__
            if name in self.NON_COPIED_BOUNDARIES:
                return False
            if name in self.COPIED_EXPRESSION_BOUNDARIES:
                return True
            if name in self.OPERATOR_NODES:
                return True
            if name == "StatementNode":
                return False
            current = current.parent
        return False

    def _ensure_return_slots(
        self,
        function_entry: SymbolEntry,
        function_table: SymbolTable,
    ) -> None:
        if function_entry.type not in {None, "void"} and not any(
            entry.kind == "return_value" for entry in function_table.entries
        ):
            function_table.entries.append(
                SymbolEntry(
                    name=self.RETURN_VALUE_SLOT,
                    kind="return_value",
                    type=function_entry.type,
                ),
            )

        if not any(entry.kind == "return_address" for entry in function_table.entries):
            function_table.entries.append(
                SymbolEntry(
                    name=self.RETURN_ADDRESS_SLOT,
                    kind="return_address",
                    type="address",
                ),
            )

    def _split_type(self, type_name: str | None) -> tuple[str | None, list[int | None]]:
        if type_name is None:
            return None, []

        base_type = type_name.split("[", 1)[0]
        dimensions: list[int | None] = []
        index = type_name.find("[")
        while index != -1:
            end = type_name.find("]", index)
            dimension = type_name[index + 1:end]
            dimensions.append(None if dimension == "" else int(dimension))
            index = type_name.find("[", end)
        return base_type, dimensions

    def _diagnostic(self, severity: str, code: str, message: str, node) -> None:
        line = getattr(getattr(node, "token", None), "line", 0)
        self.diagnostics.append(
            Diagnostic(severity=severity, code=code, message=message, line=line)
        )
