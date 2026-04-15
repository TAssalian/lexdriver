from frontend.semantics.symbols import Diagnostic, SymbolEntry, SymbolTable
from frontend.semantics.visitors.visitor import Visitor
from frontend.ast.nodes import (
    AParamsNode,
    AddOpNode,
    AssignOpNode,
    ClassDeclNode,
    FParamNode,
    FParamsNode,
    FloatNumNode,
    FuncDeclNode,
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
    ReturnNode,
    StartNode,
    StatementNode,
    VarDeclNode,
    VariableNode,
)


class SemanticCheckingVisitor(Visitor):
    def __init__(self) -> None:
        self.global_table: SymbolTable | None = None
        self.current_scope: SymbolTable | None = None
        self.current_function: SymbolEntry | None = None
        self.current_class: SymbolTable | None = None # SymbolTable of the class whose member function body we are currently inside
        self.diagnostics: list[Diagnostic] = []

    def visit_ProgNode(self, node: ProgNode):
        self.global_table = node.symtab
        self.current_scope = self.global_table
        self._check_circular_classes() # Reject inheritance and/or member-type cycles before normal traversal
        self.visit_children(node)
        return None

    def visit_ClassDeclNode(self, node: ClassDeclNode):
        # Check for undeclared inherited class
        for inherits_node in node.inherits:
            parent_name = inherits_node.id_node.token.lexeme
            if self._lookup_class(parent_name) is None:
                self._diagnostic(
                    "error",
                    "undeclared_class",
                    f"undeclared class '{parent_name}'.",
                    inherits_node,
                )

        class_table = node.symtab
        previous_scope = self.current_scope
        self.current_scope = class_table
        for member in node.members:
            member.accept(self)
        self.current_scope = previous_scope
        return None

    # Check the declared return type and visit parameter declarations inside the function's scope, but without a function body because this is only a declaration
    def visit_FuncDeclNode(self, node: FuncDeclNode):
        self._check_type_exists(node.return_type_node.token.lexeme, node.return_type_node)
        
        function_table = node.symtab
        previous_scope = self.current_scope
        self.current_scope = function_table
        node.fparams_node.accept(self)
        self.current_scope = previous_scope
        return None

    # Enter a function definition, establish its local/function/class context, then validate parameters and the body before restoring the previous traversal state
    def visit_FuncDefNode(self, node: FuncDefNode):
        self._check_type_exists(type_name=node.return_type_node.token.lexeme, node=node.return_type_node)
        function_table = node.symtab
        
        previous_scope = self.current_scope
        previous_function = self.current_function
        previous_class = self.current_class

        self.current_scope = function_table
        self.current_function = node.symtab_entry # Needed for return-statement type checking
        
        if function_table and function_table.parent_scope and function_table.parent_scope.kind == "class":
            self.current_class = function_table.parent_scope # Member function definitions also need their owning class context for inherited members
        else:
            self.current_class = None # Free functions should not resolve data members implicitly

        node.fparams_node.accept(self)
        node.func_body_node.accept(self)

        self.current_scope = previous_scope
        self.current_function = previous_function
        self.current_class = previous_class
        return None

    def visit_ProgramBlockNode(self, node: ProgramBlockNode):
        previous_scope = self.current_scope
        previous_function = self.current_function
        previous_class = self.current_class
        self.current_scope = node.symtab
        self.current_function = node.symtab_entry
        self.current_class = None
        for child in node.iter_children():
            child.accept(self)
        self.current_scope = previous_scope
        self.current_function = previous_function
        self.current_class = previous_class
        return None

    def visit_FParamNode(self, node: FParamNode):
        self._check_type_exists(node.type_node.token.lexeme, node.type_node)
        return None

    def visit_VarDeclNode(self, node: VarDeclNode):
        self._check_type_exists(node.type_node.token.lexeme, node.type_node)
        return None

    # Distinguish between expression statements and assignment statements, then resolve the left/right sides and enforce assignment compatibility
    # Expression statement: A call without '=' character
    # Assignment statement: Something with '=' character
    def visit_StatementNode(self, node: StatementNode):
        children = list(node.iter_children())
        assign_index = None
        for index, child in enumerate(children):
            if isinstance(child, AssignOpNode):
                assign_index = index
                break

        if assign_index is None: # Not assignment statement
            self._resolve_chain(children, node)
            return None

        left_type = self._resolve_chain(children[:assign_index], node) # Is an id-started reference chain stored as multiple children of StatementNode
        right_type = children[assign_index + 1].accept(self) # Is a single EXPR node following the AssignOpNode
        if left_type is not None and right_type is not None and not self._is_assignable(left_type, right_type):
            self._diagnostic(
                "error",
                "type_error_assignment_statement",
                f"type error in assignment statement: cannot assign '{right_type}' to '{left_type}'.",
                children[assign_index],
            )
        return None

    # Compare returned type to the enclosing function's declared type
    def visit_ReturnNode(self, node: ReturnNode):
        expr_type = node.first_child.accept(self)
        if self.current_function is None:
            return None
        expected_type = self.current_function.type
        if expr_type is not None and not self._is_assignable(expected_type, expr_type):
            self._diagnostic(
                "error",
                "type_error_return_statement",
                f"type error in return statement: expected '{expected_type}' but got '{expr_type}'.",
                node,
            )
        return None

    # Appears on lhs
    def visit_VariableNode(self, node: VariableNode):
        node.inferred_type = self._resolve_chain(list(node.iter_children()), node)
        return node.inferred_type

    # Appears on rhs
    def visit_IdNode(self, node: IdNode):
        node.inferred_type = self._resolve_chain([node, *node.iter_children()], node)
        return node.inferred_type

    def visit_IntNumNode(self, node: IntNumNode):
        node.inferred_type = "integer"
        return node.inferred_type

    def visit_FloatNumNode(self, node: FloatNumNode):
        node.inferred_type = "float"
        return node.inferred_type

    # Visit all actual arguments and store their types for overload matching during function-call resolution
    def visit_AParamsNode(self, node: AParamsNode):
        types = []
        for arg in node.args:
            types.append(arg.accept(self))
        return types

    def visit_MinusNode(self, node: MinusNode):
        return self._get_unary_type(node, allowed_types={"integer", "float"})

    def visit_PlusNode(self, node: PlusNode):
        return self._get_unary_type(node, allowed_types={"integer", "float"})

    def visit_NotNode(self, node: NotNode):
        return self._get_unary_type(node, allowed_types={"integer"})

    def visit_AddOpNode(self, node: AddOpNode):
        operator = node.token.lexeme
        if operator == "or":
            return self._get_binary_type(node, allowed_types={"integer"}, result_type="integer")
        return self._get_binary_type(node, allowed_types={"integer", "float"})
    
    def visit_MultOpNode(self, node: MultOpNode):
        operator = node.token.lexeme
        if operator == "and":
            return self._get_binary_type(node, allowed_types={"integer"}, result_type="integer")
        return self._get_binary_type(node, allowed_types={"integer", "float"})

    def visit_RelOpNode(self, node: RelOpNode):
        left_type = node.first_child.accept(self)
        right_type = node.first_child.next_sibling.accept(self)
        if left_type is None or right_type is None:
            node.inferred_type = None
            return None
        if not self._is_numeric_operand(left_type) or not self._is_numeric_operand(right_type) or left_type != right_type:
            self._diagnostic(
                "error",
                "type_error_expression",
                f"type error in expression: incompatible operands '{left_type}' and '{right_type}'.",
                node,
            )
            node.inferred_type = None
            return None
        node.inferred_type = "integer"
        return node.inferred_type

    def _get_unary_type(self, node, allowed_types: set[str]):
        operand_type = node.first_child.accept(self)
        if operand_type is None:
            node.inferred_type = None
            return None
        if operand_type not in allowed_types:
            self._diagnostic(
                "error",
                "type_error_expression",
                f"type error in expression: invalid operand type '{operand_type}'.",
                node,
            )
            node.inferred_type = None
            return None
        node.inferred_type = operand_type
        return node.inferred_type

    def _get_binary_type(self, node, allowed_types: set[str], result_type: str | None = None):
        left = node.first_child.accept(self)
        right = node.first_child.next_sibling.accept(self)
        if left is None or right is None:
            node.inferred_type = None
            return None

        if left != right or left not in allowed_types:
            self._diagnostic(
                "error",
                "type_error_expression",
                f"type error in expression: incompatible operands '{left}' and '{right}'.",
                node,
            )
            node.inferred_type = None
            return None
        node.inferred_type = result_type or left
        return node.inferred_type

    # Get type of a chained reference specifically for id/member/function call chains. Arithmetic expressions are by the OpNodes
    def _resolve_chain(self, children, owner_node) -> str | None:
        current_type = None # Holds the type of the resolved chain 
        current_entry = None # Type of previously resolved segment to check type compatibility later
        is_call = False # Remembers if most recent segment was a function call
        index = 0 # To go through the list of children

        while index < len(children):
            id_node = children[index]
            name = id_node.token.lexeme
            index += 1

            indices = []
            while index < len(children) and isinstance(children[index], IndexNode):
                indices.append(children[index]) 
                index += 1

            call_node = None
            if index < len(children) and isinstance(children[index], AParamsNode):
                call_node = children[index] 
                index += 1

            if current_type is None:
                if call_node is not None:
                    current_entry = self._resolve_initial_call(name, call_node, id_node)
                    is_call = True
                else: 
                    current_entry = self._resolve_identifier(name, id_node)
                    is_call = False
            
            else:
                if call_node is not None: 
                    current_entry = self._resolve_member_call(current_type, name, call_node, id_node) 
                    is_call = True
                else: 
                    current_entry = self._resolve_member_variable(current_type, name, id_node)
                    is_call = False

            if current_entry is None:
                setattr(owner_node, "inferred_type", None)
                return None

            id_node.symtab_entry = current_entry
            id_node.inferred_type = self._entry_type(current_entry) 
            current_type = self._entry_type(current_entry)
            if not is_call:
                current_type = self._remove_indices(current_entry, indices, id_node)
                id_node.inferred_type = current_type

            if call_node is not None:
                call_node.inferred_type = current_type

        if current_entry is not None and self._get_str_dimensions_list(current_type) and not self._allows_array_reference(owner_node):
            self._diagnostic(
                "error",
                "wrong_array_dimensionality",
                f"wrong array dimensionality for '{current_entry.name}'.",
                children[0],
            )
            setattr(owner_node, "inferred_type", None)
            return None

        setattr(owner_node, "inferred_type", current_type) # Publish the final type on the owning AST node
        if current_entry is not None and not isinstance(owner_node, IdNode):
            owner_node.symtab_entry = current_entry
        return current_type

    # Resolve entry of first identifier in a chained expression. Check in the current local/function scope first, then in the current class scope bc of data member inheritance
    def _resolve_identifier(self, name: str, node: IdNode) -> SymbolEntry | None:
        entry = None
        if self.current_scope is not None:
            locals_found = self.current_scope.lookup(name, {"param", "local_var"}) 
            if locals_found:
                entry = locals_found[0]
        if entry is None and self.current_class is not None:
            members = self.current_class.lookup(name, {"data_member"})
            if members:
                entry = members[0]
        if entry is None:
            self._diagnostic(
                "error",
                "undeclared_local_variable",
                f"undeclared local variable '{name}'.",
                node,
            )
        return entry

    # Resolve the first callable element in a chain, checking the class' member functions and then checking for free functions in the global scope
    def _resolve_initial_call(self, name: str, call_node: AParamsNode, node: IdNode) -> SymbolEntry | None:
        argument_types = call_node.accept(self)
        if self.current_class is not None:
            member_candidates = self.current_class.lookup(name, {"member_function"})
            if member_candidates:
                return self._select_function_candidate(
                    member_candidates,
                    argument_types,
                    node,
                    missing_code="undeclared_member_function",
                    missing_message=f"undeclared member function '{self.current_class.name}::{name}'.",
                )

        candidates = self.global_table.lookup(name, {"function"}) # Free functions are stored in the global table
        return self._select_function_candidate(
            candidates,
            argument_types,
            node,
            missing_code="undeclared_undefined_free_function",
            missing_message=f"undeclared or undefined free function '{name}'.",
        )

    # Resolve obj.field after the owner expression has already been typed
    def _resolve_member_variable(self, owner_type: str, name: str, node: IdNode) -> SymbolEntry | None:
        if self._get_str_dimensions_list(owner_type):
            self._diagnostic(
                "error",
                "invalid_dot_operator_member_access",
                f"invalid dot operator member access on non-class type '{owner_type}'.",
                node,
            )
            return None
        class_table = self._get_class_table_for_type(owner_type) # Translate the owner type name into its class symbol table
        if class_table is None:
            self._diagnostic(
                "error",
                "invalid_dot_operator_member_access",
                f"invalid dot operator member access on non-class type '{owner_type}'.",
                node,
            )
            return None
        candidates = class_table.lookup(name, {"data_member"}) # Lookup can also see inherited members because symbol tables link inheritance
        if not candidates:
            self._diagnostic(
                "error",
                "undeclared_member_variable",
                f"undeclared member variable '{class_table.name}.{name}'.",
                node,
            )
            return None
        return candidates[0]

    # Resolve obj.method(args) after the owner expression has already been typed
    def _resolve_member_call(self, owner_type: str, name: str, call_node: AParamsNode, node: IdNode) -> SymbolEntry | None:
        if self._get_str_dimensions_list(owner_type):
            self._diagnostic(
                "error",
                "invalid_dot_operator_member_access",
                f"invalid dot operator member access on non-class type '{owner_type}'.",
                node,
            )
            return None
        class_table = self._get_class_table_for_type(owner_type)
        if class_table is None:
            self._diagnostic(
                "error",
                "invalid_dot_operator_member_access",
                f"invalid dot operator member access on non-class type '{owner_type}'.",
                node,
            )
            return None
        candidates = class_table.lookup(name, {"member_function"})
        return self._select_function_candidate(
            candidates,
            call_node.accept(self), # Evaluate argument types before overload matching
            node,
            missing_code="undeclared_member_function",
            missing_message=f"undeclared member function '{class_table.name}::{name}'.",
        )

    # Choose the unique function candidate compatible with the argument list, reporting type errors when no candidate matches
    def _select_function_candidate(
        self,
        candidates: list[SymbolEntry],
        argument_types: list[str | None],
        node,
        missing_code: str,
        missing_message: str,
    ) -> SymbolEntry | None:
        if not candidates:
            self._diagnostic("error", missing_code, missing_message, node)
            return None

        expected_count = len(argument_types)
        same_arity = [candidate for candidate in candidates if len(candidate.parameter_types) == expected_count] # Filter same function names by same num of arguments
        if not same_arity:
            self._diagnostic(
                "error",
                "function_call_wrong_number_of_parameters",
                f"function call with wrong number of parameters for '{candidates[0].name}'.",
                node,
            )
            return None

        concrete_argument_types = [arg for arg in argument_types if arg is not None] # If any argument failed to type-check, suppress duplicate call-type errors
        if len(concrete_argument_types) != len(argument_types):
            return None

        for candidate in same_arity:
            if self._argument_lists_match(candidate.parameter_types, concrete_argument_types): # Check if every type of parameter matches
                node.symtab_entry = candidate
                return candidate

        self._diagnostic(
            "error",
            "function_call_wrong_type_of_parameters",
            f"function call with wrong type of parameters for '{same_arity[0].name}'.",
            node,
        )
        return None

    # Apply every array index in a chain segment, ensuring each index expression is an integer and that the number of indexes does not exceed the declared num of indices
    def _remove_indices(self, entry: SymbolEntry, indices: list[IndexNode], node) -> str | None:
        dimensions = list(entry.array_dimensions) # Work from the declared array shape stored on the symbol entry
        if len(indices) > len(dimensions):
            self._diagnostic(
                "error",
                "wrong_array_dimensionality",
                f"wrong array dimensionality for '{entry.name}'.",
                node,
            )
            return None

        for index_node in indices:
            index_type = index_node.first_child.accept(self)
            if index_type != "integer":
                self._diagnostic(
                    "error",
                    "non_integer_index",
                    f"array index for '{entry.name}' must be of type 'integer'.",
                    index_node,
                )
                return None

        remaining_get_str_dimensions_list = dimensions[len(indices):] # Each successful index removes one array dimension from the resulting type
        return self._compose_type(entry.type, remaining_get_str_dimensions_list)

    # Compare two parameter lists using the same exact-type compatibility rules to see if theyre the same
    def _argument_lists_match(self, expected_types: list[str], actual_types: list[str]) -> bool:
        if len(expected_types) != len(actual_types):
            return False
        return all(self._is_assignable(expected, actual) for expected, actual in zip(expected_types, actual_types))

    # Type-compatibility rule used for assignments, returns, and argument passing.
    def _is_assignable(self, target_type: str | None, source_type: str | None) -> bool:
        if target_type == source_type:
            return True

        target_dims = self._get_str_dimensions_list(target_type)
        source_dims = self._get_str_dimensions_list(source_type)
        if target_dims or source_dims:
            if self._get_base_type(target_type) != self._get_base_type(source_type): # Array element base types must match exactly
                return False
            if len(target_dims) != len(source_dims):
                return False
            for expected_dim, actual_dim in zip(target_dims, source_dims):
                if expected_dim and actual_dim and expected_dim != actual_dim:
                    return False
            return True

        return False

    # Return whether a type can participate in numeric operators. Arrays are excluded even if their base type is numeric
    def _is_numeric_operand(self, type_name: str | None) -> bool:
        return self._get_base_type(type_name) in {"integer", "float"} and not self._get_str_dimensions_list(type_name)

    def _check_type_exists(self, type_name: str, node) -> None:
        if type_name in {"integer", "float", "void"}: # Built-in data type
            return
        if self._lookup_class(type_name) is None: # Data type is of a class
            self._diagnostic("error", "undeclared_class", f"undeclared class '{type_name}'.", node)

    # Lookup a class entry by name in the global symbol table
    def _lookup_class(self, class_name: str) -> SymbolEntry | None:
        if self.global_table is None:
            return None
        classes = self.global_table.lookup(class_name, {"class"})
        if not classes:
            return None
        return classes[0]

    # Get the class table based on the Class type of the object we're currently at
    def _get_class_table_for_type(self, type_name: str) -> SymbolTable | None:
        class_entry = self._lookup_class(self._get_base_type(type_name))
        if class_entry is None:
            return None
        return class_entry.inner_scope_table

    def _entry_type(self, entry: SymbolEntry) -> str:
        return self._compose_type(entry.type, entry.array_dimensions)

    # Compose a type like `integer[][]` or `MyClass[10]` from its base type plus stored array dimensions
    def _compose_type(self, base_type: str | None, dimensions: list[int | None]) -> str | None:
        if base_type is None:
            return None
        suffix = []
        for dimension in dimensions:
            if dimension is None:
                suffix.append("[]")
            else:
                suffix.append(f"[{dimension}]")
        return f"{base_type}{''.join(suffix)}"

    # Get type while stripping array suffix
    def _get_base_type(self, type_name: str | None) -> str | None:
        if type_name is None:
            return None
        return type_name.split("[", 1)[0]

    # Returns list of dimensions as string numbers without the square brackets
    def _get_str_dimensions_list(self, type_name: str | None) -> list[str]:
        if type_name is None:
            return []
        dimensions = []
        index = type_name.find("[")
        while index != -1:
            end = type_name.find("]", index)
            dimensions.append(type_name[index + 1:end]) # Empty string means an unspecified dimension like []
            index = type_name.find("[", end)
        return dimensions

    # Array values may only remain array-typed when they are being passed as actual arguments; otherwise references must index all declared dimensions
    def _allows_array_reference(self, owner_node) -> bool:
        return isinstance(getattr(owner_node, "parent", None), AParamsNode)

    def _check_circular_classes(self) -> None:
        graph: dict[str, set[str]] = {} # Maps classes to their dependencies (inherited classes or class-members whose type is another class)
        for class_entry in self.global_table.lookup(kinds={"class"}):
            inherited_classes = set()
            class_node = class_entry.node
            
            for inherits_node in class_node.inherits:
                inherited_classes.add(inherits_node.id_node.token.lexeme)
            
            class_table = class_entry.inner_scope_table
            for member in class_table.lookup(kinds={"data_member"}):
                member_type = member.type
                if (member_type not in {"integer", "float", "void"} and member_type != class_entry.name) or member_type == class_entry.name:
                    inherited_classes.add(member_type)
            graph[class_entry.name] = inherited_classes

        
        visiting: set[str] = set() # classes currently on the active DFS recursion path
        explored: set[str] = set() # classes whose dependency graphs have already been checked
        errors: set[str] = set() # prevent duplicate diagnostics when multiple DFS paths hit the same cycle
        stack: list[str] = [] # track the current DFS path so the exact cycle can be extracted

        def dfs(class_name: str) -> None:
            visiting.add(class_name)
            stack.append(class_name)
            for dependency in graph[class_name]:
                if dependency not in graph:
                    continue # Ignore inherited_classes that are not known classes like inheriting a class that doesn't exist, that specific check is reported elsewhere like in visit_ClassDeclNode
                
                if dependency in visiting: #There's a cycle
                    cycle = stack[stack.index(dependency):] # Get cur path
                    for cycle_name in cycle:
                        if cycle_name in errors:
                            continue
                        class_entry = self._lookup_class(cycle_name)
                        self._diagnostic(
                            "error",
                            "circular_class_dependency",
                            f"circular class dependency involving '{cycle_name}'.",
                            class_entry.node,
                        )
                        errors.add(cycle_name)
                    continue
                
                elif dependency not in explored: # The class has not been seen in the current recursion and also has not been explored, so we must explore this one for circular dependencies
                    dfs(dependency)
                
            stack.pop() 
            visiting.remove(class_name)
            explored.add(class_name)

        # Loop through the classes in the graph
        for class_name in graph:
            if class_name not in explored: # If this class wasn't explored yet, start a DFS from it because we might have multiply disconnected graphs
                dfs(class_name)

    def _diagnostic(self, severity: str, code: str, message: str, node) -> None:
        self.diagnostics.append(Diagnostic(severity=severity, code=code, message=message, line=node.token.line))
