from __future__ import annotations

from frontend.ast.nodes import (
    AParamsNode,
    AddOpNode,
    AssignOpNode,
    FloatNumNode,
    IndexNode,
    IntNumNode,
    MinusNode,
    MultOpNode,
    NotNode,
    PlusNode,
    RelOpNode,
    VariableNode,
)
from frontend.semantics.visitors.visitor import Visitor

# Nodes whose memory has already been calculated earlier
_VALUE_NODES = (IntNumNode, FloatNumNode, AddOpNode, MultOpNode, RelOpNode, PlusNode, MinusNode, NotNode)


class CodeGenVisitor(Visitor):
    def __init__(self):
        self.global_table = None
        self.current_scope = None
        self.code_stream = []
        self.data_stream = []
        self.label_counter = 0

    def emit(self, line):
        self.code_stream.append(f"          {line}")

    def label(self, lbl):
        self.code_stream.append(f"{lbl:<10}")

    def new_label(self, prefix):
        lbl = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return lbl

    def slot(self, entry):
        return f"{entry.offset}(r14)"

    def output(self):
        return "\n".join([*self.code_stream, *self.data_stream]) + "\n"

    def lookup(self, name, scope=None):
        return (scope or self.current_scope).lookup(name)[0]

    def function_label(self, entry):
        def safe(part):
            return part.replace("[]", "_array")

        parts = []
        if entry.owner_class:
            parts.append(entry.owner_class)
        parts.append(entry.name)
        parts.extend(safe(param_type) for param_type in entry.parameter_types)
        return "_".join(parts)

    def type_size(self, type_name):
        if not type_name or type_name == "void":
            return 0
        base = type_name.split("[", 1)[0]
        size = 4 if base == "integer" else 8 if base == "float" else self.global_table.lookup(base, {"class"})[0].inner_scope_table.size
        for chunk in type_name.split("[")[1:]:
            dim = chunk.split("]", 1)[0]
            if dim:
                size *= int(dim)
        return size

    def entry_size(self, entry):
        t = entry.type + "".join("[]" if d is None else f"[{d}]" for d in entry.array_dimensions)
        return self.type_size(t)

    # puts the value of node into reg
    # reg: destination register for final value
    # addr_reg: scratch register for if node is not in a known slot
    def load_value(self, node, reg, addr_reg="r12"):
        # if node is a computer/literal value from the prev compute_mem_visitor
        if isinstance(node, _VALUE_NODES):
            self.emit(f"lw {reg},{self.slot(node.symtab_entry)}")
        # something addressable like a variable / attribute access
        else:
            self.find_address(node, addr_reg) # computes address of node into addr_reg
            self.emit(f"lw {reg},0({addr_reg})") # load that address into reg

    # used in rhs to check if we have direct slot or need address resolution
    def get_source_address(self, node, reg):
        # If its memory we've already allocated that exists cleanly, get its memory slot and return the address of it
        if isinstance(node, _VALUE_NODES):
            self.emit(f"addi {reg},r14,{node.symtab_entry.offset}")
        # If it doesn't have a temp slot so its a variable, dot-chain, or function call
        else:
            self.find_address(node, reg)

    # flatten list of nodes making up the expr so we could call _get_address_of_chained_expr and get the address of chained expr
    def find_address(self, node, reg):
        # VariableNode is a wrapper where its children are IndexNode and AParamsNodes, so pass children
        if isinstance(node, VariableNode):
            self._get_address_of_chained_expr(list(node.iter_children()), reg)
        # Node itself is an IdNode so its part of a larger Statement, we add that + its children to the list to resolve a potential dot access.
        else:
            self._get_address_of_chained_expr([node, *node.iter_children()], reg)

    def address_of_base_entry(self, entry, reg):
        if entry.kind == "data_member":
            self.emit(f"lw r11,{self.current_scope.size}(r14)")
            self.emit(f"addi {reg},r11,{entry.offset}")
        elif entry.kind == "param" and entry.array_dimensions:
            self.emit(f"lw {reg},{self.slot(entry)}")
        else:
            self.emit(f"addi {reg},r14,{entry.offset}")

    # Goes through a dot-separated chain like 'a.b[2].c(x)' to compute the final memory address to store in 'reg'
    # 'children' is a flat list of nodes: [IdNode, IndexNode?, AParamsNode?, IdNode, IndexNode?, ...]
    # Example: 'a.b[2].c(x)' is children = [a_id, b_id, IndexNode(2), c_id, AParamsNode(x)]
    # For a simple variable like 'x', there's only one iteration: address_of_base_entry writes 'addi reg, r14, offset'
    def _get_address_of_chained_expr(self, children, reg):
        is_first = True
        i = 0

        while i < len(children):
            # Parse one segment: grab the identifier node
            id_node = children[i]
            i += 1 

            # Collect any array index nodes following this identifier
            indices = []
            while i < len(children) and isinstance(children[i], IndexNode):
                indices.append(children[i])
                i += 1
    
            # Check if a function call follows
            call_node = None
            if i < len(children) and isinstance(children[i], AParamsNode):
                call_node = children[i]
                i += 1

            entry = id_node.symtab_entry
            
            if is_first and call_node is not None:
                self._call(entry, call_node, None)
                self.emit(f"addi {reg},r14,{self.lookup('__ret_val', entry.inner_scope_table).offset - entry.inner_scope_table.size}") # compute address of callee's return value slot from caller's frame and puts that address into reg. gives negative offset to move downward from caller's frame pointer to callee frame

            elif is_first:
                # First identifier in the chain: resolve as data member, array param, or local var
                self.address_of_base_entry(entry, reg)
            
            elif call_node is not None:
                # Dot-chain function call like a.f(): reg already holds address of 'a'
                self._call(entry, call_node, reg)
                # Point reg to where the return value landed in our frame
                self.emit(f"addi {reg},r14,{self.lookup('__ret_val', entry.inner_scope_table).offset - entry.inner_scope_table.size}")

            else:
                # Dot-chain data access like a.b: offset into the class member
                self.emit(f"addi {reg},{reg},{entry.offset}")

            if indices:
                self._apply_index_offsetting(reg, entry, indices)

            is_first = False

    def _call(self, entry, call_node, receiver_reg):
        callee = entry.inner_scope_table
        for arg, param in zip(call_node.args, callee.lookup(kinds={"param"})): # write each passed argument into its slot in the callee frame
            arg.accept(self) # generate code for argument expression first
            dst = param.offset - callee.size # compute where the slot for that argument lives relative to caller's current r14. example param.offset = 8 and callee.size = 20 writes at -12(r14)
            if param.array_dimensions: # means its passed by address
                self.find_address(arg, "r1") 
                self.emit(f"sw {dst}(r14),r1") # store retrieved address directly into dst
            else: # passed by value
                self.get_source_address(arg, "r12") # copmute where argument lives in memory and puts address into r12
                self.emit(f"addi r2,r14,{dst}") # compute destination address and put it into r2
                self._copy_block("r12", "r2", self.entry_size(param)) # copies passed argument at r12 into the parameter slot r2
        if receiver_reg is not None: # receiver_reg is reg of an object for which the method is being called upon. for a.f(), receiver_reg is the reg of 'a'
            self.emit(f"sw 0(r14),{receiver_reg}") # store the object address into 0(r14) relative to caller's current frame
        self.emit(f"subi r14,r14,{callee.size}") # now make r14 point to the base of the stack frame of the function we just called, instead of the base of what called the function
        self.emit(f"jl r15,{self.function_label(entry)}") # jump to the label of the function we're calling
        self.emit(f"addi r14,r14,{callee.size}") # restore stack frame pointer to the caller's stack frame instead of the callee

    # reg: holds base address of array object that we then want to find real address of the indexing we're doing upon it
    # indices: index nodes that were referenced in lhs
    # compute offset for each index and add it into reg to get address of variable we're referencing
    def _apply_index_offsetting(self, reg, entry, indices):
        dims = list(entry.array_dimensions)
        remaining_type = entry.type + "".join("[]" if d is None else f"[{d}]" for d in dims[len(indices):]) # get all the remaining indices after the indices we wrote that are not referenced
        elem_size = self.type_size(remaining_type) # get entire size of array and its indices included in the size
        for i, idx_node in enumerate(indices):
            expr = next(idx_node.iter_children())
            addr_reg = "r8" if reg != "r8" else "r11"
            idx_node_reg = "r9" if reg != "r9" else "r11" # value of current index expression
            stride_reg = "r10" if reg != "r10" and idx_node_reg != "r10" else "r11" # byte offset
            self.load_value(expr, idx_node_reg, addr_reg) # evaluate expression representing current index into a scratch register idx_node_reg
            stride = elem_size
            for d in dims[i + 1:]:
                if d is not None:
                    stride *= d # compute size for current dimension by multiplying element size by later fixed dimensions
            self.emit(f"muli {stride_reg},{idx_node_reg},{stride}")
            self.emit(f"add {reg},{reg},{stride_reg}") # reg contains the exact destination for the lhs now

    def _copy_block(self, src, dst, size):
        if size <= 4:
            self.emit(f"lw r3,0({src})")
            self.emit(f"sw 0({dst}),r3")
            return
        # continously copy one 4-byte word at a time from address in src to address in dst
        # used to handle floats, class objects
        # r4 holds the offset
        loop, done = self.new_label("copy"), self.new_label("copy_end")
        self.emit("addi r4,r0,0")
        self.label(loop)
        self.emit(f"ceqi r5,r4,{size}") # compare the current offset val to size
        self.emit(f"bnz r5,{done}") # if equal, jump to done
        self.emit(f"add r6,{src},r4") # compute src + r4 into r6
        self.emit("lw r7,0(r6)") # load one 4-byte word from r6 into r7
        self.emit(f"add r8,{dst},r4") # compute r4 + dst into r8 to know address of where to store in dst
        self.emit("sw 0(r8),r7") # store the word at r8
        self.emit("addi r4,r4,4") # increment r4 by 4
        self.emit(f"j {loop}") # jump back to beginning of loop
        self.label(done)


    def visit_ProgNode(self, node):
        self.global_table = node.symtab
        self.current_scope = node.symtab
        class_list, func_defs, main_block = node.iter_children()
        main_label = self.new_label("main")
        self.emit("entry")
        self.emit("addi r14,r0,topaddr") # initialize r14 as stack/frame pointer
        self.emit(f"subi r14,r14,{main_block.symtab.size}") # reserve stack space for main block
        self.emit(f"j {main_label}") # write line, jump happens later when compiling with moon.c
        class_list.accept(self)
        func_defs.accept(self)
        self.label(main_label)
        main_block.accept(self)
        self.emit("hlt")

    def visit_ProgramBlockNode(self, node):
        prev = self.current_scope
        self.current_scope = node.symtab
        self.visit_children(node)
        self.current_scope = prev

    # Skip because we dont care about declared vars, their memory is already allocated from prev. visitor
    def visit_VarDeclNode(self, node):
        pass

    # jl (ex: jl r15, f1): Holds return address of current instruction we jumped to f1 from
    # sw (ex: sw -4(r14), r15): Copies a value from register into memory
    # lw (ex: lw r15, -4(r14): Copies a value from memory into a register
    # jr (ex: jr r15): Jump back to saved return address inside r15
    def visit_FuncDefNode(self, node):
        prev = self.current_scope
        self.current_scope = node.symtab
        self.label(self.function_label(node.symtab_entry)) # create unique label out of owner class, func name and func params
        self.emit(f"sw {self.slot(self.lookup('__ret_addr'))},r15") # gernerate Moon instruction that saves caller's return address into current function's stack frame, and use r15 bc function calls use jl r15, <label> that puts ret_addr into r15. we store in stack frame bc of recursion
        node.func_body_node.accept(self)
        self.emit(f"lw r15,{self.slot(self.lookup('__ret_addr'))}")
        self.emit("jr r15") # r15 stored next code to be executed after this function call, so we jump back to there
        self.current_scope = prev
    
    def visit_ReturnNode(self, node):
        expr = node.first_child
        expr.accept(self)
        self.get_source_address(expr, "r12") # get address of returned expression and put address into r12
        ret_val_entry = self.lookup("__ret_val") # get symbol entry
        self.emit(f"addi r2,r14,{ret_val_entry.offset}") # compute address of the return-value slot into r2 so it contains address of functions __ret_val slot
        self._copy_block("r12", "r2", self.type_size(expr.inferred_type)) # copy returned value from r12 into the r2 slot
        self.emit(f"lw r15,{self.slot(self.lookup('__ret_addr'))}") # restore saved address from current function's stack frame (where execution should continue after this function call finishes) which is stored in stack frame's ret addr
        self.emit("jr r15") # jump back to saved address

    # Assignment or func call statements
    def visit_StatementNode(self, node):
        children = list(node.iter_children())
        assign_index = next((i for i, c in enumerate(children) if isinstance(c, AssignOpNode)), None) # Get index of AssignOpNode
        
        # Doesn't exist, so we have a function call
        if assign_index is None:
            for c in children:
                c.accept(self)
            self._get_address_of_chained_expr(children, "r12")
            return
        left = children[:assign_index] # The left side becomes the target
        right = children[assign_index + 1] # The right side becomes the value
        for c in left:
            c.accept(self) # go through each child because it could contain subexppressions like arr[i + 1], address depends on (i + 1)
        right.accept(self)
        self.get_source_address(right, "r13") # Get where the RHS lives in memory and store it in r13
        self._get_address_of_chained_expr(left, "r1") # Get the destination address where we want to write to and store it in r1.
        self._copy_block("r13", "r1", self.type_size(node.inferred_type)) 

    def visit_IfNode(self, node):
        condition, then_block, *else_block = node.iter_children()
        else_lbl = self.new_label("if_else")
        end_lbl = self.new_label("if_end")
        condition.accept(self)
        self.load_value(condition, "r1")
        self.emit(f"bz r1,{else_lbl}") # branch if zero to else_lbl
        then_block.accept(self)
        self.emit(f"j {end_lbl}")
        self.label(else_lbl)
        if else_block:
            else_block[0].accept(self)
        self.label(end_lbl)

    # bz: jump to end_lbl if the value is 0, which skips the body and exits loop
    # j: Unconditional jump
    def visit_WhileNode(self, node):
        condition, body = node.iter_children()
        start_lbl = self.new_label("while_start")
        end_lbl = self.new_label("while_end")
        self.label(start_lbl) # Appends the label
        condition.accept(self) # Generate code making up the condition to get T/F value
        self.load_value(condition, "r1") 
        self.emit(f"bz r1,{end_lbl}") # Choose whether to jump to end and exit loop or not based on value in r1
        body.accept(self) # Condition is true so generate code
        self.emit(f"j {start_lbl}") # Jump back to start_lbl for next iteration
        self.label(end_lbl) # Continue with whats after this loop

    def visit_ReadNode(self, node):
        target = node.first_child
        target.accept(self)
        self.emit("jl r15,getint") # save address in r15 and jump to getint. getint saves result in r1
        self.find_address(target, "r12") # get memory address of target variable and store in r12
        self.emit("sw 0(r12),r1") # store integer returned by getint that user inputs into the target memory location of r12

    def visit_WriteNode(self, node):
        expr = node.first_child
        expr.accept(self)
        self.load_value(expr, "r1") # get value of what we want to print and store into r1 bc putint expects it there
        self.emit("jl r15,putint") # call printing subroutine which prints integer stored at r1. r15 implicitly holds address to line below it because of jl command
        self.emit("addi r1,r0,10") # load newline char into r1
        self.emit("putc r1") # print the character

    def visit_IntNumNode(self, node):
        self.emit(f"addi r1,r0,{node.token.lexeme}")
        self.emit(f"sw {self.slot(node.symtab_entry)},r1")

    # just stores 0 in both slots, does not generate real float value
    def visit_FloatNumNode(self, node):
        self.emit("addi r1,r0,0")
        self.emit(f"sw {self.slot(node.symtab_entry)},r1")
        self.emit(f"sw {node.symtab_entry.offset + 4}(r14),r1")

    def visit_PlusNode(self, node):
        self.visit_children(node)
        self.load_value(node.first_child, "r1")
        self.emit(f"sw {self.slot(node.symtab_entry)},r1")

    def visit_MinusNode(self, node):
        self.visit_children(node)
        self.load_value(node.first_child, "r1")
        self.emit("sub r1,r0,r1") # does r1 = 0 - r1 to store negated value
        self.emit(f"sw {self.slot(node.symtab_entry)},r1")

    def visit_NotNode(self, node):
        self.visit_children(node)
        self.load_value(node.first_child, "r1")
        self.emit("ceq r1,r1,r0") # compare equal, store 1 if equal
        self.emit(f"sw {self.slot(node.symtab_entry)},r1")

    def visit_AddOpNode(self, node):
        self._binary(node, {"+": "add", "-": "sub", "or": "or"})

    def visit_MultOpNode(self, node):
        self._binary(node, {"*": "mul", "/": "div", "and": "and"})

    def visit_RelOpNode(self, node):
        self._binary(node, {"==": "ceq", "<>": "cne", "<": "clt", "<=": "cle", ">": "cgt", ">=": "cge"})

    def _binary(self, node, ops):
        self.visit_children(node)
        left = node.first_child
        right = left.next_sibling
        self.load_value(left, "r1") # store value of left into r1
        self.load_value(right, "r2") # store value of right into r2
        self.emit(f"{ops[node.token.lexeme]} r3,r1,r2") # store result into r3 with operation applied
        self.emit(f"sw {self.slot(node.symtab_entry)},r3") # store result 
