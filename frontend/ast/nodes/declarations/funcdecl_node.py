from dataclasses import dataclass

from frontend.ast.nodes.base import Node
from frontend.ast.nodes.declarations.fparams_node import FParamsNode
from frontend.ast.nodes.declarations.param_list_node import ParamListNode
from frontend.ast.nodes.references.id_node import IdNode
from frontend.ast.nodes.type.type_node import TypeNode


@dataclass
class FuncDeclNode(Node):
    id_node: IdNode | None = None
    fparams_node: FParamsNode | ParamListNode | None = None
    return_type_node: TypeNode | None = None
