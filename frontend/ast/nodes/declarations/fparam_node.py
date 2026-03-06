from dataclasses import dataclass

from frontend.ast.nodes.arrays.array_size_node import ArraySizeNode
from frontend.ast.nodes.base import Node
from frontend.ast.nodes.references.id_node import IdNode
from frontend.ast.nodes.type.type_node import TypeNode


@dataclass
class FParamNode(Node):
    type_node: TypeNode | None = None
    id_node: IdNode | None = None
    array_size_node: ArraySizeNode | None = None
