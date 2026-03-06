from dataclasses import dataclass, field

from frontend.ast.nodes.base import Node
from frontend.ast.nodes.expressions.intnum_node import IntNumNode


@dataclass
class ArraySizeNode(Node):
    dimensions: list[IntNumNode] = field(default_factory=list)
