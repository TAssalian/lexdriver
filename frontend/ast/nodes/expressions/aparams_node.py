from dataclasses import dataclass, field

from frontend.ast.nodes.base import Node


@dataclass
class AParamsNode(Node):
    args: list[Node] = field(default_factory=list)
