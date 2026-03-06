from dataclasses import dataclass, field

from frontend.ast.nodes.base import Node
from frontend.ast.nodes.declarations.fparam_node import FParamNode


@dataclass
class FParamsNode(Node):
    params: list[FParamNode] = field(default_factory=list)
