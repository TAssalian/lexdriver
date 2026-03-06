from dataclasses import dataclass

from frontend.ast.nodes.base import Node
from frontend.lexer.tokens import Token

@dataclass
class PublicNode(Node):
    pass