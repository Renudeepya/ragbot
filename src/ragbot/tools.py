"""Tool registry for the chatbot's function-calling loop.

Tool errors are returned to the model as text rather than raised. A model that
sees "error: unknown order id" can apologize or retry; an exception just kills
the conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolError(Exception):
    """Raised inside a tool to signal a handled, model-visible failure."""


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Run a tool, returning a string the model can read.

        Never raises: unknown tools, bad arguments, and tool exceptions all come
        back as `error: ...` so the model can recover within the same turn.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}; available: {', '.join(self._tools) or 'none'}"
        try:
            result = tool.fn(**(arguments or {}))
        except ToolError as e:
            return f"error: {e}"
        except TypeError as e:
            return f"error: bad arguments for {name!r}: {e}"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)


# ------------------------------------------------------------------ built-ins

def calculator(expression: str) -> str:
    """Evaluate a small arithmetic expression.

    Uses the `ast` module with an explicit node allowlist — never `eval` on
    model-supplied text, which would be arbitrary code execution.
    """
    import ast
    import operator as op

    ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        ast.USub: op.neg, ast.UAdd: op.pos, ast.FloorDiv: op.floordiv,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ToolError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"could not parse {expression!r}: {e}") from e
    try:
        return str(_eval(tree.body))
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e


def make_calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression, e.g. '3 * (4 + 5)'.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression"}
            },
            "required": ["expression"],
        },
        fn=calculator,
    )


def make_lookup_tool(records: dict[str, dict[str, Any]], name: str = "lookup_order") -> Tool:
    """A record-lookup tool over an in-memory table."""

    def lookup(order_id: str) -> dict[str, Any]:
        rec = records.get(str(order_id))
        if rec is None:
            raise ToolError(f"unknown order id {order_id!r}")
        return rec

    return Tool(
        name=name,
        description="Look up an order by its id and return its status and details.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order identifier"}},
            "required": ["order_id"],
        },
        fn=lookup,
    )
