"""Load solver definitions without launching its command-line simulation."""

import ast
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_solver(source_path=None):
    source = Path(source_path) if source_path else Path(__file__).resolve().parents[1] / "FPQL_201.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    module.body = [
        node for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
    ]
    namespace = {"__name__": "fpql_test_solver", "__file__": str(source)}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace
