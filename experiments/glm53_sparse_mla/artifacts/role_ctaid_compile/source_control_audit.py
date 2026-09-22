"""Verify the isolated source transformations, independent of runtime checks."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
results = {}
for parent, candidate in [("v272", "v278"), ("v273", "v279"),
                          ("v278", "v280"), ("v279", "v281")]:
    original = ast.parse((root / f"kernel_{parent}.py").read_text())
    modified = ast.parse((root / f"kernel_{candidate}.py").read_text())
    cls = next(n for n in modified.body if isinstance(n, ast.ClassDef)
               and n.name == "SparseMLA")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "kernel")
    if candidate in ("v278", "v279"):
        role = method.body[-1]
        while role.orelse and isinstance(role.orelse[0], ast.If):
            role = role.orelse[0]
        release = role.body.pop()
        assert isinstance(release, ast.If)
        assert ast.unparse(release.test) == "warp == 0"
        method.body.append(release)
    else:
        helper = next(n for n in modified.body if isinstance(n, ast.FunctionDef)
                      and n.name == "role_cta_id")
        assert ast.unparse(helper).count("%ctaid.x") == 1
        modified.body.remove(helper)
        method.body.insert(1, ast.parse("cta_id, _, _ = cute.arch.block_idx()").body[0])
        restored = 0
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "range" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                        if arg.func.id == "role_cta_id":
                            node.args[0] = ast.Name(id="cta_id", ctx=ast.Load())
                            restored += 1
        assert restored == 3
    for tree in (original, modified):
        tree.body.pop(0)  # Version docstring.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace(candidate, parent)
    assert ast.dump(original, include_attributes=False) == ast.dump(modified, include_attributes=False)
    results[candidate] = {
        "parent": parent,
        "isolated_source_transformation": True,
        "sha256": hashlib.sha256((root / f"kernel_{candidate}.py").read_bytes()).hexdigest(),
    }
path = Path(__file__).with_name("source_control_audit.json")
path.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
