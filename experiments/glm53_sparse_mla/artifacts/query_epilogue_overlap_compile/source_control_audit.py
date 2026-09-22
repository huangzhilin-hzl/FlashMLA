"""Check that only Q publication placement changes in the overlap controls."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
results = {}
for parent, child in [("v278", "v284"), ("v281", "v285")]:
    old = ast.parse((root / f"kernel_{parent}.py").read_text())
    new = ast.parse((root / f"kernel_{child}.py").read_text())
    cls = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "SparseMLA")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    role = method.body[-1]
    while role.orelse and isinstance(role.orelse[0], ast.If):
        role = role.orelse[0]
    loop = next(n for n in role.body if isinstance(n, ast.For))
    assert ast.unparse(loop.body[0].test) == "(warp == 0) & (qi < min(q.shape[0], 148))"
    loop.body[0].test = ast.parse("warp == 0", mode="eval").body
    prefetch = [n for n in loop.body if isinstance(n, ast.If) and
                ast.unparse(n.test) == "(warp == 0) & (qi + min(q.shape[0], 148) < q.shape[0])"]
    assert len(prefetch) == 1
    loop.body.remove(prefetch[0])
    for tree in (old, new):
        tree.body.pop(0)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                n.value = n.value.replace(child, parent)
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    results[child] = {"parent": parent, "isolated_q_publication_change": True,
                      "source_sha256": hashlib.sha256((root / f"kernel_{child}.py").read_bytes()).hexdigest()}
for batch in [1, 2, 148, 149, 296, 297, 512, 513, 8192]:
    grid = min(batch, 148)
    loads = []
    for cta in range(grid):
        loads.append(cta)
        for qi in range(cta, batch, grid):
            if qi + grid < batch:
                loads.append(qi + grid)
    assert sorted(loads) == list(range(batch))
Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
