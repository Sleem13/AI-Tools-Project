import json

with open("notebooks/01_eda.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    issues = []
    # Check for references to undefined variables (common issues)
    for ref in ["pipeline.run(", "pipeline.apply_step(", "pipeline.steps", "unified_dir"]:
        if ref in src:
            issues.append(ref)
    if issues:
        print(f"Cell {i}: WARN {issues}")
    else:
        print(f"Cell {i}: OK")

n_cells = len(nb["cells"])
n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
print(f"\nTotal: {n_cells} cells ({n_code} code, {n_md} markdown)")
print("JSON: valid" if json.dumps(nb) else "INVALID")
