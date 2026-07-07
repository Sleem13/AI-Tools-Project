import json

with open("notebooks/02_preprocessing.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    issues = []
    if "pipeline.run(" in src and "pipeline.run_on_dataset" not in src:
        issues.append("pipeline.run()")
    if "pipeline.apply_step(" in src:
        issues.append("pipeline.apply_step()")
    if "pipeline.steps" in src:
        issues.append("pipeline.steps")
    if "pipeline.target_size" in src:
        issues.append("pipeline.target_size")
    if "unified_dir" in src:
        issues.append("unified_dir")
    if issues:
        print(f"Cell {i}: WARN {issues}")
    else:
        print(f"Cell {i}: OK")

n_cells = len(nb["cells"])
n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
print(f"\nTotal: {n_cells} cells ({n_code} code, {n_md} markdown)")

# Check JSON validity by re-serializing
json.dumps(nb)
print("JSON: valid")
