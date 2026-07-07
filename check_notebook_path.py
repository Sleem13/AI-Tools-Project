import json

for fname in ["01_eda.ipynb", "02_preprocessing.ipynb"]:
    with open(f"notebooks/{fname}") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            print(f"{fname}: first code cell:")
            print(f"  sys.path setup: {'sys.path' in src}")
            print(f"  First 300 chars: {src[:300]}")
            print()
            break
