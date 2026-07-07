import json
from pathlib import Path

for nb_name in ["01_eda.ipynb", "02_preprocessing.ipynb"]:
    path = Path("notebooks") / nb_name
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        print(f"\n=== {nb_name} ===")
        print(f"Valid JSON. Cells: {len(nb.get('cells', []))}")
        for i, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            print(f"  Cell {i}: {cell.get('cell_type'):8s} - {len(src)} chars")
    except Exception as e:
        print(f"Error loading {nb_name}: {e}")
