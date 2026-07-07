import json

with open("notebooks/02_preprocessing.ipynb", "r") as f:
    nb = json.load(f)

for i in [25, 26, 28]:
    src = "".join(nb["cells"][i]["source"])
    print(f"=== Cell {i} ===")
    print(src)
    print()
