import json

with open("notebooks/02_preprocessing.ipynb", "r") as f:
    nb = json.load(f)

for i in [11, 25, 26, 28]:
    src = "".join(nb["cells"][i]["source"])
    print(f"=== Cell {i} ({nb['cells'][i]['cell_type']}) ===")
    print(src[:500])
    print("---")

# Also find cell 12 which might be markdown
for i in [12]:
    if i < len(nb["cells"]):
        src = "".join(nb["cells"][i]["source"])
        print(f"=== Cell {i} ({nb['cells'][i]['cell_type']}) ===")
        print(src[:500])
        print("---")
