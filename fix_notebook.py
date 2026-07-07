import json

with open("notebooks/02_preprocessing.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "PreprocessingConfig.load" in src:
        new_src = src.replace(
            "from alpr_dataset.config import PipelineConfig, PreprocessingConfig",
            "from alpr_dataset.config import PipelineConfig"
        ).replace(
            'prep_config = PreprocessingConfig.load(\n    PROJECT_ROOT / "configs" / "preprocessing_config.yaml"\n)',
            'prep_config = config.preprocessing_config(\n    PROJECT_ROOT / "configs" / "preprocessing_config.yaml"\n)'
        )
        nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
        print(f"Fixed cell {i}")
        break

with open("notebooks/02_preprocessing.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved")
