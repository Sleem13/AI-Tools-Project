import json

with open("notebooks/02_preprocessing.ipynb", "r") as f:
    nb = json.load(f)

# Fix cell 25: wrong imports + API
new25 = (
    "from alpr_dataset.splitting.splitter import stratified_split, write_split_manifests\n"
    "from alpr_dataset.config import SplitConfig\n"
    "\n"
    'split_cfg = config.split_config(\n'
    '    PROJECT_ROOT / "configs" / "preprocessing_config.yaml"\n'
    ")\n"
    "\n"
    "for spec in config.datasets:\n"
    '    print(f"\\n{spec.name}:")\n'
    '    image_dir = processed_root / spec.name / "images"\n'
    '    label_dir = processed_root / spec.name / "labels"\n'
    "\n"
    "    if not label_dir.is_dir():\n"
    "        label_dir = unified_dir / spec.name\n"
    "\n"
    "    all_annotations = load_dataset_annotations(spec)\n"
    "    annotations = list(all_annotations.values())\n"
    "\n"
    "    result = stratified_split(annotations, split_cfg)\n"
    "    summary = result.summary()\n"
    '    print(f"  Train: {summary[\\"n_train\\"]}, Val: {summary[\\"n_val\\"]}, Test: {summary[\\"n_test\\"]}")\n'
    "\n"
    "    out_dir = config.processed_dir / \"split\" / spec.name\n"
    "    write_split_manifests(result, out_dir)\n"
    '    print(f"  Manifests -> {out_dir}")\n'
)
nb["cells"][25]["source"] = new25.splitlines(keepends=True)

# Fix cell 26: use correct SplitResult API
new26 = (
    "# Visualise split distribution\n"
    "spec = config.datasets[0]\n"
    "\n"
    "all_annotations = load_dataset_annotations(spec)\n"
    "annotations = list(all_annotations.values())\n"
    "\n"
    "split_cfg = config.split_config(\n"
    '    PROJECT_ROOT / "configs" / "preprocessing_config.yaml"\n'
    ")\n"
    "result = stratified_split(annotations, split_cfg)\n"
    "summary = result.summary()\n"
    "\n"
    "counts = {\n"
    '    "train": summary["n_train"],\n'
    '    "val": summary["n_val"],\n'
    '    "test": summary["n_test"],\n'
    "}\n"
    "\n"
    "if any(counts.values()):\n"
    '    fig, ax = plt.subplots(figsize=(5, 4))\n'
    '    colors = ["#3d5a80", "#ee6c4d", "#98c1d9"]\n'
    "    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor=\"white\")\n"
    '    ax.set_title(f"{spec.name}: Train / Val / Test Split")\n'
    '    ax.set_ylabel("Number of images")\n'
    "    for bar, val in zip(bars, counts.values()):\n"
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,\n"
    '                str(val), ha="center", fontsize=10)\n'
    "    fig.tight_layout(); plt.show()\n"
    "\n"
    "    total = sum(counts.values())\n"
    '    print(f"Total: {total}  |  Ratios: T={counts[\\"train\\"]/total:.1%} V={counts[\\"val\\"]/total:.1%} Te={counts[\\"test\\"]/total:.1%}")\n'
)
nb["cells"][26]["source"] = new26.splitlines(keepends=True)

# Fix cell 28: add back the processed_root definition just in case
new28 = (
    "print(\"All stages complete.\")\n"
    'print(f"\\nOutput directories:")\n'
    'print(f"  Unified YOLO:   {unified_dir}")\n'
    'print(f"  Preprocessed:   {processed_root}")\n'
    'print(f"  Split:          {config.processed_dir / \\"split\\"}")\n'
    'print(f"  Reports:        {config.reports_dir}")\n'
    'print(f"  Logs:           {config.logs_dir}")\n'
)
nb["cells"][28]["source"] = new28.splitlines(keepends=True)

with open("notebooks/02_preprocessing.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Cells 25, 26, 28 fixed")
