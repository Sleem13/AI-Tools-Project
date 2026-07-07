import json
from pathlib import Path

for fname in ["01_eda.ipynb", "02_preprocessing.ipynb"]:
    with open(f"notebooks/{fname}") as f:
        nb = json.load(f)

    # Find first code cell and fix the PROJECT_ROOT logic
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "PROJECT_ROOT" in src:
                old = (
                    '# --- path setup ---\n'
                    'PROJECT_ROOT = Path.cwd().resolve()\n'
                    'if str(PROJECT_ROOT) not in sys.path:\n'
                    '    sys.path.insert(0, str(PROJECT_ROOT / "src"))\n'
                )
                new = (
                    '# --- path setup ---\n'
                    '# Walk up to find project root (handles kernels with different CWD)\n'
                    '_candidate = Path.cwd().resolve()\n'
                    'for _parent in [_candidate] + list(_candidate.parents):\n'
                    '    if (_parent / "src" / "alpr_dataset").is_dir():\n'
                    '        PROJECT_ROOT = _parent\n'
                    '        break\n'
                    'else:\n'
                    '    PROJECT_ROOT = _candidate\n'
                    'if str(PROJECT_ROOT) not in sys.path:\n'
                    '    sys.path.insert(0, str(PROJECT_ROOT / "src"))\n'
                )
                if old in src:
                    src = src.replace(old, new)
                    cell["source"] = src.splitlines(keepends=True)
                    print(f"{fname}: path setup updated (found old format)")
                else:
                    # Check for the compact format (without -- path setup -- comment)
                    old2 = (
                        'PROJECT_ROOT = Path.cwd().resolve()\n'
                        'if str(PROJECT_ROOT) not in sys.path:\n'
                        '    sys.path.insert(0, str(PROJECT_ROOT / "src"))\n'
                    )
                    new2 = (
                        '# Walk up to find project root (handles kernels with different CWD)\n'
                        '_candidate = Path.cwd().resolve()\n'
                        'for _parent in [_candidate] + list(_candidate.parents):\n'
                        '    if (_parent / "src" / "alpr_dataset").is_dir():\n'
                        '        PROJECT_ROOT = _parent\n'
                        '        break\n'
                        'else:\n'
                        '    PROJECT_ROOT = _candidate\n'
                        'if str(PROJECT_ROOT) not in sys.path:\n'
                        '    sys.path.insert(0, str(PROJECT_ROOT / "src"))\n'
                    )
                    if old2 in src:
                        src = src.replace(old2, new2)
                        cell["source"] = src.splitlines(keepends=True)
                        print(f"{fname}: path setup updated (found compact format)")
                    else:
                        print(f"{fname}: unknown path format, printing first 400 chars")
                        print(src[:400])
                        print("---")
                break

    with open(f"notebooks/{fname}", "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
