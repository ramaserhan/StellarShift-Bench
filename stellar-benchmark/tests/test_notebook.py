import json
import os
from pathlib import Path
import sys
import types


def test_instant_results_notebook_runs_all_code_cells():
    if "IPython.display" not in sys.modules:
        try:
            import IPython.display  # noqa: F401
        except ImportError:
            display_module = types.ModuleType("IPython.display")

            class Image:
                def __init__(self, filename=None, **_):
                    if filename is not None and not Path(filename).is_file():
                        raise FileNotFoundError(filename)

            display_module.Image = Image
            display_module.display = lambda *_args, **_kwargs: None
            ipython_module = types.ModuleType("IPython")
            ipython_module.display = display_module
            sys.modules["IPython"] = ipython_module
            sys.modules["IPython.display"] = display_module

    root = Path(__file__).parents[1]
    notebook_path = root / "examples" / "StellarShift_v1.2.3_Instant_Results.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_test__"}
    previous = Path.cwd()
    os.chdir(root)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous)
