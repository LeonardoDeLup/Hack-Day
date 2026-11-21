"""Execute the evaluation notebook with the current project's Python kernel."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
notebook_path = root / "output/jupyter-notebook/fruit-evaluation.ipynb"
runtime = root / "tmp/jupyter-runtime"
runtime.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(runtime / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(root)}})
# Jupyter's native python3 kernel resolves "python" against the launching interpreter.
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, notebook_path)
print(f"Executed and validated {notebook_path} with {sys.executable}")
