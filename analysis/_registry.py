import importlib.util
import os


def discover(analysis_dir: str = None) -> list:
    """Return list of (display_name, description, module) tuples."""
    if analysis_dir is None:
        analysis_dir = os.path.dirname(__file__)
    results = []
    for fname in sorted(os.listdir(analysis_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(analysis_dir, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            if hasattr(module, "DISPLAY_NAME") and hasattr(module, "run"):
                results.append((module.DISPLAY_NAME,
                                 getattr(module, "DESCRIPTION", ""),
                                 module))
        except Exception as e:
            print(f"[registry] Could not load {fname}: {e}")
    return results
