import contextlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PYEDR_FILE = PROJECT_ROOT / 'pyedr' / 'pyedr.py'

_LOCAL_PYEDR_MODULE = None


def _get_local_pyedr_module():
    global _LOCAL_PYEDR_MODULE
    if _LOCAL_PYEDR_MODULE is not None:
        return _LOCAL_PYEDR_MODULE

    if not LOCAL_PYEDR_FILE.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location('edterm_local_pyedr', str(LOCAL_PYEDR_FILE))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOCAL_PYEDR_MODULE = module
        return _LOCAL_PYEDR_MODULE
    except Exception:
        return None


def load_data(file_path, verbose=False, stderr_sink=None, progress_callback=None):
    try:
        local_pyedr = _get_local_pyedr_module()
        if local_pyedr is not None:
            all_energies, all_names, times = local_pyedr.read_edr(
                file_path,
                verbose=verbose,
                progress_callback=progress_callback,
            )
            return pd.DataFrame(all_energies, columns=all_names, index=times)

        import panedr
        if stderr_sink is not None:
            with contextlib.redirect_stderr(stderr_sink):
                return panedr.edr_to_df(file_path, verbose=verbose)
        return panedr.edr_to_df(file_path, verbose=verbose)
    except Exception as exc:
        print(f'Error reading file: {exc}', file=sys.stderr)
        return pd.DataFrame()  # return empty dataframe on failure


def load_units(file_path):
    try:
        local_pyedr = _get_local_pyedr_module()
        if local_pyedr is not None:
            return local_pyedr.get_unit_dictionary(file_path)

        import panedr
        if hasattr(panedr, 'get_unit_dictionary'):
            return panedr.get_unit_dictionary(file_path)
    except Exception:
        return {}
    return {}
