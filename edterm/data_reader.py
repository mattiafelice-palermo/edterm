import contextlib
import hashlib
import importlib.util
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PYEDR_FILE = PROJECT_ROOT / 'pyedr' / 'pyedr.py'
CACHE_DIR = Path(
    os.environ.get(
        'EDTERM_CACHE_DIR',
        Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'edterm',
    )
)
PARSER_CACHE_VERSION = 'v3'
MAX_CACHE_FILES = 24
MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

_LOCAL_PYEDR_MODULE = None
_RUST_EDR_MODULE = None
logger = logging.getLogger('edterm.edterm')


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


def load_data(
    file_path,
    verbose=False,
    stderr_sink=None,
    progress_callback=None,
    frame_stride=1,
    use_cache=True,
    reader_selected_callback=None,
):
    t0 = time.perf_counter()
    try:
        file_path = str(Path(file_path).resolve())
        frame_stride = max(1, int(frame_stride))
        cache_key = _cache_key(file_path, frame_stride)
        file_size = os.path.getsize(file_path)
        _log_timing('load_data.start', t0, file=file_path, stride=frame_stride, cache=use_cache, size=file_size)
        if use_cache:
            cache_t0 = time.perf_counter()
            cached_data = _read_cached_df(cache_key)
            _log_timing('load_data.cache_lookup', cache_t0, hit=(cached_data is not None))
            if cached_data is not None:
                _notify_reader_selected(reader_selected_callback, 'cache')
                _log_timing(
                    'load_data.done_cache',
                    t0,
                    rows=len(cached_data.get('time', [])),
                    cols=(1 + len(cached_data.get('columns', []))),
                )
                return cached_data

        rust_t0 = time.perf_counter()
        rust_reader = _get_rust_edr_module()
        _log_timing('load_data.rust_module_check', rust_t0, available=(rust_reader is not None))
        if rust_reader is not None and not verbose:
            try:
                read_t0 = time.perf_counter()
                _notify_reader_selected(reader_selected_callback, 'rust')
                if hasattr(rust_reader, 'read_edr_packed'):
                    payload, nrows, ncols, all_names = rust_reader.read_edr_packed(
                        file_path,
                        frame_stride=frame_stride,
                        progress_callback=progress_callback,
                        progress_stride=1000,
                    )
                    _log_timing('load_data.rust_read', read_t0, frames=nrows, fields=ncols, mode='packed')
                    build_t0 = time.perf_counter()
                    if nrows > 0 and ncols > 0:
                        matrix = np.frombuffer(payload, dtype=np.float64).reshape((nrows, ncols)).copy()
                        data = _dataset_from_matrix(matrix, all_names)
                    else:
                        data = _empty_dataset(all_names)
                    _log_timing(
                        'load_data.dataframe_build',
                        build_t0,
                        rows=len(data['time']),
                        cols=(1 + len(data['columns'])),
                        mode='packed',
                    )
                else:
                    all_energies, all_names, times = rust_reader.read_edr(
                        file_path,
                        frame_stride=frame_stride,
                        progress_callback=progress_callback,
                        progress_stride=1000,
                    )
                    _log_timing('load_data.rust_read', read_t0, frames=len(times), fields=len(all_names), mode='legacy')
                    build_t0 = time.perf_counter()
                    data = _dataset_from_rows(all_energies, all_names)
                    _log_timing(
                        'load_data.dataframe_build',
                        build_t0,
                        rows=len(data['time']),
                        cols=(1 + len(data['columns'])),
                        mode='legacy',
                    )

                if use_cache:
                    cache_write_t0 = time.perf_counter()
                    _write_cached_df(cache_key, data)
                    _log_timing('load_data.cache_write', cache_write_t0)
                _log_timing(
                    'load_data.done_rust',
                    t0,
                    rows=len(data['time']),
                    cols=(1 + len(data['columns'])),
                )
                return data
            except Exception as exc:
                logger.warning('Rust reader failed, falling back to Python reader: %s', exc)
                _log_timing('load_data.rust_failed', t0, error=str(exc))

        py_t0 = time.perf_counter()
        local_pyedr = _get_local_pyedr_module()
        _log_timing('load_data.local_pyedr_check', py_t0, available=(local_pyedr is not None))
        if local_pyedr is not None:
            read_t0 = time.perf_counter()
            _notify_reader_selected(reader_selected_callback, 'python-local')
            all_energies, all_names, times = local_pyedr.read_edr(
                file_path,
                verbose=verbose,
                progress_callback=progress_callback,
                frame_stride=frame_stride,
            )
            _log_timing('load_data.python_read', read_t0, frames=len(times), fields=len(all_names))
            build_t0 = time.perf_counter()
            data = _dataset_from_rows(all_energies, all_names)
            _log_timing(
                'load_data.dataframe_build',
                build_t0,
                rows=len(data['time']),
                cols=(1 + len(data['columns'])),
            )
            if use_cache:
                cache_write_t0 = time.perf_counter()
                _write_cached_df(cache_key, data)
                _log_timing('load_data.cache_write', cache_write_t0)
            _log_timing(
                'load_data.done_python_local',
                t0,
                rows=len(data['time']),
                cols=(1 + len(data['columns'])),
            )
            return data

        import panedr

        panedr_t0 = time.perf_counter()
        _notify_reader_selected(reader_selected_callback, 'panedr')
        if stderr_sink is not None:
            with contextlib.redirect_stderr(stderr_sink):
                df = panedr.edr_to_df(file_path, verbose=verbose)
        else:
            df = panedr.edr_to_df(file_path, verbose=verbose)
        _log_timing('load_data.panedr_read', panedr_t0, rows=len(df), cols=len(df.columns))
        data = _dataset_from_pandas(df)
        if use_cache:
            cache_write_t0 = time.perf_counter()
            _write_cached_df(cache_key, data)
            _log_timing('load_data.cache_write', cache_write_t0)
        _log_timing(
            'load_data.done_panedr',
            t0,
            rows=len(data['time']),
            cols=(1 + len(data['columns'])),
        )
        return data
    except Exception as exc:
        _log_timing('load_data.failed', t0, error=str(exc))
        print(f'Error reading file: {exc}', file=sys.stderr)
        return _empty_dataset([])


def load_units(file_path, use_cache=True):
    t0 = time.perf_counter()
    try:
        file_path = str(Path(file_path).resolve())
        cache_key = _cache_key(file_path, frame_stride=1)
        if use_cache:
            cache_t0 = time.perf_counter()
            cached_units = _read_cached_units(cache_key)
            _log_timing('load_units.cache_lookup', cache_t0, hit=(cached_units is not None))
            if cached_units is not None:
                _log_timing('load_units.done_cache', t0, count=len(cached_units))
                return cached_units

        rust_t0 = time.perf_counter()
        rust_reader = _get_rust_edr_module()
        _log_timing('load_units.rust_module_check', rust_t0, available=(rust_reader is not None))
        if rust_reader is not None and hasattr(rust_reader, 'get_unit_dictionary'):
            try:
                read_t0 = time.perf_counter()
                units = rust_reader.get_unit_dictionary(file_path)
                _log_timing('load_units.rust_read', read_t0, count=len(units))
                if use_cache:
                    cache_write_t0 = time.perf_counter()
                    _write_cached_units(cache_key, units)
                    _log_timing('load_units.cache_write', cache_write_t0)
                _log_timing('load_units.done_rust', t0, count=len(units))
                return units
            except Exception as exc:
                logger.warning('Rust unit reader failed, falling back to Python reader: %s', exc)
                _log_timing('load_units.rust_failed', t0, error=str(exc))

        py_t0 = time.perf_counter()
        local_pyedr = _get_local_pyedr_module()
        _log_timing('load_units.local_pyedr_check', py_t0, available=(local_pyedr is not None))
        if local_pyedr is not None:
            read_t0 = time.perf_counter()
            units = local_pyedr.get_unit_dictionary(file_path)
            _log_timing('load_units.python_read', read_t0, count=len(units))
            if use_cache:
                cache_write_t0 = time.perf_counter()
                _write_cached_units(cache_key, units)
                _log_timing('load_units.cache_write', cache_write_t0)
            _log_timing('load_units.done_python_local', t0, count=len(units))
            return units

        import panedr

        if hasattr(panedr, 'get_unit_dictionary'):
            read_t0 = time.perf_counter()
            units = panedr.get_unit_dictionary(file_path)
            _log_timing('load_units.panedr_read', read_t0, count=len(units))
            if use_cache:
                cache_write_t0 = time.perf_counter()
                _write_cached_units(cache_key, units)
                _log_timing('load_units.cache_write', cache_write_t0)
            _log_timing('load_units.done_panedr', t0, count=len(units))
            return units
    except Exception:
        _log_timing('load_units.failed', t0)
        return {}
    _log_timing('load_units.empty', t0)
    return {}


def _get_rust_edr_module():
    global _RUST_EDR_MODULE
    if _RUST_EDR_MODULE is not None:
        return _RUST_EDR_MODULE

    if os.environ.get('EDTERM_DISABLE_RUST_READER', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        return None

    try:
        import edterm_rust_ext

        _RUST_EDR_MODULE = edterm_rust_ext
        return _RUST_EDR_MODULE
    except Exception:
        return None


def _notify_reader_selected(callback, reader_name):
    if callback is None:
        return
    try:
        callback(str(reader_name))
    except Exception:
        return


def _log_timing(event, started_at, **fields):
    try:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        extras = ' '.join(f'{k}={v}' for k, v in fields.items())
        message = f'[timing] {event} {elapsed_ms:.1f}ms'
        if extras:
            message = f'{message} {extras}'
        logger.info(message)
    except Exception:
        return


def _cache_key(file_path, frame_stride):
    stat = os.stat(file_path)
    payload = (
        f'{PARSER_CACHE_VERSION}|{file_path}|{stat.st_size}|{stat.st_mtime_ns}|'
        f'{frame_stride}|{LOCAL_PYEDR_FILE.exists()}'
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _df_cache_path(cache_key):
    return CACHE_DIR / f'{cache_key}.df.pkl'


def _units_cache_path(cache_key):
    return CACHE_DIR / f'{cache_key}.units.pkl'


def _read_cached_df(cache_key):
    path = _df_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        if _is_dataset(data):
            return data
    except Exception:
        return None
    return None


def _write_cached_df(cache_key, data):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _prune_cache_dir()
        with open(_df_cache_path(cache_key), 'wb') as fh:
            pickle.dump(data, fh)
    except Exception:
        return


def _read_cached_units(cache_key):
    path = _units_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _write_cached_units(cache_key, units):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _prune_cache_dir()
        with open(_units_cache_path(cache_key), 'wb') as fh:
            pickle.dump(units, fh)
    except Exception:
        return


def _prune_cache_dir():
    try:
        files = [p for p in CACHE_DIR.glob('*.pkl') if p.is_file()]
    except Exception:
        return

    if not files:
        return

    try:
        files_with_stat = [(p, p.stat()) for p in files]
    except Exception:
        return

    total_size = sum(st.st_size for _, st in files_with_stat)
    if len(files_with_stat) <= MAX_CACHE_FILES and total_size <= MAX_CACHE_BYTES:
        return

    files_with_stat.sort(key=lambda item: item[1].st_mtime)
    remaining_files = len(files_with_stat)
    for path, st in files_with_stat:
        if remaining_files <= MAX_CACHE_FILES and total_size <= MAX_CACHE_BYTES:
            break
        try:
            path.unlink(missing_ok=True)
            total_size -= st.st_size
            remaining_files -= 1
        except Exception:
            continue


def _empty_dataset(all_names):
    columns = list(all_names[1:]) if len(all_names) > 1 else []
    return {
        'time': np.array([], dtype=np.float64),
        'columns': columns,
        'values': {col: np.array([], dtype=np.float64) for col in columns},
    }


def _dataset_from_matrix(matrix, all_names):
    if matrix.size == 0 or matrix.shape[1] == 0:
        return _empty_dataset(all_names)
    columns = list(all_names[1:])
    values = {}
    for idx, col in enumerate(columns, start=1):
        values[col] = np.asarray(matrix[:, idx], dtype=np.float64)
    return {
        'time': np.asarray(matrix[:, 0], dtype=np.float64),
        'columns': columns,
        'values': values,
    }


def _dataset_from_rows(all_energies, all_names):
    if not all_energies:
        return _empty_dataset(all_names)
    matrix = np.asarray(all_energies, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return _dataset_from_matrix(matrix, all_names)


def _dataset_from_pandas(df):
    all_names = list(df.columns)
    if not all_names:
        return _empty_dataset([])
    matrix = df.to_numpy(dtype=np.float64, copy=False)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return _dataset_from_matrix(matrix, all_names)


def _is_dataset(obj):
    if not isinstance(obj, dict):
        return False
    if 'time' not in obj or 'columns' not in obj or 'values' not in obj:
        return False
    return True
