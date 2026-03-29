import argparse
import curses
import locale
import logging
import os
import queue
import re
import sys
import threading
import time
from collections import deque

import numpy as np

from .analysis import (
    TIME_UNIT_CYCLE,
    _compute_trend_for_column,
    _format_stats,
    range_has_data,
    visible_mask_for_range,
)
from .data_reader import load_data, load_units, stream_data
from .plotting import plot_ascii, plot_histogram
from .ui.render import (
    _clear_region,
    _safe_addstr,
    draw_overview,
    parse_and_print_ansi,
    render_loading_box,
    setup_colors,
)


logger = logging.getLogger(__name__)
logger_handler = logging.FileHandler('.edterm_debug.log')
logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_handler)

PERCENT_RE = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')


def setup_environment():
    preferred_locale = 'en_US.UTF-8'

    os.environ.setdefault('LANG', preferred_locale)
    os.environ.setdefault('LC_ALL', preferred_locale)

    try:
        locale.setlocale(locale.LC_ALL, preferred_locale)
        return
    except locale.Error:
        logger.warning("Locale '%s' is not available. Falling back to system default locale.", preferred_locale)

    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        logger.warning('Failed to configure locale. Continuing with process defaults.')


def setup_logger(logger_level='info'):
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    log_level = level_dict.get(str(logger_level).lower(), logging.INFO)
    logger.setLevel(log_level)
    logger_handler.setLevel(log_level)


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be an integer') from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError('must be >= 1')

    return parsed


def validate_loaded_dataframe(df):
    if df is None:
        return 'No data was loaded from the EDR file.'

    if not isinstance(df, dict):
        return 'Invalid data format loaded from reader.'

    if 'time' not in df or 'columns' not in df or 'values' not in df:
        return "EDR data is missing required 'Time' column."

    time_values = np.asarray(df.get('time', np.array([], dtype=float)))
    if time_values.size == 0:
        return 'EDR data is empty or could not be parsed.'

    observable_columns = list(df.get('columns', []))
    if not observable_columns:
        return 'No observable columns were found in the EDR data.'

    values = df.get('values', {})
    for column in observable_columns:
        arr = np.asarray(values.get(column, np.array([], dtype=float)))
        if arr.shape[0] != time_values.shape[0]:
            return f"Column '{column}' has inconsistent length."

    return None


def _get_or_compute_trend(df, column, trend_cache, trend_cache_lock):
    expected_len = int(np.asarray(df.get('time', np.array([], dtype=np.float64))).shape[0])
    with trend_cache_lock:
        cached = trend_cache.get(column)
    if cached is not None and int(np.asarray(cached).shape[0]) == expected_len:
        return cached

    trend = _compute_trend_for_column(df, column)
    with trend_cache_lock:
        existing = trend_cache.get(column)
        if existing is not None and int(np.asarray(existing).shape[0]) == expected_len:
            return existing
        trend_cache[column] = trend
    return trend


class TrendPrecomputeWorker:
    def __init__(self, df, columns, trend_cache, trend_cache_lock):
        self._df = df
        self._columns = list(columns)
        self._trend_cache = trend_cache
        self._trend_cache_lock = trend_cache_lock
        self._queue = deque()
        self._queued = set()
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self.prioritize(self._columns)
        self._thread = threading.Thread(target=self._run, name='trend-precompute', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def prioritize(self, columns):
        if not columns:
            return
        with self._queue_lock:
            for col in reversed(list(columns)):
                if col in self._queued:
                    continue
                self._queue.appendleft(col)
                self._queued.add(col)

    def _pop_next(self):
        with self._queue_lock:
            if not self._queue:
                return None
            col = self._queue.popleft()
            self._queued.discard(col)
            return col

    def _run(self):
        while not self._stop_event.is_set():
            col = self._pop_next()
            if col is None:
                time.sleep(0.05)
                continue
            with self._trend_cache_lock:
                if col in self._trend_cache:
                    continue
            try:
                trend = _compute_trend_for_column(self._df, col)
                with self._trend_cache_lock:
                    if col not in self._trend_cache:
                        self._trend_cache[col] = trend
            except Exception:
                logger.exception("Background trend precompute failed for column '%s'", col)


class ProgressBuffer:
    def __init__(self):
        self._latest_line = ''
        self._latest_percent = None
        self._current_line = ''
        self._bytes_read = 0
        self._total_bytes = 0
        self._records_read = 0
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return
        chunk = str(text)
        with self._lock:
            for char in chunk:
                if char in ('\r', '\n'):
                    line = self._current_line.strip()
                    if line:
                        self._latest_line = line
                        match = PERCENT_RE.search(line)
                        if match:
                            try:
                                pct = float(match.group(1))
                                self._latest_percent = max(0.0, min(100.0, pct))
                            except ValueError:
                                pass
                    self._current_line = ''
                else:
                    if len(self._current_line) >= 512:
                        self._current_line = self._current_line[-256:]
                    self._current_line += char

    def flush(self):
        return

    def snapshot(self):
        with self._lock:
            line = self._latest_line
            current = self._current_line.strip()
            if current:
                line = current
                match = PERCENT_RE.search(current)
                if match:
                    try:
                        pct = float(match.group(1))
                        self._latest_percent = max(0.0, min(100.0, pct))
                    except ValueError:
                        pass
            if self._total_bytes > 0:
                bytes_pct = max(0.0, min(100.0, (100.0 * self._bytes_read) / self._total_bytes))
                if self._latest_percent is None:
                    self._latest_percent = bytes_pct
                line = f'{self._records_read} frames, {self._bytes_read}/{self._total_bytes} bytes'
            return self._latest_percent, line

    def update_bytes(self, bytes_read, total_bytes, records_read):
        with self._lock:
            self._bytes_read = max(0, int(bytes_read))
            self._total_bytes = max(0, int(total_bytes))
            self._records_read = max(0, int(records_read))
            if self._total_bytes > 0:
                self._latest_percent = max(
                    0.0,
                    min(100.0, (100.0 * self._bytes_read) / self._total_bytes),
                )


def _log_ui_timing(event, started_at, **fields):
    try:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        extras = ' '.join(f'{k}={v}' for k, v in fields.items())
        message = f'[timing] {event} {elapsed_ms:.1f}ms'
        if extras:
            message = f'{message} {extras}'
        logger.debug(message)
    except Exception:
        return


def edterm_main(stdscr, args, preloaded_df=None):
    curses.curs_set(0)
    setup_colors(args.theme)
    try:
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    except curses.error:
        pass
    stdscr.nodelay(1)
    stdscr.clear()

    max_y, max_x = stdscr.getmaxyx()
    menu_width = 20

    data_ready = preloaded_df is not None
    df = preloaded_df
    units = getattr(args, '_preloaded_units', {}) if data_ready else {}
    columns = list(df.get('columns', [])) if data_ready else []

    streaming_mode = bool(getattr(args, 'stream_load', False) and not data_ready)
    stream_queue = None
    stream_thread = None
    stream_stop_event = None
    stream_complete = False
    stream_error = None
    stream_records = 0
    stream_total_bytes = 0
    stream_reader = 'python-local'

    if not data_ready and not streaming_mode:
        progress_buffer = ProgressBuffer()
        loading_started_at = time.time()
        last_draw_time = [0.0]
        loading_reader = ['auto']

        def draw_loading_snapshot(force=False):
            now = time.time()
            if not force and now - last_draw_time[0] < 0.12:
                return
            last_draw_time[0] = now

            _safe_addstr(stdscr, 0, 0, 'Welcome to the GROMACS Data Plotter Tool')
            _safe_addstr(stdscr, 1, 0, f'File: {args.file}')
            _safe_addstr(stdscr, 2, 0, "Loading mode. Keyboard input is disabled during parsing.")
            for y in range(4, max_y - 1):
                try:
                    stdscr.addch(y, menu_width, curses.ACS_VLINE)
                except curses.error:
                    break
            elapsed = now - loading_started_at
            progress_pct, progress_line = progress_buffer.snapshot()
            render_loading_box(stdscr, menu_width, max_x, max_y, elapsed, progress_pct, progress_line)
            _safe_addstr(stdscr, 3, 0, ' ' * max(0, max_x - 1))
            loading_line = (
                f'Reader: {loading_reader[0]} | '
                'Auto-stride mode=minmax envelope downsampling.'
            )
            _safe_addstr(stdscr, 3, 0, loading_line[: max(0, max_x - 1)])
            stdscr.noutrefresh()
            curses.doupdate()

        def progress_callback(bytes_read, total_bytes, records_read):
            progress_buffer.update_bytes(bytes_read, total_bytes, records_read)
            draw_loading_snapshot(force=False)

        def reader_selected_callback(reader_name):
            loading_reader[0] = reader_name
            draw_loading_snapshot(force=True)

        draw_loading_snapshot(force=True)
        load_t0 = time.perf_counter()
        loaded_df = load_data(
            args.file,
            verbose=False,
            stderr_sink=None,
            progress_callback=progress_callback,
            frame_stride=args.stride,
            use_cache=not args.no_cache,
            reader_selected_callback=reader_selected_callback,
        )
        logger.info('[timing] ui.load_data %.1fms', (time.perf_counter() - load_t0) * 1000.0)
        units_t0 = time.perf_counter()
        loaded_units = load_units(args.file, use_cache=not args.no_cache)
        logger.info('[timing] ui.load_units %.1fms', (time.perf_counter() - units_t0) * 1000.0)
        draw_loading_snapshot(force=True)
        validation_error = validate_loaded_dataframe(loaded_df)
        if validation_error:
            load_error_message = f'{validation_error} Press q to quit.'
            status_message = load_error_message
            plot_info_message = ''
        else:
            df = loaded_df
            units = loaded_units if isinstance(loaded_units, dict) else {}
            columns = list(df.get('columns', []))
            data_ready = True
            resize_happened = True
            range_changed = True
            status_message = ''
    elif streaming_mode:
        # Initialize empty dataset and start background streaming parser.
        df = {'time': np.array([], dtype=np.float64), 'columns': [], 'values': {}}
        units = {}
        columns = []
        stream_queue = queue.Queue()
        stream_stop_event = threading.Event()

        def _on_metadata(cols, unit_dict, total_bytes):
            stream_queue.put(('meta', cols, unit_dict, int(total_bytes)))

        def _on_batch(batch_time, batch_values, kept_count, bytes_read, total_bytes):
            stream_queue.put(('batch', batch_time, batch_values, int(kept_count), int(bytes_read), int(total_bytes)))

        def _on_progress(bytes_read, total_bytes, records_read):
            stream_queue.put(('progress', int(bytes_read), int(total_bytes), int(records_read)))

        def _on_complete(error_text):
            stream_queue.put(('done', error_text))

        def _run_stream():
            stream_data(
                args.file,
                frame_stride=args.stride,
                batch_size=200,
                progress_stride=500,
                on_metadata=_on_metadata,
                on_batch=_on_batch,
                on_progress=_on_progress,
                on_complete=_on_complete,
                stop_event=stream_stop_event,
            )

        stream_thread = threading.Thread(target=_run_stream, name='edr-stream-loader', daemon=True)
        stream_thread.start()

    current_index = 0
    last_index = -1
    last_number_time = time.time()
    input_mode = None
    resize_happened = True
    range_changed = True
    x_min, x_max = None, None
    number_buffer = ''
    status_message = ''
    plot_info_message = ''
    load_error_message = ''
    overview_page = 0
    overview_total_pages = 1
    current_stride = args.stride
    time_unit_mode = 'auto'
    header_expanded = False
    current_stats = None
    view_mode = 'time'
    trend_cache = {}
    trend_cache_lock = threading.Lock()
    trend_worker = None

    if data_ready and columns:
        trend_worker = TrendPrecomputeWorker(df, columns, trend_cache, trend_cache_lock)
        trend_worker.start()

    try:
        while True:
            if streaming_mode and stream_queue is not None:
                processed_messages = 0
                max_messages_per_tick = 32
                while processed_messages < max_messages_per_tick:
                    try:
                        message = stream_queue.get_nowait()
                    except queue.Empty:
                        break
                    processed_messages += 1

                    tag = message[0]
                    if tag == 'meta':
                        _, cols, unit_dict, total_bytes = message
                        columns = list(cols)
                        units = dict(unit_dict) if isinstance(unit_dict, dict) else {}
                        df['columns'] = columns
                        if not df['values']:
                            df['values'] = {col: np.array([], dtype=np.float64) for col in columns}
                        stream_total_bytes = int(total_bytes)
                        data_ready = True
                        range_changed = True
                    elif tag == 'batch':
                        _, batch_time, batch_values, kept_count, bytes_read, total_bytes = message
                        if batch_time is not None and len(batch_time) > 0:
                            batch_time = np.asarray(batch_time, dtype=np.float64)
                            df['time'] = np.concatenate([np.asarray(df['time'], dtype=np.float64), batch_time])
                            for col in columns:
                                existing = np.asarray(df['values'].get(col, np.array([], dtype=np.float64)), dtype=np.float64)
                                incoming = np.asarray(batch_values.get(col, np.array([], dtype=np.float64)), dtype=np.float64)
                                df['values'][col] = np.concatenate([existing, incoming])
                            with trend_cache_lock:
                                trend_cache.clear()
                            stream_records = int(kept_count)
                            stream_total_bytes = int(total_bytes)
                            range_changed = True
                        if stream_total_bytes > 0:
                            plot_info_message = (
                                f'Streaming {stream_records} frames '
                                f'({bytes_read}/{stream_total_bytes} bytes) reader={stream_reader}'
                            )
                    elif tag == 'progress':
                        _, bytes_read, total_bytes, records_read = message
                        stream_records = int(records_read)
                        stream_total_bytes = int(total_bytes)
                        if stream_total_bytes > 0:
                            plot_info_message = (
                                f'Streaming {stream_records} frames '
                                f'({bytes_read}/{stream_total_bytes} bytes) reader={stream_reader}'
                            )
                        # Keep UI responsive during long parse gaps between batches.
                        range_changed = True
                    elif tag == 'done':
                        _, error_text = message
                        stream_complete = True
                        stream_error = error_text
                        if error_text and error_text != 'stopped':
                            status_message = f'Streaming failed: {error_text}'
                        else:
                            status_message = f'Stream complete: {len(df["time"])} frames loaded.'
                        range_changed = True

                if stream_complete and trend_worker is None and columns:
                    trend_worker = TrendPrecomputeWorker(df, columns, trend_cache, trend_cache_lock)
                    trend_worker.start()

            if input_mode == 'range':
                stdscr.nodelay(0)
                _safe_addstr(stdscr, max_y - 1, 0, 'Provide the desired time window (x_min x_max): ')
                stdscr.clrtoeol()
                curses.echo()
                try:
                    input_raw = stdscr.getstr(max_y - 1, 50)
                    input_str = input_raw.decode('utf-8').strip()
                except Exception:
                    input_str = ''

                previous_x_min, previous_x_max = x_min, x_max
                try:
                    new_x_min, new_x_max = map(float, input_str.split())
                    if new_x_min > new_x_max:
                        raise ValueError('x_min must be <= x_max')

                    if data_ready and range_has_data(df, new_x_min, new_x_max):
                        x_min, x_max = new_x_min, new_x_max
                        range_changed = True
                        status_message = f'Time window set to: {x_min} - {x_max}'
                    else:
                        x_min, x_max = previous_x_min, previous_x_max
                        status_message = 'No data in selected range. Keeping previous range.'
                except ValueError:
                    x_min, x_max = previous_x_min, previous_x_max
                    status_message = 'Invalid input. Enter two numbers with x_min <= x_max.'

                curses.noecho()
                input_mode = None
                stdscr.nodelay(1)
            elif input_mode == 'stride':
                stdscr.nodelay(0)
                _safe_addstr(stdscr, max_y - 1, 0, f'Provide stride (>=1), current {current_stride}: ')
                stdscr.clrtoeol()
                curses.echo()
                try:
                    input_raw = stdscr.getstr(max_y - 1, 45)
                    input_str = input_raw.decode('utf-8').strip()
                except Exception:
                    input_str = ''
                try:
                    new_stride = int(input_str)
                    if new_stride < 1:
                        raise ValueError('Stride must be >=1')
                    current_stride = new_stride
                    range_changed = True
                    status_message = f'Stride set to {current_stride}'
                except ValueError:
                    status_message = 'Invalid stride. Enter an integer >= 1.'
                curses.noecho()
                input_mode = None
                stdscr.nodelay(1)
            new_max_y, new_max_x = stdscr.getmaxyx()
            if new_max_y != max_y or new_max_x != max_x:
                max_y, max_x = new_max_y, new_max_x
                resize_happened = True
                stdscr.clear()
                stdscr.refresh()

            footer_sep_row = max(1, max_y - 2)
            footer_row = max(0, max_y - 1)

            auto_mode = 'off' if args.no_auto_stride else 'on'
            short_name = os.path.basename(args.file)
            if header_expanded:
                _safe_addstr(stdscr, 0, 0, ' ' * max(0, max_x - 1))
                _safe_addstr(stdscr, 1, 0, ' ' * max(0, max_x - 1))
                _safe_addstr(stdscr, 2, 0, ' ' * max(0, max_x - 1))
                _safe_addstr(stdscr, 3, 0, ' ' * max(0, max_x - 1))
                _safe_addstr(stdscr, 0, 0, 'Welcome to the GROMACS Data Plotter Tool'[: max(0, max_x - 1)])
                _safe_addstr(stdscr, 1, 0, f'File: {args.file}'[: max(0, max_x - 1)])
                _safe_addstr(
                    stdscr,
                    2,
                    0,
                    "q quit | arrows select | r range | s stride | u unit | h view | i header | L/R overview pages"[: max(0, max_x - 1)],
                )
                detail_line = (
                    f"unit={time_unit_mode} stride={current_stride} auto={auto_mode} view={view_mode} "
                    f"stats={_format_stats(current_stats)}"
                )
                _safe_addstr(stdscr, 3, 0, detail_line[: max(0, max_x - 1)])
                _safe_addstr(stdscr, 0, max(0, max_x - 26), '[i/click: collapse]')
                for x in range(max_x):
                    try:
                        stdscr.addch(4, x, curses.ACS_HLINE)
                    except curses.error:
                        break
                header_plot_info_row = 3
                content_top = 5
            else:
                _safe_addstr(stdscr, 0, 0, ' ' * max(0, max_x - 1))
                compact_line = (
                    f"{short_name} | unit={time_unit_mode} stride={current_stride} auto={auto_mode} view={view_mode} "
                    f"| {_format_stats(current_stats)} | [i/click: expand]"
                )
                _safe_addstr(stdscr, 0, 0, compact_line[: max(0, max_x - 1)])
                for x in range(max_x):
                    try:
                        stdscr.addch(1, x, curses.ACS_HLINE)
                    except curses.error:
                        break
                header_plot_info_row = None
                content_top = 2

            for x in range(max_x):
                try:
                    stdscr.addch(footer_sep_row, x, curses.ACS_HLINE)
                except curses.error:
                    break

            menu_row = content_top
            if menu_row < max_y:
                overview_mode = curses.A_REVERSE if current_index == 0 else curses.A_NORMAL
                _safe_addstr(stdscr, menu_row, 0, '0. Overview'.ljust(menu_width), overview_mode)
            for i, col in enumerate(columns):
                row = menu_row + 1 + i
                if row < max_y:
                    mode = curses.A_REVERSE if (i + 1) == current_index else curses.A_NORMAL
                    _safe_addstr(stdscr, row, 0, f'{i + 1}. {col}'.ljust(menu_width), mode)

            for y in range(content_top, footer_sep_row):
                try:
                    stdscr.addch(y, menu_width, curses.ACS_VLINE)
                except curses.error:
                    break

            should_redraw_plot = (
                current_index != last_index
                or resize_happened
                or range_changed
            )

            if should_redraw_plot:
                redraw_t0 = time.perf_counter()
                last_index = current_index
                plot_width = max_x - menu_width - 5
                content_bottom = footer_sep_row
                plot_height = content_bottom - content_top
                _clear_region(stdscr, content_top, menu_width + 1, max(0, content_bottom - content_top), max(0, max_x - menu_width - 1))
                visible_t0 = time.perf_counter()
                if data_ready:
                    visible_mask = visible_mask_for_range(np.asarray(df['time'], dtype=np.float64), x_min, x_max)
                    visible_rows = int(np.count_nonzero(visible_mask))
                else:
                    visible_mask = np.array([], dtype=bool)
                    visible_rows = 0
                _log_ui_timing(
                    'ui.visible_df',
                    visible_t0,
                    index=current_index,
                    rows=visible_rows,
                )

                if not columns:
                    plot_info_message = ''
                    if load_error_message:
                        status_message = load_error_message
                    else:
                        status_message = 'No data columns available to plot.'
                elif plot_width <= 0 or plot_height <= 0:
                    plot_info_message = ''
                    status_message = 'Terminal is too small to draw plot area.'
                elif not data_ready:
                    plot_info_message = ''
                    status_message = 'Data not available for plotting.'
                else:
                    try:
                        if current_index == 0:
                            overview_t0 = time.perf_counter()
                            if trend_worker is not None:
                                start_idx = overview_page * 6
                                trend_worker.prioritize(columns[start_idx:start_idx + 6])
                            plot_info_message, _, overview_total_pages = draw_overview(
                                stdscr,
                                df,
                                columns,
                                units,
                                menu_width,
                                max_x,
                                max_y,
                                content_top,
                                content_bottom,
                                x_min,
                                x_max,
                                current_stride,
                                not args.no_auto_stride,
                                overview_page,
                                args.theme,
                                time_unit_mode,
                                visible_mask=visible_mask,
                                trend_cache=trend_cache,
                                trend_cache_lock=trend_cache_lock,
                                trend_getter=_get_or_compute_trend,
                                trend_computer=_compute_trend_for_column,
                            )
                            _log_ui_timing(
                                'ui.draw_overview',
                                overview_t0,
                                page=overview_page,
                                shown=min(6, max(0, len(columns) - overview_page * 6)),
                            )
                            current_stats = None
                        else:
                            selected_column = columns[current_index - 1]
                            trend_t0 = time.perf_counter()
                            trend_series = _get_or_compute_trend(df, selected_column, trend_cache, trend_cache_lock)
                            _log_ui_timing('ui.trend_lookup', trend_t0, column=selected_column)
                            if trend_worker is not None:
                                near = [selected_column]
                                prev_idx = current_index - 2
                                next_idx = current_index
                                if prev_idx >= 0:
                                    near.append(columns[prev_idx])
                                if next_idx < len(columns):
                                    near.append(columns[next_idx])
                                trend_worker.prioritize(near)

                            plot_build_t0 = time.perf_counter()
                            if view_mode == 'hist':
                                plot_lines, plotted_points, bins, sampling_mode, stats = plot_histogram(
                                    df,
                                    selected_column,
                                    units,
                                    plot_width,
                                    plot_height,
                                    x_min,
                                    x_max,
                                    visible_mask=visible_mask,
                                )
                                effective_stride = current_stride
                                total_visible_points = visible_rows
                            else:
                                plot_lines, effective_stride, plotted_points, total_visible_points, sampling_mode, stats = plot_ascii(
                                    df,
                                    selected_column,
                                    units,
                                    plot_width,
                                    plot_height,
                                    x_min,
                                    x_max,
                                    current_stride,
                                    not args.no_auto_stride,
                                    time_unit_mode,
                                    visible_mask=visible_mask,
                                    trend_series=trend_series,
                                )
                            _log_ui_timing(
                                'ui.plot_ascii',
                                plot_build_t0,
                                column=selected_column,
                                plotted=plotted_points,
                                total=total_visible_points,
                                mode=sampling_mode,
                            )
                            paint_t0 = time.perf_counter()
                            plot_row = content_top
                            for line in plot_lines:
                                if plot_row < content_bottom:
                                    parse_and_print_ansi(stdscr, plot_row, menu_width + 4, line, args.theme)
                                    plot_row += 1
                            _log_ui_timing(
                                'ui.paint_plot',
                                paint_t0,
                                column=selected_column,
                                lines=(plot_row - content_top),
                            )
                            current_stats = stats
                            if view_mode == 'hist':
                                normal_label = ''
                                if stats and 'jb_p' in stats:
                                    normal_label = f", jb_p={stats['jb_p']:.3g}, gaussian≈{'yes' if stats.get('gaussian_like') else 'no'}"
                                plot_info_message = (
                                    f'Histogram {plotted_points} samples '
                                    f'(bins={bins}, unit={time_unit_mode}{normal_label})'
                                )
                            else:
                                plot_info_message = (
                                    f'Rendering {plotted_points}/{total_visible_points} points '
                                    f'(mode={sampling_mode}, stride={effective_stride}, set={current_stride}, auto={auto_mode}, unit={time_unit_mode})'
                                )
                        status_message = ''
                    except Exception:
                        logger.exception('Failed to render plot for selection %s', current_index)
                        plot_info_message = ''
                        status_message = 'Plot rendering failed. See .edterm_debug.log for details.'
                _log_ui_timing('ui.redraw_total', redraw_t0, index=current_index, width=plot_width, height=plot_height)

                resize_happened = False
                range_changed = False

            if status_message:
                stdscr.move(footer_row, 0)
                stdscr.clrtoeol()
                _safe_addstr(stdscr, footer_row, 0, status_message[: max(0, max_x - 1)])

            if header_plot_info_row is not None:
                _safe_addstr(stdscr, header_plot_info_row, 0, ' ' * max(0, max_x - 1))
                if plot_info_message:
                    _safe_addstr(stdscr, header_plot_info_row, 0, plot_info_message[: max(0, max_x - 1)])
            elif plot_info_message and not status_message:
                stdscr.move(footer_row, 0)
                stdscr.clrtoeol()
                _safe_addstr(stdscr, footer_row, 0, plot_info_message[: max(0, max_x - 1)])

            stdscr.noutrefresh()
            curses.doupdate()

            try:
                k = stdscr.getch()
            except Exception:
                continue

            if k != -1 and 0 <= k < 256:
                try:
                    char = chr(k)
                except ValueError:
                    char = ''

                if time.time() - last_number_time > 1.0:
                    number_buffer = ''

                if char.isdigit():
                    number_buffer += char
                    last_number_time = time.time()
                    number = int(number_buffer)
                    if 0 <= number <= len(columns):
                        current_index = number
                elif char == 'q':
                    break
                elif char == 'r' and data_ready:
                    input_mode = 'range'
                elif char == 's' and data_ready:
                    input_mode = 'stride'
                elif char == 'u':
                    current_idx = TIME_UNIT_CYCLE.index(time_unit_mode) if time_unit_mode in TIME_UNIT_CYCLE else 0
                    time_unit_mode = TIME_UNIT_CYCLE[(current_idx + 1) % len(TIME_UNIT_CYCLE)]
                    range_changed = True
                    status_message = f'Time unit mode: {time_unit_mode}'
                elif char == 'h':
                    view_mode = 'hist' if view_mode == 'time' else 'time'
                    range_changed = True
                    status_message = f'View mode: {view_mode}'
                elif char == 'i':
                    header_expanded = not header_expanded
                    range_changed = True

            if k == curses.KEY_MOUSE:
                try:
                    _, mx, my, _, bstate = curses.getmouse()
                except Exception:
                    bstate = 0
                    mx, my = -1, -1

                left_click = bool(
                    bstate
                    & (
                        getattr(curses, 'BUTTON1_CLICKED', 0)
                        | getattr(curses, 'BUTTON1_PRESSED', 0)
                        | getattr(curses, 'BUTTON1_RELEASED', 0)
                    )
                )
                if left_click:
                    menu_row = content_top
                    # Header click toggles expanded/collapsed top section.
                    header_click_row = 0 if not header_expanded else 1
                    if my == header_click_row and 0 <= mx < max_x:
                        header_expanded = not header_expanded
                        range_changed = True
                        status_message = 'Header expanded.' if header_expanded else 'Header collapsed.'
                    # Left menu click selects overview/observable.
                    elif 0 <= mx < menu_width and menu_row <= my < (menu_row + 1 + len(columns)):
                        selected = my - menu_row
                        if 0 <= selected <= len(columns):
                            current_index = selected
                            range_changed = True
                    # Overview page click zones on plot area edges.
                    elif current_index == 0 and content_top <= my < footer_sep_row and mx > menu_width:
                        left_zone = menu_width + 1
                        right_zone = max_x - 8
                        if mx <= left_zone + 2 and overview_page > 0:
                            overview_page -= 1
                            range_changed = True
                        elif mx >= right_zone and overview_page < (overview_total_pages - 1):
                            overview_page += 1
                            range_changed = True

            if k == curses.KEY_UP and current_index > 0:
                current_index -= 1
                if current_index == 0:
                    range_changed = True
            elif k == curses.KEY_DOWN and current_index < len(columns):
                current_index += 1
            elif k == curses.KEY_LEFT and current_index == 0 and overview_page > 0:
                overview_page -= 1
                range_changed = True
            elif k == curses.KEY_RIGHT and current_index == 0 and overview_page < (overview_total_pages - 1):
                overview_page += 1
                range_changed = True
    finally:
        if stream_stop_event is not None:
            stream_stop_event.set()
        if stream_thread is not None and stream_thread.is_alive():
            stream_thread.join(timeout=0.5)
        if trend_worker is not None:
            trend_worker.stop()


def main():
    parser = argparse.ArgumentParser(description='GROMACS Data Plotter Tool')
    parser.add_argument('file', type=str, help='Path to the GROMACS EDR file')
    parser.add_argument(
        '--stride',
        '-s',
        type=positive_int,
        default=1,
        help='Stride for data plotting to reduce plot density (must be >= 1)',
    )
    parser.add_argument(
        '--logging-level',
        '-ll',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Set logger level',
    )
    parser.add_argument(
        '--theme',
        '-t',
        type=str,
        default='transparent',
        choices=['dark', 'light', 'transparent'],
        help='Set the color theme',
    )
    parser.add_argument(
        '--no-auto-stride',
        action='store_true',
        help='Disable automatic terminal-aware downsampling and use only the requested stride.',
    )
    parser.add_argument(
        '--load-progress',
        action='store_true',
        help='Show parser-provided loading progress (can be slower on very large files).',
    )
    parser.add_argument(
        '--stream-load',
        action='store_true',
        help='Experimental: progressively stream frames and render partial plots while loading.',
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable disk cache reads/writes for this run.',
    )

    args = parser.parse_args()

    setup_logger(args.logging_level)
    setup_environment()

    if not os.path.isfile(args.file):
        print(f"Error: file '{args.file}' does not exist.", file=sys.stderr)
        return 1

    try:
        if args.load_progress or args.stream_load:
            curses.wrapper(edterm_main, args, None)
        else:
            load_t0 = time.perf_counter()
            loaded_df = load_data(
                args.file,
                verbose=False,
                stderr_sink=None,
                frame_stride=args.stride,
                use_cache=not args.no_cache,
            )
            logger.info('[timing] main.load_data %.1fms', (time.perf_counter() - load_t0) * 1000.0)
            units_t0 = time.perf_counter()
            loaded_units = load_units(args.file, use_cache=not args.no_cache)
            logger.info('[timing] main.load_units %.1fms', (time.perf_counter() - units_t0) * 1000.0)
            validation_error = validate_loaded_dataframe(loaded_df)
            if validation_error:
                print(f'Error: {validation_error}', file=sys.stderr)
                return 1
            # Store preloaded units in args for initial non-progress startup.
            args._preloaded_units = loaded_units if isinstance(loaded_units, dict) else {}
            curses.wrapper(edterm_main, args, loaded_df)
    except Exception:
        logger.exception('Fatal UI error')
        print('Error: terminal UI failed. See .edterm_debug.log for details.', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
