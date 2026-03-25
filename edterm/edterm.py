import argparse
import curses
import locale
import logging
import math
import os
import re
import sys
import threading
import time

import pandas as pd
import plotext

from .data_reader import load_data, load_units


logger = logging.getLogger(__name__)
logger_handler = logging.FileHandler('.edterm_debug.log')
logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_handler)

ANSI_ESCAPE = re.compile(r'\x1b\[([0-9;]*)m')
PERCENT_RE = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')
AUTO_STRIDE_OVERSAMPLE_FACTOR = 3
TIME_UNIT_FACTORS_PS = {
    'fs': 0.001,
    'ps': 1.0,
    'ns': 1000.0,
    'us': 1_000_000.0,
}
TIME_UNIT_CYCLE = ['auto', 'fs', 'ps', 'ns', 'us']


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

    if df.empty:
        return 'EDR data is empty or could not be parsed.'

    if 'Time' not in df.columns:
        return "EDR data is missing required 'Time' column."

    observable_columns = list(df.columns[1:])
    if not observable_columns:
        return 'No observable columns were found in the EDR data.'

    return None


def setup_colors(theme):
    curses.start_color()
    curses.use_default_colors()

    background_color = -1
    if theme == 'dark':
        background_color = 16
    elif theme == 'light':
        background_color = 15

    max_colors = max(0, getattr(curses, 'COLORS', 0) - 1)
    max_pairs = max(0, getattr(curses, 'COLOR_PAIRS', 0) - 1)
    init_limit = min(max_colors, max_pairs)

    for i in range(1, init_limit + 1):
        try:
            curses.init_pair(i, i, background_color)
        except curses.error:
            break


def _safe_color_pair(color_id):
    try:
        return curses.color_pair(color_id)
    except curses.error:
        return curses.color_pair(0)


def _theme_color_ids(theme):
    if theme == 'dark':
        return {
            'default': 15,
            'line_primary': 15,
            'line_secondary': 156,
            'axis': 156,
        }
    if theme == 'light':
        return {
            'default': 232,
            'line_primary': 4,
            'line_secondary': 156,
            'axis': 8,
        }
    return {
        'default': 0,
        'line_primary': 11,
        'line_secondary': 15,
        'axis': 15,
    }


def _safe_addstr(stdscr, y, x, text, color_pair=None):
    try:
        if color_pair is None:
            stdscr.addstr(y, x, text)
        else:
            stdscr.addstr(y, x, text, color_pair)
    except curses.error:
        return


def _clear_region(stdscr, top, left, height, width):
    if height <= 0 or width <= 0:
        return
    blank = ' ' * width
    for y in range(top, top + height):
        _safe_addstr(stdscr, y, left, blank)


def _color_for_ansi_code(code, theme_color_ids, default_pair):
    if not code:
        return default_pair

    if code.startswith('38;5;0'):
        return _safe_color_pair(theme_color_ids['default'])
    if code == '38;5;12':
        return _safe_color_pair(theme_color_ids['line_primary'])
    if code.startswith('48;5;'):
        return _safe_color_pair(theme_color_ids['line_secondary'])
    if code == '38;5;10':
        return _safe_color_pair(theme_color_ids['axis'])

    return default_pair


def parse_and_print_ansi(stdscr, y, x, ansi_string, theme, extra_attr=0):
    theme_color_ids = _theme_color_ids(theme)
    default_pair = _safe_color_pair(theme_color_ids['default'])

    pieces = ANSI_ESCAPE.split(ansi_string)
    if not pieces:
        return

    x_offset = 0
    current_pair = default_pair

    first_text = pieces[0]
    if first_text:
        _safe_addstr(stdscr, y, x + x_offset, first_text, current_pair | extra_attr)
        x_offset += len(first_text)

    index = 1
    while index < len(pieces):
        code = pieces[index] if index < len(pieces) else ''
        text = pieces[index + 1] if index + 1 < len(pieces) else ''
        current_pair = _color_for_ansi_code(code, theme_color_ids, default_pair)

        if text:
            _safe_addstr(stdscr, y, x + x_offset, text, current_pair | extra_attr)
            x_offset += len(text)

        index += 2


def _adaptive_centered_window(num_points):
    if num_points <= 2:
        return 1
    window = max(5, min(301, num_points // 25))
    if window % 2 == 0:
        window += 1
    return min(window, num_points if num_points % 2 == 1 else max(1, num_points - 1))


def calculate_centered_moving_average(df, column):
    window = _adaptive_centered_window(len(df))
    return df[column].rolling(window=window, center=True, min_periods=1).mean()


def range_has_data(df, x_min, x_max):
    return not df[(df['Time'] >= x_min) & (df['Time'] <= x_max)].empty


def calculate_effective_stride(total_points, plot_width, user_stride, oversample_factor=AUTO_STRIDE_OVERSAMPLE_FACTOR):
    base_stride = max(1, user_stride)
    if total_points <= 0 or plot_width <= 0:
        return base_stride

    target_points = max(1, plot_width * oversample_factor)
    auto_stride = max(1, math.ceil(total_points / target_points))
    return max(base_stride, auto_stride)


def _dedup_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def downsample_minmax_by_chunks(df, column, target_points):
    total_points = len(df)
    if total_points <= target_points or target_points <= 2:
        return df

    # Keep an envelope by splitting into chunks and preserving extrema order.
    bins = max(1, target_points // 2)
    chunk_size = max(1, math.ceil(total_points / bins))
    sampled_parts = []

    for start in range(0, total_points, chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        if chunk.empty:
            continue

        first_idx = chunk.index[0]
        last_idx = chunk.index[-1]

        y = chunk[column]
        min_idx = y.idxmin()
        max_idx = y.idxmax()

        min_pos = chunk.index.get_loc(min_idx)
        max_pos = chunk.index.get_loc(max_idx)
        extrema = [min_idx, max_idx] if min_pos <= max_pos else [max_idx, min_idx]

        selected_indices = _dedup_preserve_order([first_idx] + extrema + [last_idx])
        sampled_parts.append(chunk.loc[selected_indices, ['Time', column]])

    if not sampled_parts:
        return df

    sampled_df = pd.concat(sampled_parts)

    sampled_df = sampled_df[~sampled_df.index.duplicated(keep='first')]
    return sampled_df


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


def _render_loading_box(stdscr, menu_width, max_x, max_y, elapsed_seconds, progress_pct=None, progress_line=''):
    plot_left = menu_width + 4
    box_width = max(10, max_x - plot_left - 2)
    bar_width = max(10, min(60, box_width - 2))

    if progress_pct is None:
        spinner = ['|', '/', '-', '\\']
        spin_char = spinner[int(elapsed_seconds * 8) % len(spinner)]
        phase = elapsed_seconds % 2.0
        progress = phase if phase <= 1.0 else (2.0 - phase)
        filled = int(progress * bar_width)
        bar = '[' + ('#' * filled) + ('-' * (bar_width - filled)) + f'] {spin_char}'
        mode_line = 'Progress: records processed (total unknown)'
    else:
        filled = int((progress_pct / 100.0) * bar_width)
        bar = '[' + ('#' * filled) + ('-' * (bar_width - filled)) + f'] {progress_pct:5.1f}%'
        mode_line = 'Progress: percentage reported by loader'

    message = 'Loading EDR file...'
    hint = "Press 'q' to quit."
    elapsed_text = f'Elapsed: {elapsed_seconds:0.1f}s'

    center_y = max(6, max_y // 2)
    for y in range(max(4, center_y - 1), min(max_y - 1, center_y + 5)):
        _safe_addstr(stdscr, y, plot_left, ' ' * max(0, box_width - 1))
    _safe_addstr(stdscr, center_y - 1, plot_left, message[: max(0, box_width - 1)])
    _safe_addstr(stdscr, center_y, plot_left, bar[: max(0, box_width - 1)])
    _safe_addstr(stdscr, center_y + 1, plot_left, elapsed_text[: max(0, box_width - 1)])
    _safe_addstr(stdscr, center_y + 2, plot_left, mode_line[: max(0, box_width - 1)])
    if progress_line:
        _safe_addstr(stdscr, center_y + 3, plot_left, progress_line[: max(0, box_width - 1)])
        _safe_addstr(stdscr, center_y + 4, plot_left, hint[: max(0, box_width - 1)])
    else:
        _safe_addstr(stdscr, center_y + 3, plot_left, hint[: max(0, box_width - 1)])


def plot_ascii(df, column, units, width, height, x_min=None, x_max=None, stride=1, auto_stride_enabled=True, time_unit_mode='auto'):
    if width <= 0 or height <= 0:
        return ['Terminal too small to render plot.'], 1, 0, 0, 'raw'

    plotext.clf()
    plotext.plotsize(width, height)

    if x_min is not None and x_max is not None:
        visible_df = df[(df['Time'] >= x_min) & (df['Time'] <= x_max)]
    else:
        visible_df = df

    visible_df = visible_df[['Time', column]].dropna()
    total_visible_points = len(visible_df)
    sampling_mode = 'raw'

    if auto_stride_enabled:
        target_points = max(1, width * AUTO_STRIDE_OVERSAMPLE_FACTOR)
        sampled_df = downsample_minmax_by_chunks(visible_df, column, target_points)
        if stride > 1 and not sampled_df.empty:
            sampled_df = sampled_df.iloc[::stride]
        filtered_df = sampled_df
        effective_stride = calculate_effective_stride(total_visible_points, width, stride)
        sampling_mode = 'minmax'
    else:
        effective_stride = max(1, stride)
        filtered_df = visible_df.iloc[::effective_stride] if total_visible_points > 0 else visible_df

    trend_df = pd.DataFrame(columns=['Time', 'Trend'])
    if not visible_df.empty:
        trend_full = calculate_centered_moving_average(visible_df, column)
        trend_df = pd.DataFrame({'Time': visible_df['Time'], 'Trend': trend_full}).dropna()
        if auto_stride_enabled and not trend_df.empty:
            trend_df = downsample_minmax_by_chunks(trend_df, 'Trend', max(1, width * AUTO_STRIDE_OVERSAMPLE_FACTOR))

    if not filtered_df.empty:
        time_scale, time_unit = _time_axis_config(filtered_df['Time'], time_unit_mode)
        x_values = filtered_df['Time'] / time_scale
    else:
        time_scale, time_unit = _time_axis_config(visible_df['Time'] if not visible_df.empty else df['Time'], time_unit_mode)
        x_values = (visible_df['Time'] / time_scale) if not visible_df.empty else (df['Time'] / time_scale)

    if not filtered_df.empty:
        plotext.plot(x_values, filtered_df[column], label=column)
    elif not visible_df.empty:
        plotext.plot(x_values, visible_df[column], label=column)

    if not trend_df.empty:
        trend_x_values = trend_df['Time'] / time_scale
        plotext.plot(trend_x_values, trend_df['Trend'], label=f'Centered MA of {column}')

    y_label = _column_with_unit(column, units)
    plotext.title(f'{y_label} over Time')
    plotext.xlabel(f'Time ({time_unit})')
    plotext.ylabel(y_label)

    stats = _series_stats(visible_df[column] if not visible_df.empty else pd.Series(dtype=float))
    return plotext.build().split('\n'), effective_stride, len(filtered_df), total_visible_points, sampling_mode, stats


def _prepare_trend_df(df, column, x_min, x_max, stride, auto_stride_enabled, target_width):
    if x_min is not None and x_max is not None:
        visible_df = df[(df['Time'] >= x_min) & (df['Time'] <= x_max)]
    else:
        visible_df = df

    if visible_df.empty or column not in visible_df:
        return pd.DataFrame(columns=['Time', 'Trend']), 0, 0

    working_df = visible_df[['Time', column]].dropna()
    total_visible_points = len(working_df)
    if total_visible_points == 0:
        return pd.DataFrame(columns=['Time', 'Trend']), 0, 0

    if working_df.empty:
        return pd.DataFrame(columns=['Time', 'Trend']), 0, total_visible_points

    trend = calculate_centered_moving_average(working_df, column)
    trend_df = pd.DataFrame({'Time': working_df['Time'], 'Trend': trend}).dropna()
    if auto_stride_enabled and not trend_df.empty:
        trend_df = downsample_minmax_by_chunks(trend_df, 'Trend', max(1, target_width * 2))
    return trend_df, len(working_df), total_visible_points


def _time_axis_config(time_series, unit_mode='auto'):
    if time_series is None or len(time_series) == 0:
        return 1.0, 'ps'
    if unit_mode in TIME_UNIT_FACTORS_PS:
        return TIME_UNIT_FACTORS_PS[unit_mode], unit_mode
    max_time = float(time_series.max())
    if max_time < 1000.0:
        return 1.0, 'ps'
    return 1000.0, 'ns'


def _column_with_unit(column, units):
    if not units:
        return column
    unit = units.get(column)
    if unit:
        return f'{column} ({unit})'
    return column


def _series_stats(series):
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return {
        'n': int(clean.shape[0]),
        'mean': float(clean.mean()),
        'std': float(clean.std(ddof=0)),
    }


def _format_stats(stats):
    if not stats:
        return 'no data'
    return f"n={stats['n']} mu={stats['mean']:.4g} sigma={stats['std']:.4g}"


def _build_overview_panel_lines(trend_df, display_label, base_column, width, height, units, time_unit_mode, stats):
    if width <= 0 or height <= 0:
        return ['']
    if trend_df.empty:
        return ['No data']

    plotext.clf()
    plotext.plotsize(width, height)
    time_scale, time_unit = _time_axis_config(trend_df['Time'], time_unit_mode)
    x_values = trend_df['Time'] / time_scale
    plotext.plot(x_values, trend_df['Trend'])
    panel_label = _column_with_unit(base_column, units)
    if display_label:
        panel_label = f'{display_label} {panel_label}'
    plotext.title(panel_label[: max(1, width - 2)])
    plotext.xlabel(f'Time ({time_unit})')
    plotext.ylabel('')
    if stats:
        note = f"mu={stats['mean']:.3g} sigma={stats['std']:.3g}"
        plotext.hline(stats['mean'], color='white')
        plotext.xlabel((f'Time ({time_unit}) | {note}')[: max(1, width - 2)])
    return plotext.build().split('\n')


def draw_overview(stdscr, df, columns, units, menu_width, max_x, max_y, content_top, content_bottom, x_min, x_max, stride, auto_stride_enabled, page, theme, time_unit_mode):
    plot_left = menu_width + 4
    plot_width = max_x - menu_width - 5
    plot_height = max(0, content_bottom - content_top)

    if plot_width <= 30 or plot_height <= 12:
        return 'Terminal is too small for overview.', 0, 0

    n_cols = 2
    n_rows = 3
    per_page = 6
    total_pages = max(1, math.ceil(len(columns) / per_page))
    page = max(0, min(page, total_pages - 1))
    start_idx = page * per_page
    shown_columns = columns[start_idx:start_idx + per_page]

    cell_w = max(14, plot_width // n_cols)
    cell_h = max(8, plot_height // n_rows)

    _clear_region(stdscr, content_top, plot_left, plot_height, plot_width)

    plotted_count = 0
    no_data_count = 0
    for local_idx, column in enumerate(shown_columns):
        absolute_idx = start_idx + local_idx
        grid_row = local_idx // n_cols
        grid_col = local_idx % n_cols

        x0 = plot_left + (grid_col * cell_w)
        y0 = content_top + (grid_row * cell_h)
        panel_w = max(10, cell_w - 1)
        panel_h = max(6, cell_h - 1)

        trend_df, _, _ = _prepare_trend_df(
            df,
            column,
            x_min,
            x_max,
            stride,
            auto_stride_enabled,
            panel_w,
        )
        if x_min is not None and x_max is not None:
            stats_source = df[(df['Time'] >= x_min) & (df['Time'] <= x_max)][column].dropna()
        else:
            stats_source = df[column].dropna()
        stats = _series_stats(stats_source)
        panel_lines = _build_overview_panel_lines(
            trend_df,
            f'{absolute_idx + 1}.',
            column,
            panel_w,
            panel_h,
            units,
            time_unit_mode,
            stats,
        )

        if trend_df.empty:
            no_data_count += 1
        else:
            plotted_count += 1

        row_ptr = y0
        for line in panel_lines:
            if row_ptr >= content_bottom:
                break
            # Do not slice raw ANSI strings; slicing can break escape sequences.
            line_attr = curses.A_BOLD if row_ptr == y0 else 0
            parse_and_print_ansi(stdscr, row_ptr, x0, line, theme, extra_attr=line_attr)
            row_ptr += 1

    info = (
        f'Overview page {page + 1}/{total_pages} '
        f'({len(shown_columns)} shown, trend={plotted_count}, no-data={no_data_count}) '
        f'- LEFT/RIGHT pages'
    )
    return info, plotted_count, total_pages


def edterm_main(stdscr, args, preloaded_df=None):
    curses.curs_set(0)
    setup_colors(args.theme)
    stdscr.nodelay(1)
    stdscr.clear()

    max_y, max_x = stdscr.getmaxyx()
    menu_width = 20

    data_ready = preloaded_df is not None
    df = preloaded_df
    units = getattr(args, '_preloaded_units', {}) if data_ready else {}
    columns = list(df.columns[1:]) if data_ready else []

    if not data_ready:
        progress_buffer = ProgressBuffer()
        loading_started_at = time.time()
        last_draw_time = [0.0]

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
            _render_loading_box(stdscr, menu_width, max_x, max_y, elapsed, progress_pct, progress_line)
            _safe_addstr(stdscr, 3, 0, ' ' * max(0, max_x - 1))
            _safe_addstr(stdscr, 3, 0, 'Auto-stride mode=minmax envelope downsampling.'[: max(0, max_x - 1)])
            stdscr.noutrefresh()
            curses.doupdate()

        def progress_callback(bytes_read, total_bytes, records_read):
            progress_buffer.update_bytes(bytes_read, total_bytes, records_read)
            draw_loading_snapshot(force=False)

        draw_loading_snapshot(force=True)
        loaded_df = load_data(
            args.file,
            verbose=False,
            stderr_sink=None,
            progress_callback=progress_callback,
        )
        loaded_units = load_units(args.file)
        draw_loading_snapshot(force=True)
        validation_error = validate_loaded_dataframe(loaded_df)
        if validation_error:
            load_error_message = f'{validation_error} Press q to quit.'
            status_message = load_error_message
            plot_info_message = ''
        else:
            df = loaded_df
            units = loaded_units if isinstance(loaded_units, dict) else {}
            columns = list(df.columns[1:])
            data_ready = True
            resize_happened = True
            range_changed = True
            status_message = ''

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

    while True:
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
                "q quit | arrows select | r range | s stride | u unit | i header | L/R overview pages"[: max(0, max_x - 1)],
            )
            detail_line = (
                f"unit={time_unit_mode} stride={current_stride} auto={auto_mode} "
                f"stats={_format_stats(current_stats)}"
            )
            _safe_addstr(stdscr, 3, 0, detail_line[: max(0, max_x - 1)])
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
                f"{short_name} | unit={time_unit_mode} stride={current_stride} auto={auto_mode} "
                f"| {_format_stats(current_stats)} | press i to expand"
            )
            _safe_addstr(stdscr, 0, 0, compact_line[: max(0, max_x - 1)])
            for x in range(max_x):
                try:
                    stdscr.addch(1, x, curses.ACS_HLINE)
                except curses.error:
                    break
            header_plot_info_row = 0
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
            last_index = current_index
            plot_width = max_x - menu_width - 5
            content_bottom = footer_sep_row
            plot_height = content_bottom - content_top
            _clear_region(stdscr, content_top, menu_width + 1, max(0, content_bottom - content_top), max(0, max_x - menu_width - 1))

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
                        )
                        current_stats = None
                    else:
                        selected_column = columns[current_index - 1]
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
                        )
                        plot_row = content_top
                        for line in plot_lines:
                            if plot_row < content_bottom:
                                parse_and_print_ansi(stdscr, plot_row, menu_width + 4, line, args.theme)
                                plot_row += 1
                        current_stats = stats
                        plot_info_message = (
                            f'Rendering {plotted_points}/{total_visible_points} points '
                            f'(mode={sampling_mode}, stride={effective_stride}, set={current_stride}, auto={auto_mode}, unit={time_unit_mode})'
                        )
                    status_message = ''
                except Exception:
                    logger.exception('Failed to render plot for selection %s', current_index)
                    plot_info_message = ''
                    status_message = 'Plot rendering failed. See .edterm_debug.log for details.'

            resize_happened = False
            range_changed = False

        if status_message:
            stdscr.move(footer_row, 0)
            stdscr.clrtoeol()
            _safe_addstr(stdscr, footer_row, 0, status_message[: max(0, max_x - 1)])

        _safe_addstr(stdscr, header_plot_info_row, 0, ' ' * max(0, max_x - 1))
        if plot_info_message:
            _safe_addstr(stdscr, header_plot_info_row, 0, plot_info_message[: max(0, max_x - 1)])

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
            elif char == 'i':
                header_expanded = not header_expanded
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

    args = parser.parse_args()

    setup_logger(args.logging_level)
    setup_environment()

    if not os.path.isfile(args.file):
        print(f"Error: file '{args.file}' does not exist.", file=sys.stderr)
        return 1

    try:
        if args.load_progress:
            curses.wrapper(edterm_main, args, None)
        else:
            loaded_df = load_data(args.file, verbose=False, stderr_sink=None)
            loaded_units = load_units(args.file)
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
