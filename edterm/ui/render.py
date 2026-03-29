import curses
import math
import re

import numpy as np

from ..analysis import _prepare_trend_df, _series_stats
from ..plotting import build_overview_panel_lines


ANSI_ESCAPE = re.compile(r'\x1b\[([0-9;]*)m')


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


def render_loading_box(stdscr, menu_width, max_x, max_y, elapsed_seconds, progress_pct=None, progress_line=''):
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


def draw_overview(
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
    stride,
    auto_stride_enabled,
    page,
    theme,
    time_unit_mode,
    visible_mask=None,
    trend_cache=None,
    trend_cache_lock=None,
    trend_getter=None,
    trend_computer=None,
):
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
    if visible_mask is None:
        time_values = np.asarray(df['time'], dtype=np.float64)
        visible_mask = np.ones(time_values.shape[0], dtype=bool)

    for local_idx, column in enumerate(shown_columns):
        absolute_idx = start_idx + local_idx
        grid_row = local_idx // n_cols
        grid_col = local_idx % n_cols

        x0 = plot_left + (grid_col * cell_w)
        y0 = content_top + (grid_row * cell_h)
        panel_w = max(10, cell_w - 1)
        panel_h = max(6, cell_h - 1)

        trend_series = None
        if trend_cache is not None:
            if trend_cache_lock is None:
                trend_series = trend_cache.get(column)
            else:
                with trend_cache_lock:
                    trend_series = trend_cache.get(column)
            if trend_series is None:
                if trend_cache_lock is None:
                    if trend_computer is not None:
                        trend_series = trend_computer(df, column)
                        trend_cache[column] = trend_series
                elif trend_getter is not None:
                    trend_series = trend_getter(df, column, trend_cache, trend_cache_lock)

        trend_x, trend_y, _, _ = _prepare_trend_df(
            df,
            column,
            auto_stride_enabled,
            panel_w,
            visible_mask=visible_mask,
            trend_series=trend_series,
        )
        col_values = np.asarray(df['values'].get(column, np.array([], dtype=np.float64)), dtype=np.float64)
        stats_source = col_values[visible_mask] if col_values.size > 0 else np.array([], dtype=np.float64)
        stats = _series_stats(stats_source)
        panel_lines = build_overview_panel_lines(
            trend_x,
            trend_y,
            f'{absolute_idx + 1}.',
            column,
            panel_w,
            panel_h,
            units,
            time_unit_mode,
            stats,
        )

        if trend_x.size == 0:
            no_data_count += 1
        else:
            plotted_count += 1

        row_ptr = y0
        for line in panel_lines:
            if row_ptr >= content_bottom:
                break
            line_attr = curses.A_BOLD if row_ptr == y0 else 0
            parse_and_print_ansi(stdscr, row_ptr, x0, line, theme, extra_attr=line_attr)
            row_ptr += 1

    info = (
        f'Overview page {page + 1}/{total_pages} '
        f'({len(shown_columns)} shown, trend={plotted_count}, no-data={no_data_count}) '
        f'- LEFT/RIGHT keys or click [<] [>]'
    )

    nav_y = content_top + max(0, plot_height // 2)
    if 0 <= nav_y < content_bottom:
        left_hint = '[<]'
        right_hint = '[>]'
        _safe_addstr(stdscr, nav_y, plot_left + 1, left_hint)
        _safe_addstr(stdscr, nav_y, max(plot_left + 1, plot_left + plot_width - len(right_hint) - 1), right_hint)
    return info, plotted_count, total_pages
