import math

import numpy as np
import plotext

from .analysis import (
    AUTO_STRIDE_OVERSAMPLE_FACTOR,
    _column_with_unit,
    _compute_trend_for_column,
    _histogram_bin_count,
    _normality_stats,
    _series_stats,
    _time_axis_config,
    calculate_effective_stride,
    downsample_minmax_by_chunks,
    visible_mask_for_range,
)


def plot_ascii(
    df,
    column,
    units,
    width,
    height,
    x_min=None,
    x_max=None,
    stride=1,
    auto_stride_enabled=True,
    time_unit_mode='auto',
    visible_mask=None,
    trend_series=None,
):
    if width <= 0 or height <= 0:
        return ['Terminal too small to render plot.'], 1, 0, 0, 'raw', None

    plotext.clf()
    plotext.plotsize(width, height)

    time_values = np.asarray(df['time'], dtype=np.float64)
    y_values = np.asarray(df['values'].get(column, np.array([], dtype=np.float64)), dtype=np.float64)
    if visible_mask is None:
        visible_mask = visible_mask_for_range(time_values, x_min, x_max)
    visible_mask = np.asarray(visible_mask, dtype=bool)

    x_visible = time_values[visible_mask]
    y_visible = y_values[visible_mask]
    finite_visible = np.isfinite(x_visible) & np.isfinite(y_visible)
    x_visible = x_visible[finite_visible]
    y_visible = y_visible[finite_visible]
    total_visible_points = x_visible.shape[0]
    sampling_mode = 'raw'

    if auto_stride_enabled:
        target_points = max(1, width * AUTO_STRIDE_OVERSAMPLE_FACTOR)
        sampled_x, sampled_y = downsample_minmax_by_chunks(x_visible, y_visible, target_points)
        if stride > 1 and sampled_x.size > 0:
            sampled_x = sampled_x[::stride]
            sampled_y = sampled_y[::stride]
        filtered_x, filtered_y = sampled_x, sampled_y
        effective_stride = calculate_effective_stride(total_visible_points, width, stride)
        sampling_mode = 'minmax'
    else:
        effective_stride = max(1, stride)
        filtered_x = x_visible[::effective_stride] if total_visible_points > 0 else x_visible
        filtered_y = y_visible[::effective_stride] if total_visible_points > 0 else y_visible

    trend_x = np.array([], dtype=np.float64)
    trend_y = np.array([], dtype=np.float64)
    if x_visible.size > 0:
        if trend_series is None:
            trend_series = _compute_trend_for_column(df, column)
        trend_arr = np.asarray(trend_series, dtype=np.float64)
        if trend_arr.shape[0] != time_values.shape[0]:
            trend_arr = _compute_trend_for_column(df, column)
        trend_visible = trend_arr[visible_mask]
        trend_visible = trend_visible[finite_visible]
        trend_finite = np.isfinite(trend_visible)
        trend_x = x_visible[trend_finite]
        trend_y = trend_visible[trend_finite]
        if auto_stride_enabled and trend_x.size > 0:
            trend_x, trend_y = downsample_minmax_by_chunks(trend_x, trend_y, max(1, width * AUTO_STRIDE_OVERSAMPLE_FACTOR))

    if filtered_x.size > 0:
        time_scale, time_unit = _time_axis_config(filtered_x, time_unit_mode)
        x_plot = filtered_x / time_scale
    else:
        time_scale, time_unit = _time_axis_config(x_visible if x_visible.size > 0 else time_values, time_unit_mode)
        x_plot = (x_visible / time_scale) if x_visible.size > 0 else (time_values / time_scale)

    if filtered_x.size > 0:
        plotext.plot(x_plot, filtered_y, label=column)
    elif x_visible.size > 0:
        plotext.plot(x_plot, y_visible, label=column)

    if trend_x.size > 0:
        trend_x_values = trend_x / time_scale
        plotext.plot(trend_x_values, trend_y, label=f'Centered MA of {column}')

    y_label = _column_with_unit(column, units)
    plotext.title(f'{y_label} over Time')
    plotext.xlabel(f'Time ({time_unit})')
    plotext.ylabel(y_label)

    stats = _series_stats(y_visible)
    return plotext.build().split('\n'), effective_stride, filtered_x.size, total_visible_points, sampling_mode, stats


def plot_histogram(df, column, units, width, height, x_min=None, x_max=None, visible_mask=None):
    if width <= 0 or height <= 0:
        return ['Terminal too small to render plot.'], 0, 0, 'hist', None

    plotext.clf()
    plotext.plotsize(width, height)

    time_values = np.asarray(df['time'], dtype=np.float64)
    y_values = np.asarray(df['values'].get(column, np.array([], dtype=np.float64)), dtype=np.float64)
    if visible_mask is None:
        visible_mask = visible_mask_for_range(time_values, x_min, x_max)
    visible_mask = np.asarray(visible_mask, dtype=bool)

    y_visible = y_values[visible_mask]
    clean = y_visible[np.isfinite(y_visible)]
    total_points = int(clean.size)
    if total_points == 0:
        return ['No data in selected range for histogram.'], 0, 0, 'hist', None

    bins = _histogram_bin_count(clean, width=width)
    counts, edges = np.histogram(clean, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plotext.bar(centers, counts, label=f'Histogram of {column}')

    normal = _normality_stats(clean)
    if normal and normal['std'] > 0 and bins > 1:
        xs = np.linspace(float(np.min(clean)), float(np.max(clean)), num=min(180, max(40, width)))
        sigma = normal['std']
        mu = normal['mean']
        pdf = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        bin_width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
        scaled_pdf = pdf * total_points * bin_width
        plotext.plot(xs, scaled_pdf, label='Gaussian fit')

    y_label = _column_with_unit(column, units)
    plotext.title(f'{y_label} Distribution')
    plotext.xlabel(y_label)
    plotext.ylabel('Count')
    try:
        xt = np.linspace(float(np.min(clean)), float(np.max(clean)), num=5)
        xl = [f'{v:.4g}' for v in xt]
        plotext.xticks(list(xt), xl)
    except Exception:
        pass

    return plotext.build().split('\n'), total_points, bins, 'hist', normal


def build_overview_panel_lines(trend_x, trend_y, display_label, base_column, width, height, units, time_unit_mode, stats):
    if width <= 0 or height <= 0:
        return ['']
    if trend_x.size == 0 or trend_y.size == 0:
        return ['No data']

    plotext.clf()
    plotext.plotsize(width, height)
    time_scale, time_unit = _time_axis_config(trend_x, time_unit_mode)
    x_values = trend_x / time_scale
    plotext.plot(x_values, trend_y)
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
