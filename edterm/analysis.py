import math

import numpy as np


AUTO_STRIDE_OVERSAMPLE_FACTOR = 3
TIME_UNIT_FACTORS_PS = {
    'fs': 0.001,
    'ps': 1.0,
    'ns': 1000.0,
    'us': 1_000_000.0,
}
TIME_UNIT_CYCLE = ['auto', 'fs', 'ps', 'ns', 'us']


def _adaptive_centered_window(num_points):
    if num_points <= 2:
        return 1
    window = max(5, min(301, num_points // 25))
    if window % 2 == 0:
        window += 1
    return min(window, num_points if num_points % 2 == 1 else max(1, num_points - 1))


def calculate_centered_moving_average(values):
    y = np.asarray(values, dtype=np.float64)
    if y.size == 0:
        return y
    window = _adaptive_centered_window(y.size)
    if window <= 1:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64)
    valid = np.isfinite(y)
    filled = np.where(valid, y, 0.0)
    summed = np.convolve(filled, kernel, mode='same')
    counts = np.convolve(valid.astype(np.float64), kernel, mode='same')
    trend = np.empty_like(y)
    trend.fill(np.nan)
    np.divide(summed, counts, out=trend, where=counts > 0)
    return trend


def _compute_trend_for_column(df, column):
    values = np.asarray(df['values'].get(column, np.array([], dtype=np.float64)), dtype=np.float64)
    if values.size == 0:
        return values
    return calculate_centered_moving_average(values)


def range_has_data(df, x_min, x_max):
    time_values = np.asarray(df['time'], dtype=np.float64)
    if time_values.size == 0:
        return False
    mask = (time_values >= x_min) & (time_values <= x_max)
    return bool(np.any(mask))


def visible_mask_for_range(time_values, x_min, x_max):
    if x_min is not None and x_max is not None:
        return (time_values >= x_min) & (time_values <= x_max)
    return np.ones(time_values.shape[0], dtype=bool)


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


def downsample_minmax_by_chunks(x_values, y_values, target_points):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    total_points = x.shape[0]
    if total_points <= target_points or target_points <= 2:
        return x, y

    bins = max(1, target_points // 2)
    chunk_size = max(1, math.ceil(total_points / bins))
    x_parts = []
    y_parts = []

    for start in range(0, total_points, chunk_size):
        end = min(total_points, start + chunk_size)
        cx = x[start:end]
        cy = y[start:end]
        if cx.size == 0:
            continue

        min_pos = int(np.argmin(cy))
        max_pos = int(np.argmax(cy))
        extrema = [min_pos, max_pos] if min_pos <= max_pos else [max_pos, min_pos]
        selected_local = _dedup_preserve_order([0] + extrema + [cx.size - 1])
        x_parts.append(cx[selected_local])
        y_parts.append(cy[selected_local])

    if not x_parts:
        return x, y
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _prepare_trend_df(df, column, auto_stride_enabled, target_width, visible_mask=None, trend_series=None):
    time_values = np.asarray(df['time'], dtype=np.float64)
    y_values = np.asarray(df['values'].get(column, np.array([], dtype=np.float64)), dtype=np.float64)
    if y_values.size == 0:
        return np.array([]), np.array([]), 0, 0

    if visible_mask is None:
        visible_mask = np.ones(time_values.shape[0], dtype=bool)

    x_vis = time_values[visible_mask]
    y_vis = y_values[visible_mask]
    finite = np.isfinite(x_vis) & np.isfinite(y_vis)
    x_vis = x_vis[finite]
    total_visible_points = x_vis.shape[0]
    if total_visible_points == 0:
        return np.array([]), np.array([]), 0, 0

    if trend_series is None:
        trend_series = _compute_trend_for_column(df, column)
    trend_arr = np.asarray(trend_series, dtype=np.float64)
    if trend_arr.shape[0] != time_values.shape[0]:
        trend_arr = _compute_trend_for_column(df, column)
    trend_vis = trend_arr[visible_mask]
    trend_vis = trend_vis[finite]
    trend_finite = np.isfinite(trend_vis)
    trend_x = x_vis[trend_finite]
    trend_y = trend_vis[trend_finite]
    if auto_stride_enabled and trend_x.size > 0:
        trend_x, trend_y = downsample_minmax_by_chunks(trend_x, trend_y, max(1, target_width * 2))
    return trend_x, trend_y, x_vis.shape[0], total_visible_points


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
    arr = np.asarray(series, dtype=np.float64)
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return None
    return {
        'n': int(clean.size),
        'mean': float(np.mean(clean)),
        'std': float(np.std(clean)),
    }


def _format_stats(stats):
    if not stats:
        return 'no data'
    base = f"n={stats['n']} mu={stats['mean']:.4g} sigma={stats['std']:.4g}"
    jb_p = stats.get('jb_p')
    if jb_p is not None:
        base = f'{base} jb_p={jb_p:.3g}'
    return base


def _normality_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[np.isfinite(arr)]
    if clean.size < 3:
        return None
    n = float(clean.size)
    mu = float(np.mean(clean))
    sigma = float(np.std(clean))
    if sigma <= 0:
        return {
            'n': int(clean.size),
            'mean': mu,
            'std': sigma,
            'skew': 0.0,
            'kurtosis': 0.0,
            'jb': 0.0,
            'jb_p': 1.0,
            'gaussian_like': True,
        }
    z = (clean - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurtosis = float(np.mean(z ** 4) - 3.0)
    jb = float((n / 6.0) * (skew ** 2 + 0.25 * (kurtosis ** 2)))
    jb_p = float(math.exp(-0.5 * jb))
    return {
        'n': int(clean.size),
        'mean': mu,
        'std': sigma,
        'skew': skew,
        'kurtosis': kurtosis,
        'jb': jb,
        'jb_p': jb_p,
        'gaussian_like': jb_p >= 0.05,
    }


def _histogram_bin_count(values, width=None):
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[np.isfinite(arr)]
    n = clean.size
    if n <= 2:
        return 1
    target_bins = 24 if width is None else max(10, min(32, int(width // 5)))
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 1
    q75, q25 = np.percentile(clean, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= 0:
        return max(10, min(target_bins, int(round(math.sqrt(n)))))
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if bin_width <= 0:
        return max(10, min(target_bins, int(round(math.sqrt(n)))))
    bins = int(math.ceil((vmax - vmin) / bin_width))
    return max(10, min(target_bins, bins))
