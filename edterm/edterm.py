import os
import argparse
import pandas as pd
from .data_reader import load_data
import time
import plotext
import logging
import locale

# Create a custom logger
logger = logging.getLogger(__name__)
# Create handlers
logger_handler = logging.FileHandler('.edterm_debug.log')

# Create formatters and add it to handlers
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(logger_handler)


def setup_environment():
    # Check and set environment variables for locale settings
    if os.environ.get('LANG', '') != 'en_US.UTF-8':
        print("hmpg")
        os.environ['LANG'] = 'en_US.UTF-8'
    if os.environ.get('LC_ALL', '') != 'en_US.UTF-8':
        os.environ['LC_ALL'] = 'en_US.UTF-8'
    # Set locale settings in the locale module
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


setup_environment()

import curses

def setup_logger(logger_level=logging.DEBUG):
    # Map the string logging level to logging module levels
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    # Get the logging level from the dictionary
    log_level = level_dict.get(logger_level.lower(), logging.INFO)

    # Set the log level
    logger.setLevel(log_level)
    logger_handler.setLevel(log_level)

def setup_colors(theme):
    curses.start_color()
    curses.use_default_colors()

    if theme == 'transparent':
        background_color = -1
    if theme == 'dark':
        background_color = 16
    if theme == 'light':
        background_color = 15

    for i in range(0, curses.COLORS):
        curses.init_pair(i, i, background_color)

def parse_and_print_ansi(stdscr, y, x, ansi_string, theme):
    # TODO: could be made much better and support plotext themes, but for now it will do
    import re
    ansi_escape = re.compile(r'\x1b\[([0-9;]*)m')
    pieces = ansi_escape.split(ansi_string)
    x_offset = 0

    clean_list = remove_consecutive_elements(pieces, ['', '48;5;15', ''])
    elements = [element for element in clean_list if element != '']

    x_offset = 0
    if theme == 'transparent':
        color_1 = 0
        color_2 = 11 # Main observable
        color_3 = 15
        color_4 = 15
    if theme == 'dark':
        color_1 = 15
        color_2 = 15
        color_3 = 156
        color_4 = 156
    if theme == 'light':    
        color_1 = 232
        color_2 = 4
        color_3 = 156
        color_4 = 8

    while elements:
        # Extract color code
        code = elements.pop(0)
        element = elements.pop(0)
        end = elements.pop(0)

        #logger.info(f"Text: {element} Code: {code}")

        if code.startswith("38;5;0"):
            color_pair = curses.color_pair(color_1)
        if code == "38;5;12":
            color_pair = curses.color_pair(color_2)
        if code.startswith("48;5;"):
            color_pair = curses.color_pair(color_3)
        if code == "38;5;10":
            color_pair = curses.color_pair(color_4)

        # Add text with the specific color pair directly
        stdscr.addstr(y, x + x_offset, element, color_pair)
        x_offset += len(element)

def calculate_moving_average(df, column, window=50):
    return df[column].rolling(window=window, min_periods=1).mean()

def calculate_expanding_average(df, column):
    return df[column].expanding().mean()

def plot_ascii(df, column, width, height, x_min = None, x_max = None, stride=1):
    # Clear previous plots
    plotext.clf()

    # Set plot size (adjust based on terminal size)
    plotext.plotsize(width, height)  # width, height in characters

    # Filter data based on x_min and x_max
    if x_min is not None and x_max is not None:
        filtered_df = df[(df['Time'] >= x_min) & (df['Time'] <= x_max)]
    else:
        filtered_df = df

    # Apply stride to reduce the number of data points plotted
    if not filtered_df.empty:
        filtered_df = filtered_df.iloc[::stride]  # This skips data points

    moving_avg = calculate_expanding_average(filtered_df, column)

    # Plot data
    if not filtered_df.empty:
        plotext.plot(filtered_df['Time'], filtered_df[column], label=f"{column}")
    else:
        plotext.plot(df['Time'], df[column])  # fallback if the filter results in no data

    # Plot the moving average data
    plotext.plot(filtered_df['Time'], moving_avg, label=f"Moving Average of {column}")

    plotext.title(f"{column} over Time")
    plotext.xlabel("Time (ps)")
    plotext.ylabel(column)

    # Build the plot as a string
    plot_str = plotext.build()
    return plot_str.split('\n')


def edterm_main(stdscr, args):
    setup_environment()  # Ensure environment variables are correctly set
    curses.curs_set(0)
    setup_colors(args.theme)
    stdscr.nodelay(1)
    stdscr.clear()

    df = load_data(args.file)
    columns = list(df.columns[1:]) if not df.empty else []
    current_index = 0
    last_index = -1  # Initialize to force redraw on first loop
    max_y, max_x = stdscr.getmaxyx()  # Initial size
    menu_width = 20  # Assume menu width to start
    first_draw = True
    last_number_time = time.time()
    input_mode = False
    resize_happened = True
    x_min, x_max = None, None
    number_buffer = ""

    while True:
        if input_mode:
            stdscr.nodelay(0)  # Switch to blocking mode for input
            stdscr.move(max_y-1, 0)
            stdscr.clrtoeol()
            stdscr.addstr(max_y-1, 0, "Provide the desired time window (x_min x_max): ")
            curses.echo()
            input_str = stdscr.getstr(max_y-1, 50).decode('utf-8')
            try:
                x_min, x_max = map(float, input_str.split())
                stdscr.move(max_y-1, 0)
                stdscr.clrtoeol()
                stdscr.addstr(max_y-1, 0, f"Time window set to: {x_min} - {x_max}")
                stdscr.refresh()
            except ValueError:
                stdscr.addstr(max_y-1, 0, "Invalid input, please enter two numbers.")
            curses.noecho()
            input_mode = False
            stdscr.nodelay(1)  # Turn non-blocking mode back on


        new_max_y, new_max_x = stdscr.getmaxyx()
        if new_max_y != max_y or new_max_x != max_x:
            max_y, max_x = new_max_y, new_max_x
            first_draw = True
            resize_happend = True
            stdscr.clear()  # Clear the screen because the terminal size has changed
            stdscr.refresh()


        # Draw static elements only if they need to be redrawn or once
        if first_draw:
            stdscr.addstr(0, 0, "Welcome to the GROMACS Data Plotter Tool")
            stdscr.addstr(1, 0, f"File: {args.file}")
            stdscr.addstr(2, 0, "Press 'q' to quit. Use UP/DOWN arrows or type numbers to select.")
            first_draw = False

        menu_row = 4

        # Draw labels and vertical line
        for i, col in enumerate(columns):
            if menu_row + i < max_y:
                mode = curses.A_REVERSE if i == current_index else curses.A_NORMAL
                stdscr.addstr(menu_row + i, 0, f"{i+1}. {col}".ljust(menu_width), mode)

        # Draw the vertical dividing line
        for y in range(4, max_y-1):
            stdscr.addch(y, menu_width, curses.ACS_VLINE)

        if (current_index != last_index) or not resize_happened:
            last_index = current_index
            plot_width = max_x - menu_width - 5  # Calculate available width for plot
            plot_height = max_y - 5  # Calculate available height for plot
            plot_lines = plot_ascii(df, columns[current_index], plot_width, plot_height, x_min, x_max, args.stride)
            plot_row = 4  # Starting row for plot
            stdscr.clrtobot()  # Clear from here to bottom of the screen
            for line in plot_lines:
                if plot_row < max_y - 1:
                    parse_and_print_ansi(stdscr, plot_row, menu_width + 4, line, args.theme)
                    plot_row += 1
            resize_happened = False

        stdscr.noutrefresh()
        curses.doupdate()
        #curses.napms(10)  # Smoother refresh rate

        k = stdscr.getch()

        # Handle number input and timeout
        if k != -1 and k < 256:  # Ensure k is a valid ASCII value
            char = chr(k)
            if time.time() - last_number_time > 1.:
                number_buffer = ""
            if '0' <= char <= '9':
                number_buffer += char
                last_number_time = time.time()
                if number_buffer:
                    number = int(number_buffer) - 1
                    if 0 <= number < len(columns):
                        current_index = number

            if char == 'q':
                break  # Quit on 'q'
            if char == 'r':
                input_mode = True

        if k == curses.KEY_UP and current_index > 0:
            current_index -= 1  # Move selection up
        elif k == curses.KEY_DOWN and current_index < len(columns) - 1:
            current_index += 1  # Move selection down

def main():
    parser = argparse.ArgumentParser(description='GROMACS Data Plotter Tool')
    
    # Add the path to the GROMACS EDR file argument
    parser.add_argument('file', type=str, help='Path to the GROMACS EDR file')
    
    # Add an optional argument for stride with default value of 1
    parser.add_argument('--stride', '-s', type=int, default=1, help='Stride for data plotting to reduce plot density')
    
    # Add an optional argument for setting logging level
    parser.add_argument('--logging-level', '-ll', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set logger level')
    
    # Add an optional argument for setting color theme
    parser.add_argument('--theme', '-t', type=str, default='transparent', choices=['dark', 'light', 'transparent'], help='Set the color theme')

    args = parser.parse_args()

    setup_logger(args.logging_level)
    
    curses.wrapper(edterm_main, args)

if __name__ == '__main__':
    main()

def remove_consecutive_elements(source_list, elements_to_remove):
    n = len(source_list)
    m = len(elements_to_remove)
    i = 0
    result = []
    
    while i < n:
        # Check if the current slice matches the elements_to_remove
        if i <= n - m and source_list[i:i+m] == elements_to_remove:
            i += m  # Skip the next m elements
        else:
            result.append(source_list[i])
            i += 1
    
    return result