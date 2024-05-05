import os
import argparse
import curses
import pandas as pd
from .data_reader import load_data
import time
import plotext
import logging

def setup_environment():
    # Check and set environment variables for locale settings
    if os.environ.get('LANG', '') != 'en_US.UTF-8':
        os.environ['LANG'] = 'en_US.UTF-8'
    if os.environ.get('LC_ALL', '') != 'en_US.UTF-8':
        os.environ['LC_ALL'] = 'en_US.UTF-8'

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the log level
    logger.setLevel(logging.DEBUG)

    # Create handlers
    f_handler = logging.FileHandler('debug.log')
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)

    return logger

def setup_colors():
    curses.start_color()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)

def parse_and_print_ansi(stdscr, y, x, ansi_string):
    import re
    ansi_escape = re.compile(r'\x1b\[([0-9;]*)m')
    pieces = ansi_escape.split(ansi_string)
    x_offset = 0

    while pieces:
        text = pieces.pop(0)
        stdscr.addstr(y, x + x_offset, text)
        x_offset += len(text)

        if pieces:
            codes = pieces.pop(0)
            # Apply the ANSI codes as curses attributes or colors
            for code in codes.split(';'):
                if code == "0":  # Reset
                    stdscr.attroff(curses.color_pair(1))
                    stdscr.attrset(curses.A_NORMAL)
                elif code.startswith("38;5;"):  # Foreground Color
                    color_number = int(code[5:])
                    stdscr.attron(curses.color_pair(color_number + 1))
                elif code.startswith("48;5;"):  # Background Color
                    color_number = int(code[5:])
                    stdscr.attron(curses.color_pair(color_number + 1))
                # Add more cases as needed

def calculate_moving_average(df, column, window=50):
    return df[column].rolling(window=window, min_periods=1).mean()

def plot_ascii(df, column, width, height, x_min = None, x_max = None):
    # Clear previous plots
    plotext.clf()

    # Set plot size (adjust based on terminal size)
    plotext.plotsize(width, height)  # width, height in characters

    # Filter data based on x_min and x_max
    if x_min is not None and x_max is not None:
        filtered_df = df[(df['Time'] >= x_min) & (df['Time'] <= x_max)]
    else:
        filtered_df = df

    # Plot data
    if not filtered_df.empty:
        plotext.plot(filtered_df['Time'], filtered_df[column])
    else:
        plotext.plot(df['Time'], df[column])  # fallback if the filter results in no data

    plotext.title(f"{column} over Time")
    plotext.xlabel("Time (ps)")
    plotext.ylabel(column)

    # Build the plot as a string
    plot_str = plotext.build()
    return plot_str.split('\n')


def edterm_main(stdscr, args):
    setup_environment()  # Ensure environment variables are correctly set
    logger = setup_logger()
    curses.curs_set(0)
    setup_colors()
    stdscr.nodelay(1)
    stdscr.clear()

    df = load_data(args.file)
    columns = list(df.columns) if not df.empty else []
    current_index = 0
    last_index = -1  # Initialize to force redraw on first loop
    max_y, max_x = stdscr.getmaxyx()  # Initial size
    menu_width = 20  # Assume menu width to start
    first_draw = True
    last_number_time = time.time()
    input_mode = False
    resize_happened = True
    x_min, x_max = None, None
    counter = 0
    number_buffer = ""

    while True:
        counter += 1
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
            logger.info(f"Resize happend! {counter}")
            max_y, max_x = new_max_y, new_max_x
            first_draw = True
            resize_happend = True
            stdscr.clear()  # Clear the screen because the terminal size has changed
            stdscr.refresh()


        if not resize_happened: logger.info(f"Resize status {resize_happened}, {counter}")

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
            plot_lines = plot_ascii(df, columns[current_index], plot_width, plot_height, x_min, x_max)
            plot_row = 4  # Starting row for plot
            stdscr.clrtobot()  # Clear from here to bottom of the screen
            for line in plot_lines:
                logger.info(len(line))
                if plot_row < max_y - 1:
                    parse_and_print_ansi(stdscr, plot_row, menu_width + 4, line)
                    plot_row += 1
            resize_happened = False

        stdscr.noutrefresh()
        curses.doupdate()
        curses.napms(10)  # Smoother refresh rate

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
        if time.time() - last_number_time > 0.5:
            if number_buffer:
                number = int(number_buffer) - 1
                if 0 <= number < len(columns):
                    current_index = number
                    number_buffer = ""

        if k == curses.KEY_UP and current_index > 0:
            current_index -= 1  # Move selection up
        elif k == curses.KEY_DOWN and current_index < len(columns) - 1:
            current_index += 1  # Move selection down

def main():
    parser = argparse.ArgumentParser(description='GROMACS Data Plotter Tool')
    parser.add_argument('file', type=str, help='Path to the GROMACS EDR file')
    args = parser.parse_args()
    
    curses.wrapper(edterm_main, args)

if __name__ == '__main__':
    main()
