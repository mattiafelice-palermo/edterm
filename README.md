# edterm

`edterm` is a command-line interface (CLI) tool designed for plotting GROMACS EDR data directly in the terminal. This tool enables users to visualize data trends from simulation outputs without leaving the command line environment, providing a quick and interactive way to analyze simulation results.

## Features

- **Terminal-based Plotting**: Utilizes the terminal to plot data directly, making it accessible on systems without a graphical user interface or when forwarding X11 is impractical.
- **Themes**: Offers three themes—transparent, dark, and light—to accommodate different viewing preferences and terminal settings.
- **Interactive Navigation**: Users can navigate through different data contained in the EDR file using keyboard inputs.
- **Dynamic Time Window Adjustment**: Offers the ability to focus on specific time windows by entering desired time ranges.
- **Resizable Interface**: Automatically adjusts the plot and interface when the terminal size changes.
- **Stride for Data Reduction**: Introduces a stride option to reduce the amount of data processed, enhancing performance for large datasets.
- **Expanding Average Plot**: Plots the expanding average to provide additional analytical insight.
- **Keyboard Controls**: Simple keyboard controls for navigating and interacting with the data.
- **Configurable Logging Level**: Users can set the logging level through command-line options to tailor logging verbosity.

## Installation

To install `edterm`, run the following command:

```
pip install edterm
```

This command will download and install `edterm` along with its necessary dependencies.

## Usage

Once installed, you can run `edterm` using the following command:

```
edterm <path_to_your_edr_file.edr> [--stride <stride_value>] [--theme {transparent,dark,light}] [--logging-level {debug,info,warning,error,critical}]
```

Replace `<path_to_your_edr_file.edr>` with the actual path to your GROMACS EDR file.

### Keyboard Commands

- **UP/DOWN Arrows**: Navigate through the list of available data columns.
- **'r' Key**: Press 'r' to enter a range selection mode, where you can specify `xmin` and `xmax` for the x-axis. This allows you to zoom into specific time windows.
  - After pressing 'r', you will be prompted to enter the range in the format: `x_min x_max`. Input the values and press Enter to apply the range.
- **'q' Key**: Quit the application.

## Configuration

The plot and interaction settings are configured to work out of the box, but they can be adjusted by modifying the source code to better fit different terminal sizes or user preferences.

## Development

This tool is open for further development and contributions. Developers interested in contributing can clone the repository and submit pull requests.

## License

`edterm` is released under the MIT license. For more details, see the LICENSE file in the repository.

## Contact

For support or to report bugs, please visit the [GitHub repository issue tracker](https://github.com/mattiafelice-palermo/edterm/issues).

## Acknowledgements

This tool was developed to assist researchers and engineers in analyzing molecular dynamics simulations. Thanks to all contributors and users for their feedback and suggestions.