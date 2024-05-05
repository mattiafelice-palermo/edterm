import pytest
from unittest.mock import MagicMock
import curses

@pytest.fixture
def mock_curses(mocker):
    # Mock the curses module functions required for initialization
    mocker.patch('curses.initscr', return_value=MagicMock())
    mocker.patch('curses.endwin')
    mocker.patch('curses.color_pair', side_effect=lambda x: f"ColorPair({x})")
    mocker.patch('curses.A_NORMAL', 'NORMAL')
    mocker.patch('curses.A_REVERSE', 'REVERSE')

    # You may need to mock more depending on what your application uses
    # For example, if you use window objects:
    win = MagicMock()
    mocker.patch('curses.newwin', return_value=win)

# Mocking curses functions globally for all tests
@pytest.fixture
def mock_curses(mocker):
    mocker.patch('curses.initscr', return_value=MagicMock())
    mocker.patch('curses.endwin')
    mocker.patch('curses.color_pair', side_effect=lambda x: f"ColorPair({x})")
    return curses

@pytest.fixture
def mock_stdscr(mocker):
    stdscr = MagicMock()
    stdscr.addstr = MagicMock()
    stdscr.attron = MagicMock()

    return stdscr

# Example test using the mocks
def test_parse_and_print_ansi_without_color_codes(mock_stdscr, mock_curses):
    y, x = 0, 0
    ansi_string = "Normal text"
    from edterm.edterm import parse_and_print_ansi  # Import here if it uses curses at module level
    parse_and_print_ansi(mock_stdscr, y, x, ansi_string)
    mock_stdscr.addstr.assert_called_with(0, 0, 'Normal text')

def test_parse_and_print_ansi_with_color_codes(mock_stdscr, mocker, mock_curses):
    y, x = 0, 0
    ansi_string = "\x1b[48;31mGreen Text\x1b[0m"
    #ansi_string = "Normal text"
    from edterm.edterm import parse_and_print_ansi  # Import here if it uses curses at module level
    parse_and_print_ansi(mock_stdscr, y, x, ansi_string)
    mock_stdscr.addstr.assert_called_with(0, 0, 'Green Text')
    #mock_stdscr.addstr.assert_called_with(0, 0, 'Normal text')

    # parse_and_print_ansi(mock_stdscr, y, x, "ColorPair(43)")
    # calls = [
    #     mocker.call(0, 0, 'Green Text'),
    # ]
    # mock_stdscr.addstr.assert_has_calls(calls, any_order=True)
    # mock_stdscr.attron.assert_called_with("ColorPair(43)")  # 42 + 1 for foreground color
    # mock_stdscr.attrset.assert_called_with('NORMAL')

# def test_parse_and_print_ansi_with_multiple_attributes(mock_stdscr, mock):
#     y, x = 0, 0
#     ansi_string = "\x1b[38;5;42;48;5;100mComplex Text\x1b[0m"
#     parse_and_print_ansi(mock_stdscr, y, x, ansi_string)
#     calls = [
#         mocker.call(0, 0, 'Complex Text'),
#     ]
#     mock_stdscr.addstr.assert_has_calls(calls, any_order=True)
#     mock_stdscr.attron.assert_any_call("ColorPair(43)")  # Foreground color
#     mock_stdscr.attron.assert_any_call("ColorPair(101)")  # Background color
#     mock_stdscr.attrset.assert_called_with('NORMAL')
