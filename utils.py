

BACKGROUND_COLORS = {"black": "0;{};40",
                     "red": "0;{};41",
                     "green": "0;{};42",
                     "orange": "0;{};43",
                     "blue": "0;{};44",
                     "purple": "0;{};45",
                     "dark green": "0;{};46",
                     "white": "0;{};47"}

COLORS = {"black": "30",
          "red": "31",
          "green": "32",
          "orange": "33",
          "blue": "34",
          "purple": "35",
          "olive green": "36",
          "white": "37"}


def cprint(text, color, background_color, end=""):
    """
    Prints text with color.
    """
    color_string = BACKGROUND_COLORS[background_color].format(COLORS[color])
    text = f"\x1b[{color_string}m{text}\x1b[0m{end}"
    print(text, end="")