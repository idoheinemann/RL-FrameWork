import sys

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


def seconds_to_string(t):
    def _pad(x):
        x = str(x)
        return x if len(x) == 2 else '0' + x

    t = int(t)
    return f'{_pad(t // 3600)}:{_pad((t % 3600) // 60)}:{_pad(t % 60)}'


def delete_last_line(file=sys.stdout):
    file.write("\033[F")  # back to previous line
    file.write("\033[K")
