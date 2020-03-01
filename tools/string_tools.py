def seconds_to_string(t):
    def _pad(x):
        x = str(x)
        return x if len(x) == 2 else '0' + x

    t = int(t)
    return f'{_pad(t // 3600)}:{_pad((t % 3600) // 60)}:{_pad(t % 60)}'
