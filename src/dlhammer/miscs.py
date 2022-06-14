def to_string(params, indent=2):
    """Format params to a string.
    Args:
        params (EasyDict): the params. 
    Returns: The string to display.
    """
    msg = '{\n'
    for i, (k, v) in enumerate(params.items()):
        if isinstance(v, dict):
            v = to_string(v, indent + 4)
        spaces = ' ' * indent
        msg += spaces + '{}: {}'.format(k, v)
        if i == len(params) - 1:
            msg += ' }'
        else:
            msg += '\n'
    return msg
