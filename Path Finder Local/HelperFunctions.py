from os import getcwd, listdir
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

def version(base_version, model_name):
    """
    Increment the fractional part as an integer counter while keeping the integer base.
    Examples:
      highest = 1.9    -> next = 1.10
      highest = 1.09   -> next = 1.10
      highest = 1.155 -> next = 1.156
      highest = 1     -> next = 1.1
    Returns a Decimal constructed from the resulting string (preserves fractional digit width).
    """
    try:
        base_dec = Decimal(str(base_version))
    except (InvalidOperation, TypeError):
        base_dec = Decimal(0)

    base_int = int(base_dec.to_integral_value(rounding=ROUND_FLOOR))

    highest_ver = None
    highest_frac_str = None

    for f_name in listdir(getcwd()):
        if not (f_name.startswith(model_name + '_') and f_name.endswith('.keras')):
            continue
        ver_str = f_name[len(model_name) + 1:-6]  # strip prefix and suffix
        try:
            ver_dec = Decimal(ver_str)
        except InvalidOperation:
            continue

        # split into integer and fractional text parts using the original string to preserve formatting
        if '.' in ver_str:
            int_part, frac_part = ver_str.split('.', 1)
        else:
            int_part, frac_part = ver_str, ''

        # only consider files with same integer base
        try:
            if int(int_part) != base_int:
                continue
        except Exception:
            continue

        if highest_ver is None or ver_dec > highest_ver:
            highest_ver = ver_dec
            highest_frac_str = frac_part  # may be '' for no fractional part

    if highest_ver is None:
        # start at base.0 (explicit fractional .0)
        return Decimal(f"{base_int}.0"), Decimal(f"{base_int}.0")

    # Use the original fractional string (or '0' if none) as an integer counter
    orig_frac = highest_frac_str if highest_frac_str != '' else '0'
    orig_width = len(orig_frac)
    units = int(orig_frac)
    new_units = units + 1

    # new width is max(orig_width, digits of new_units) so we keep padding unless carry demands wider
    new_width = max(orig_width, len(str(new_units)))
    new_frac_str = str(new_units).zfill(new_width)

    old_version_str = f"{base_int}.{orig_frac}"
    next_version_str = f"{base_int}.{new_frac_str}"
    return Decimal(next_version_str), Decimal(old_version_str)