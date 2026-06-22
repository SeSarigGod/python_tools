import sys
import os
import numpy as np

def int_to_roman(num):
    roman_parts = []
    numeral_map = zip(
        [1000, 900, 500, 400, 100,  90,  50,  40,  10,   9,   5,   4,   1],
        ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    )
    for value, symbol in numeral_map:
        count, num = divmod(num, value)
        roman_parts.append(symbol * count)
    return ''.join(roman_parts)


def string_constructor(filled_num: str, num: str):
    return """    <li><a href="index_split_000.html#page0""" + filled_num + """">""" + num + """</a></li>"""

"""
<nav epub:type="page-list" hidden="">
  <ol>
    <li><a href="index_split_000.html#page0001"></a></li>
  </ol>
</nav>
"""

if __name__ == "__main__":
    num_pages = 442
    result = """<nav epub:type="page-list" hidden="">\n  <ol>\n"""

    for i in range(1, num_pages + 1):
        num_str = str(i).zfill(3)
        num_name = int_to_roman(i) if i <= 19 else str(i - 19)
        result += string_constructor(num_str, num_name) + "\n"

    result += "  </ol>\n</nav>"

    with open("../output.txt", "w") as f:
        f.write(result)

    t_d2ms = np.float64(24. * 60. * 60. * 1000.)
    t_unix = np.float64(1773878400000)
    print(t_unix)
    # t_julian_factor = np.uint64(2440587.5 * 24 * 60 * 60 * 1000)
    # print(t_julian_factor)
    # t_julian = t_unix + t_julian_factor
    # print(t_julian)
    # t_factor = np.uint64(2451545.0 * 24 * 60 * 60 * 1000)
    # print(t_factor)
    # T = t_julian - t_factor
    t_u2jd = np.float64(2440587.5)
    print(t_u2jd)
    t_julian = (t_unix / t_d2ms) + t_u2jd
    print(t_julian)
    t_factor = np.float64(2451545.0)
    print(t_factor)
    T = (t_julian - t_factor) / 36525
    print(T)
    t_gmst_deg = np.mod(np.float64(280.46061837) + np.float64(360.98564736629) * (t_julian - t_factor) + np.float64(0.000387933) * np.pow(T, 2) - np.pow(T, 3) / np.float64(38710000.), 360)
    print(t_gmst_deg)
    t_gmst = t_gmst_deg / np.float64(15.)
    print(t_gmst)
