from python_tools.classes.error import *


def clamp(value: int or float, min_val: int or float = 0,
          max_val: int or float = 0, restrict: bool = False) -> int or float:
    if restrict:
        return max(min_val, min(value, max_val))
    else:
        return max(min_val, value)

def split_number(number: int, n: int) -> list[int]:
    return [number // n + (1 if x < number % n else 0) for x in range(n)]

def split_list(lst: list[any], n: int) -> list[list[any]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def split_array(arr: np.ndarray[any], n: int) -> list[np.ndarray[any]]:
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def split_string(string: str, n: int) -> list[str]:
    return [string[i:i + n] for i in range(0, len(string), n)]

def sign(number: any) -> list[int] or int:
    try:
        return [(i > 0) - (i < 0) for i in number]
    except TypeError("Input must be an iterable"):
        return (number > 0) - (number < 0)

def find_peaks(data, depth: int = None, stds: float = 3.0):
    # Initialize the peaks and troughs lists
    peaks = [(0, 0)]
    troughs = [(0, 0)]

    # Get statistics of the data that will be used to filter the peaks and troughs
    length = len(data)
    # max_value = np.max(data)
    # min_value = np.min(data)
    # std = stds*np.std(data)

    # If the depth is not specified, set it to 1/10 of the length of the data
    if depth is None:
        depth = int(length/10)

    # Find the peaks and troughs,
    for i in range(length):

        # Check if the index is the start or end and check data accordingly
        if i == 0 or i == length - 1:
            # checked_points: list = []
            if i == 0:
                checked_points = data[1: min(length, 1 + depth)]
            elif i == length - 1:
                checked_points = data[max(0, i - depth): i]
            else:
                raise ValueError('Index is not the start or end. How did this happen???')

            checked_points_std = stds*np.std(checked_points)
            checked_points_avg = np.average(checked_points)
            if data[i] >= np.max(checked_points) and data[i] >= checked_points_avg+checked_points_std:
                peaks.append((i, data[i]))
            if data[i] <= np.min(checked_points) and data[i] <= checked_points_avg-checked_points_std:
                troughs.append((i, data[i]))

        # Check the rest of the indices
        else:
            left = data[max(0, i - depth): i]
            right = data[i+1: min(length, i + 1 + depth)]
            lr = np.concatenate((left, right))
            lr_std = stds*np.std(lr)
            lr_avg = np.average(lr)
            if data[i] >= np.max(left) and data[i] >= np.max(right) and data[i] >= lr_avg+lr_std:
                if (np.abs(peaks[-1][1] - data[i]) > 0.01*np.abs(data[i])) or np.abs(peaks[-1][0] - i) > depth:
                    peaks.append((i, data[i]))
            if data[i] <= np.min(left) and data[i] <= np.min(right) and data[i] <= lr_avg-lr_std:
                if (np.abs(troughs[-1][1] - data[i]) > 0.01*np.abs(data[i])) or (np.abs(troughs[-1][0] - i) > depth):
                    troughs.append((i, data[i]))

    # Remove the initial values
    peaks.pop(0)
    troughs.pop(0)

    return peaks, troughs

def percent_error(measured, actual):
    return ((measured - actual) / actual) * 100

def careful_average(x, y):
    if isinstance(x, Error) and isinstance(y, Error):
        try:
            signs = [np.sign(i) for i in y.value]
        except TypeError:
            signs = np.sign(y.value)
        return signs * (np.abs(x) + np.abs(y)) / 2
    try:
        signs = [np.sign(i) for i in y]
    except TypeError:
        signs = np.sign(y)
    return signs * (np.abs(x) + np.abs(y)) / 2

def sig_figs(x, n: int):
    if isinstance(x, Error):
        return x.sig_figs(n)
    else:
        return round(x, n - int(np.floor(np.log10(np.abs(x)))) - 1) if x != 0 else x

def linear_fit(x, m, b):
    return m * x + b

def quadratic_fit(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_fit(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def exponential_fit(x, a, b):
    return a * np.exp(b * x)

def power_fit(x, a, b):
    return a * x ** b

def logarithmic_fit(x, a, b):
    return a * np.log(b * x)

def inverse_fit(x, a, b):
    return a / x + b

def sigmoid_fit(x, a, b, c):
    return a / (1 + np.exp(b * (x - c)))

def gaussian_fit(x, a, b, c, d):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d

def lorentzian_fit(x, a, b, c, d):
    return a / ((x - b) ** 2 + c ** 2) + d

def chi_squared(observed, expected):
    return np.sum(((observed - expected) ** 2) / expected)
