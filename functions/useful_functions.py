import numpy as np


def clamp(value: int or float, min_val: int or float = 0, max_val: int or float = 0, restrict: bool = False) -> int or float:
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

def sign(number: any) -> int:
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
            checked_points: list = []
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
            left = data[max(0, i - depth) : i]
            right = data[i+1 : min(length, i + 1 + depth)]
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