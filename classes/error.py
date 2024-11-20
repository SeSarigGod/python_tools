import numpy as np
import warnings
from types import FunctionType

warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)


class Error(object):
    def __init__(self, value: int or float or np.ndarray, error: int or float or np.ndarray or FunctionType):
        match value:
            case list() | tuple():
                value = np.array(value)
            case _:
                pass
        self.value = value

        match error:
            case list() | tuple():
                error = np.array(np.abs(error))
            case FunctionType():
                error = error(np.abs(value))
            case _ if isinstance(self.value, np.ndarray) and not isinstance(error, (np.ndarray, list, tuple, FunctionType)):
                error = np.full(value.shape, np.abs(error))
            case _:
                pass
        self.error = np.abs(error)

        match self.value:
            case np.ndarray():
                self.relative_error = np.array([j / i if i != 0 else j for i, j in zip(self.value, self.error)])
            case _:
                self.relative_error = self.error / self.value if self.value != 0 else self.error
        return

    def __add__(self, other):
        match other:
            case Error():
                return Error(self.value + other.value, np.sqrt(self.error ** 2 + other.error ** 2))
            case list() | tuple() | np.ndarray():
                return Error([self.value + i for i in other], self.error)
            case _:
                try:
                    return Error(self.value + other, self.error)
                except TypeError:
                    raise TypeError(f'Unsupported type for addition: {type(other)}')

    def __radd__(self, other):
        if isinstance(other, Error):
            return Error(other.value + self.value, np.sqrt(self.error**2 + other.error**2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Error([i + self.value for i in other], self.error)
        else:
            try:
                return Error(self.value + other, self.error)
            except TypeError:
                raise TypeError(f'Unsupported type for addition: {type(other)}')

    def __sub__(self, other):
        if isinstance(other, Error):
            return Error(self.value - other.value, np.sqrt(self.error**2 + other.error**2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Error([self.value - i for i in other], self.error)
        else:
            try:
                return Error(other - self.value, self.error)
            except TypeError:
                raise TypeError(f'Unsupported type for subtraction: {type(other)}')

    def __rsub__(self, other):
        if isinstance(other, Error):
            return Error(other.value - self.value, np.sqrt(self.error**2 + other.error**2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Error([i - self.value for i in other], self.error)
        else:
            try:
                return Error(other - self.value, self.error)
            except TypeError:
                raise TypeError(f'Unsupported type for subtraction: {type(other)}')

    def __mul__(self, other):
        if isinstance(other, Error):
            output = self.value * other.value
            return Error(output, output*np.sqrt(self.relative_error**2 + other.relative_error**2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            try:
                return Error([i * j for i, j in zip(self.value, other)], [i * j for i, j in zip(self.value, other)])
            except TypeError:
                return Error([self.value * i for i in other], [self.error * i for i in other])
        else:
            try:
                return Error(self.value * other, self.error * other)
            except TypeError:
                raise TypeError(f'Unsupported type for multiplication: {type(other)}')

    def __rmul__(self, other):
        if isinstance(other, Error):
            output = self.value * other.value
            return Error(output, output * np.sqrt(self.relative_error ** 2 + other.relative_error ** 2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            try:
                return Error([i * j for i, j in zip(self.value, other)], [i * j for i, j in zip(self.value, other)])
            except TypeError:
                return Error([self.value * i for i in other], [self.error * i for i in other])
        else:
            try:
                return Error(self.value * other, self.error * other)
            except TypeError:
                raise TypeError(f'Unsupported type for multiplication: {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, Error):
            output = self.value / other.value
            return Error(output, output*np.sqrt(self.relative_error**2 + other.relative_error**2))
        elif isinstance(other, (list, tuple, np.ndarray)):
            try:
                return Error([i / j for i, j in zip(self.value, other)], [i / j for i, j in zip(self.value, other)])
            except TypeError:
                return Error([self.value / i for i in other], [self.error / i for i in other])
        else:
            try:
                return Error(self.value / other, self.error / other)
            except TypeError:
                raise TypeError(f'Unsupported type for division: {type(other)}')

    def __rtruediv__(self, other):
        if isinstance(other, Error):
            output = other.value / self.value
            return Error(output, output*np.sqrt(self.relative_error**2 + other.relative_error**2))
        elif isinstance(other, (int, float)):
            output = other / self.value
            return Error(other / self.value, output * self.relative_error)
        elif isinstance(other, (list, tuple, np.ndarray)):
            try:
                output = [i / j for i, j in zip(other, self.value)]
                return Error(output, [i * j for i, j in zip(output, self.relative_error)])
            except TypeError:
                output = [i / self.value for i in other]
                return Error(output, [i * self.relative_error for i in output])
        else:
            try:
                output = other / self.value
                return Error(other / self.value, output * self.relative_error)
            except TypeError:
                raise TypeError(f'Unsupported type for division: {type(other)}')

    def __pow__(self, other):
        if isinstance(other, Error):
            output = self.value ** other.value
            return Error(output, output*np.sqrt((other.value * self.relative_error / self.value)**2 + (np.log(self.value) * other.error)**2))
        elif isinstance(other, (int, float)):
            return Error(self.value ** other, np.sqrt(other) * self.error)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Error([self.value ** i for i in other], [i * self.value ** (i - 1) * self.error for i in other])
        else:
            try:
                return Error(self.value ** other, other * self.value ** (other - 1) * self.error)
            except TypeError:
                raise TypeError(f'Unsupported type for exponentiation: {type(other)}')

    def __rpow__(self, other):
        if isinstance(other, Error):
            output = other.value ** self.value
            return Error(output, output*np.sqrt((self.value * other.relative_error / other.value)**2 + (np.log(other.value) * self.error)**2))
        elif isinstance(other, (int, float)):
            return Error(other ** self.value, other ** self.value * np.log(other) * self.error)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Error([i ** self.value for i in other], [i ** self.value * np.log(i) * self.error for i in other])
        else:
            try:
                return Error(other ** self.value, other ** self.value * np.log(other) * self.error)
            except TypeError:
                raise TypeError(f'Unsupported type for exponentiation: {type(other)}')

    def __neg__(self):
        return Error(-self.value, self.error)

    def __abs__(self):
        return Error(np.abs(self.value), self.error)

    def __round__(self, n: int):
        return Error(round(self.value, n), round(self.error, n))

    def __floor__(self):
        return Error(np.floor(self.value), np.floor(self.error))

    def __ceil__(self):
        return Error(np.ceil(self.value), np.ceil(self.error))

    def __trunc__(self):
        return Error(np.trunc(self.value), np.trunc(self.error))

    def __eq__(self, other):
        if isinstance(other, Error):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, Error):
            return self.value != other.value
        else:
            return self.value != other

    def __lt__(self, other):
        if isinstance(other, Error):
            return np.less(self.value, other.value)
        else:
            return np.less(self.value, other)

    def __le__(self, other):
        if isinstance(other, Error):
            return np.less_equal(self.value, other.value)
        else:
            return np.less_equal(self.value, other)

    def __gt__(self, other):
        if isinstance(other, Error):
            return np.greater(self.value, other.value)
        else:
            return np.greater(self.value, other)

    def __ge__(self, other):
        if isinstance(other, Error):
            return np.greater_equal(self.value, other.value)
        else:
            return np.greater_equal(self.value, other)

    def __len__(self):
        return len(self.value), len(self.error)

    def __contains__(self, item):
        return item in self.value

    def __reversed__(self):
        return Error(reversed(self.value), reversed(self.error))

    def __hash__(self):
        return hash(self.value)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    def __index__(self):
        return self.value.__index__()

    def __copy__(self):
        return Error(self.value, self.error)

    def __deepcopy__(self, memo):
        return Error(self.value, self.error)

    def __format__(self, format_spec):
        return Error(format(self.value, format_spec), format(self.error, format_spec))

    def __dir__(self):
        return dir(self)

    def __sizeof__(self):
        return self.value.__sizeof__()

    def __repr__(self):
        return f'{self.value} ± {self.error}'

    def __str__(self):
        return f'{self.value} ± {self.error}'

    def __getitem__(self, index: int):
        try:
            return Error(self.value[index], self.error[index])
        except IndexError:
            raise IndexError('Index out of range')
        except TypeError:
            return self.value if index == 0 else self.error if index == 1 else TypeError('Unsupported type for indexing')

    def __setitem__(self, index, value):
        if isinstance(value, Error):
            self.value[index] = value.value
            self.error[index] = value.error
        else:
            try:
                self.value[index] = value[0]
                self.error[index] = value[1]
            except IndexError:
                raise IndexError('Index out of range')
            except TypeError:
                self.value = value[0]
                self.error = value[1]

    def __iadd__(self, other):
        if isinstance(other, Error):
            self.value += other.value
            self.error = np.sqrt(self.error**2 + other.error**2)
            self.relative_error = np.abs(self.error / self.value)
        elif isinstance(other, (int, float)):
            self.value += other
            self.relative_error = np.abs(self.error / self.value)
        else:
            raise TypeError(f'Unsupported type for addition: {type(other)}')
        return self

    def __isub__(self, other):
        if isinstance(other, Error):
            self.value -= other.value
            self.error = np.sqrt(self.error**2 + other.error**2)
            self.relative_error = np.abs(self.error / self.value)
        elif isinstance(other, (int, float)):
            self.value -= other
            self.relative_error = np.abs(self.error / self.value)
        else:
            raise TypeError(f'Unsupported type for subtraction: {type(other)}')
        return self

    def __imul__(self, other):
        if isinstance(other, Error):
            self.value *= other.value
            self.error = np.abs(self.value*np.sqrt(self.relative_error**2 + other.relative_error**2))
            self.relative_error = np.abs(self.error / self.value)
        elif isinstance(other, (int, float)):
            self.value *= other
            self.error *= np.abs(other)
        else:
            raise TypeError(f'Unsupported type for multiplication: {type(other)}')
        return self

    def __itruediv__(self, other):
        if isinstance(other, Error):
            self.value /= other.value
            self.error = np.abs(self.value*np.sqrt(self.relative_error**2 + other.relative_error**2))
            self.relative_error = np.abs(self.error / self.value)
        elif isinstance(other, (int, float)):
            self.value /= other
            self.error /= np.abs(other)
        else:
            raise TypeError(f'Unsupported type for division: {type(other)}')
        return self

    def __ipow__(self, other):
        if isinstance(other, Error):
            self.value **= other.value
            self.error = np.abs(self.value*np.sqrt((other.value * self.relative_error / self.value)**2 + (np.log(self.value) * other.error)**2))
            self.relative_error = np.abs(self.error / self.value)
        elif isinstance(other, (int, float)):
            self.value **= other
            self.error *= np.abs(other * self.value**(other - 1))
        else:
            raise TypeError(f'Unsupported type for exponentiation: {type(other)}')
        return self

    def sig_figs(self, n: int):
        if isinstance(self.value, np.ndarray):
            return Error(np.array([round(i, n - int(np.floor(np.log10(np.abs(i)))) - 1) if i != 0 else i for i in self.value]),
                         np.array([round(i, n - int(np.floor(np.log10(np.abs(i)))) - 1) if i != 0 else i for i in self.error]))
        else:
            return Error(round(self.value, n - int(np.floor(np.log10(np.abs(self.value)))) - 1) if self.value != 0 else self.value,
                         round(self.error, n - int(np.floor(np.log10(np.abs(self.error)))) - 1) if self.error != 0 else self.error)

    def sign(self):
        if isinstance(self.value, np.ndarray):
            return np.array([np.sign(i) for i in self.value])
        else:
            return np.sign(self.value)

    def append(self, other):
        self.value = np.append(self.value, other.value)
        self.error = np.append(self.error, other.error)
        self.relative_error = np.abs(self.error / self.value)
        return self

    def extend(self, other):
        self.value = np.append(self.value, other)
        self.error = np.append(self.error, np.zeros(len(other)))
        self.relative_error = np.abs(self.error / self.value)
        return self

    def pop(self, index: int):
        self.value = np.delete(self.value, index)
        self.error = np.delete(self.error, index)
        self.relative_error = np.abs(self.error / self.value)
        return self

    def remove(self, value):
        index = np.where(self.value == value)
        self.value = np.delete(self.value, index)
        self.error = np.delete(self.error, index)
        self.relative_error = np.abs(self.error / self.value)
        return self

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

def sign(x):
    if isinstance(x, Error):
        return x.sign()
    else:
        try:
            return np.array([np.sign(i) for i in x])
        except TypeError:
            return np.sign(x)

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
