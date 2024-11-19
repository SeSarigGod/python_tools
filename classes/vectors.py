import math
import json
import pickle


class Vector2(object):
    def __init__(self, vector: not str = (0, 0)) -> None:
        self.xy = None
        if isinstance(vector, Vector2) or isinstance(vector, Vector4) or isinstance(vector, Vector4):
            self.x = vector.x
            self.y = vector.y
        elif isinstance(vector, float) or isinstance(vector, int):
            self.x = vector
            self.y = vector
        else:
            self.x = vector[0]
            self.y = vector[1]
        self.update_subsets()

    def update_subsets(self):
        super().__setattr__('xy', (self.x, self.y))

    def __repr__(self) -> str:
        return f"(x: {self.x}, y: {self.y})"

    def __get__(self, instance, owner) -> 'Vector2':
        return self

    def __set__(self, instance, value: 'Vector2') -> None:
        self.x = value.x
        self.y = value.y
        self.update_subsets()

    def __delete__(self, instance) -> None:
        del self

    def __call__(self, _x: any, _y: any) -> None:
        self.x = _x
        self.y = _y
        self.update_subsets()

    def __eq__(self, other: 'Vector2') -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: 'Vector2') -> bool:
        return self.x != other.x or self.y != other.y

    def __setattr__(self, name: str, value: any) -> None:
        attrs = ['x', 'y', 'xy']
        try:
            if len(value) != len(name):
                raise ValueError(f"Length of {value} must be {len(name)}, not {len(value)}.")
        except TypeError:
            super().__setattr__(name, value)
        else:
            if isinstance(value, Vector4) or isinstance(value, Vector3) or isinstance(value, Vector2):
                super().__setattr__('x', value.x)
                super().__setattr__('y', value.y)
            elif name in attrs:
                for dim, val in zip(name, value):
                    super().__setattr__(dim, val)
            else:
                raise AttributeError(f"Attribute {name} not found.")
            self.update_subsets()

    def __getattr__(self, key: str) -> any:
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def __delattr__(self, key: str) -> None:
        if key == 'x':
            del self.x
        elif key == 'y':
            del self.y
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def __int__(self) -> 'Vector2':
        return Vector2((int(self.x), int(self.y)))

    def __float__(self) -> 'Vector2':
        return Vector2((float(self.x), float(self.y)))

    def __str__(self) -> str:
        return f"(x: {self.x}, y: {self.y})"

    def __dict__(self) -> dict:
        return {'x': self.x, 'y': self.y}

    def __lt__(self, other: 'Vector2') -> bool:
        return self.magnitude() < other.magnitude()

    def __le__(self, other: 'Vector2') -> bool:
        return self.magnitude() <= other.magnitude()

    def __gt__(self, other: 'Vector2') -> bool:
        return self.magnitude() > other.magnitude()

    def __ge__(self, other: 'Vector2') -> bool:
        return self.magnitude() >= other.magnitude()

    def __add__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((self.x + other.x, self.y + other.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((self.x + other[0], self.y + other[1]))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((self.x + other, self.y + other))
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector2' and {type(other)}")

    def __radd__(self, other: not str) -> 'Vector2':
        return self.__add__(other)

    def __sub__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((self.x - other.x, self.y - other.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((self.x - other[0], self.y - other[1]))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((self.x - other, self.y - other))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector2' and {type(other)}")

    def __rsub__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((other.x - self.x, other.y - self.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((other[0] - self.x, other[1] - self.y))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((other - self.x, other - self.y))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector2' and {type(other)}")

    def __mul__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((self.x * other.x, self.y * other.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((self.x * other[0], self.y * other[1]))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((self.x * other, self.y * other))
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector2' and {type(other)}")

    def __rmul__(self, other: not str) -> 'Vector2':
        return self.__mul__(other)

    def __truediv__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((self.x / other.x, self.y / other.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((self.x / other[0], self.y / other[1]))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((self.x / other, self.y / other))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector2' and {type(other)}")

    def __rtruediv__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            return Vector2((other.x / self.x, other.y / self.y))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                return Vector2((other[0] / self.x, other[1] / self.y))
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector2((other / self.x, other / self.y))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector2' and {type(other)}")

    def __iadd__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            self.x += other.x
            self.y += other.y
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                self.x += other[0]
                self.y += other[1]
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x += other
            self.y += other
        else:
            raise TypeError(f"Unsupported operand type(s) for +=: 'Vector2' and {type(other)}")
        self.update_subsets()
        return self

    def __isub__(self, other: not str) -> 'Vector2':
        if isinstance(other, Vector2):
            self.x -= other.x
            self.y -= other.y
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 2:
                self.x -= other[0]
                self.y -= other[1]
            else:
                raise ValueError(f"Length of {type(other)} must be 2, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x -= other
            self.y -= other
        else:
            raise TypeError(f"Unsupported operand type(s) for -=: 'Vector2' and {type(other)}")
        self.update_subsets()
        return self

    def __imul__(self, other: not str) -> 'Vector2':
        if isinstance(other, float) or isinstance(other, int):
            self.x *= other
            self.y *= other
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector2' and {type(other)}")
        self.update_subsets()
        return self

    def __itruediv__(self, other: not str) -> 'Vector2':
        if isinstance(other, float) or isinstance(other, int):
            self.x /= other
            self.y /= other
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector2' and {type(other)}")
        self.update_subsets()
        return self

    def set(self, key: str, value: any) -> 'Vector2':
        if key == 'x':
            self.x = value
        elif key == 'y':
            self.y = value
        elif key == 'vector':
            self.x = value[0]
            self.y = value[1]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def update(self, key: str, value: any) -> 'Vector2':
        if key == 'x':
            self.x += value
        elif key == 'y':
            self.y += value
        elif key == 'vector':
            self.x += value[0]
            self.y += value[1]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def get(self, key: str) -> any:
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'vector':
            return self.x, self.y
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def magnitude(self) -> float or int:
        return math.sqrt(self.x**2 + self.y**2)

    def unit_vector(self) -> 'Vector2':
        return Vector2((self.x / self.magnitude(), self.y / self.magnitude()))

    def angle(self, other: 'Vector2') -> float:
        return math.acos((self.x * other.x + self.y * other.y) / (self.magnitude() * other.magnitude()))

    def dot_product(self, other: 'Vector2') -> float or int:
        return self.x * other.x + self.y * other.y

    def cross_product(self) -> 'Vector2':
        return Vector2((self.y, -self.x))

    def normal(self) -> 'Vector2':
        return Vector2((-self.y, self.x))

    def homogenize(self) -> 'Vector2':
        return Vector2((self.x / self.y, 1))

    def project(self, other: 'Vector2') -> 'Vector2':
        return other.unit_vector() * self.dot_product(other.unit_vector())

    def reject(self, other: 'Vector2') -> 'Vector2':
        return self - self.project(other)

    def reflect(self, other: 'Vector2') -> 'Vector2':
        return self - self.project(other) * 2

    def refract(self, other: 'Vector2', n1: float, n2: float) -> 'Vector2':
        cos_theta = self.unit_vector().dot_product(other.unit_vector())
        sin_theta = math.sqrt(1 - cos_theta**2)
        sin_phi = n1 / n2 * sin_theta
        cos_phi = math.sqrt(1 - sin_phi**2)
        return self.unit_vector() * n1 - other.unit_vector() * (n1 / n2 * cos_theta - cos_phi)

    def rotate(self, angle: float) -> 'Vector2':
        return Vector2((self.x * math.cos(angle) - self.y * math.sin(angle),
                        self.x * math.sin(angle) + self.y * math.cos(angle)))

    def lerp(self, other: 'Vector2', t: float) -> 'Vector2':
        return self * (1 - t) + other * t

    def slerp(self, other: 'Vector2', t: float) -> 'Vector2':
        omega = self.angle(other)
        sin_omega = math.sin(omega)
        return self * (math.sin((1 - t) * omega) / sin_omega) + other * (math.sin(t * omega) / sin_omega)

    def nlerp(self, other: 'Vector2', t: float) -> 'Vector2':
        return self.lerp(other, t).unit_vector()

    def nslerp(self, other: 'Vector2', t: float) -> 'Vector2':
        return self.slerp(other, t).unit_vector()

    def to_vector3(self) -> 'Vector3':
        return Vector3((self.x, self.y, 0))

    def to_vector4(self) -> 'Vector4':
        return Vector4((self.x, self.y, 0, 0))

    def to_homogeneous_vector3(self) -> 'Vector3':
        return Vector3((self.x, self.y, 1))

    def to_homogeneous_vector4(self) -> 'Vector4':
        return Vector4((self.x, self.y, 1, 1))

    def to_list(self) -> list:
        return [self.x, self.y]

    def to_tuple(self) -> tuple:
        return tuple((self.x, self.y))

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y}

    def to_json(self) -> str:
        return json.dumps({'x': self.x, 'y': self.y})

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    def to_string(self) -> str:
        return f"Vector2({self.x}, {self.y})"

    @staticmethod
    def from_list(array: list) -> 'Vector2':
        return Vector2((array[0], array[1]))

    @staticmethod
    def from_tuple(array: tuple) -> 'Vector2':
        return Vector2((array[0], array[1]))

    @staticmethod
    def from_dict(array: dict) -> 'Vector2':
        return Vector2((array['x'], array['y']))

    @staticmethod
    def from_bytes(array: bytes) -> 'Vector2':
        return pickle.loads(array)

    @staticmethod
    def from_vector4(array: 'Vector4') -> 'Vector2':
        return Vector2((array.x, array.y))

    @staticmethod
    def from_vector3(array: 'Vector3') -> 'Vector2':
        return Vector2((array.x, array.y))

class Vector3(object):
    def __init__(self, vector: not str = (0.0, 0.0, 0.0)) -> None:
        self.xy = None
        self.yz = None
        self.xz = None
        self.xyz = None
        if isinstance(vector, Vector3) or isinstance(vector, Vector3):
            self.x = vector.x
            self.y = vector.y
            self.z = vector.z
        elif isinstance(vector, float) or isinstance(vector, int):
            self.x = vector
            self.y = vector
            self.z = vector
        else:
            self.x = vector[0]
            self.y = vector[1]
            self.z = vector[2]
        self.update_subsets()

    def update_subsets(self):
        super().__setattr__('xy', (self.x, self.y))
        super().__setattr__('yz', (self.y, self.z))
        super().__setattr__('xz', (self.x, self.z))
        super().__setattr__('xyz', (self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"(x: {self.x}, y: {self.y}, z: {self.z})"

    def __get__(self, instance, owner) -> 'Vector3':
        return self

    def __set__(self, instance, value: 'Vector3') -> None:
        self.x = value.x
        self.y = value.y
        self.z = value.z
        self.update_subsets()

    def __delete__(self, instance) -> None:
        del self

    def __call__(self, _x: any, _y: any, _z: any) -> None:
        self.x = _x
        self.y = _y
        self.z = _z
        self.update_subsets()

    def __eq__(self, other: 'Vector3') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: 'Vector3') -> bool:
        return self.x != other.x or self.y != other.y or self.z != other.z

    def __aiter__(self):
        return self

    def __int__(self) -> 'Vector3':
        return Vector3((int(self.x), int(self.y), int(self.z)))

    def __float__(self) -> 'Vector3':
        return Vector3((float(self.x), float(self.y), float(self.z)))

    def __str__(self) -> str:
        return f"(x: {self.x}, y: {self.y}, z: {self.z})"

    def __dict__(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def __iter__(self):
        return iter(self.xyz)

    def __setattr__(self, name: str, value: any) -> None:
        attrs = ['x', 'y', 'z', 'xyz', 'xy', 'xz', 'yz']
        try:
            if len(value) != len(name):
                raise ValueError(f"Length of {value} must be {len(name)}, not {len(value)}.")
        except TypeError:
            super().__setattr__(name, value)
        else:
            if isinstance(value, Vector4) or isinstance(value, Vector3):
                super().__setattr__('x', value.x)
                super().__setattr__('y', value.y)
                super().__setattr__('z', value.z)
            elif name in attrs:
                for dim, val in zip(name, value):
                    super().__setattr__(dim, val)
            else:
                raise AttributeError(f"Attribute {name} not found.")
            self.update_subsets()

    def __getattr__(self, key: str) -> any:
        attrs = ['x', 'y', 'z', 'xyz', 'xy', 'xz', 'yz']
        if key not in attrs:
            raise AttributeError(f"Attribute {key} not found.")
        else:
            return super().__getattribute__(key)

    def __delattr__(self, key: str) -> None:
        if key == 'x':
            del self.x
        elif key == 'y':
            del self.y
        elif key == 'z':
            del self.z
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def __lt__(self, other: 'Vector3') -> bool:
        return self.magnitude() < other.magnitude()

    def __le__(self, other: 'Vector3') -> bool:
        return self.magnitude() <= other.magnitude()

    def __gt__(self, other: 'Vector3') -> bool:
        return self.magnitude() > other.magnitude()

    def __ge__(self, other: 'Vector3') -> bool:
        return self.magnitude() >= other.magnitude()

    def __add__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((self.x + other.x, self.y + other.y, self.z + other.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((self.x + other[0], self.y + other[1], self.z + other[2]))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((self.x + other, self.y + other, self.z + other))
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector3' and {type(other)}")

    def __radd__(self, other: not str) -> 'Vector3':
        return self.__add__(other)

    def __sub__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((self.x - other.x, self.y - other.y, self.z - other.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((self.x - other[0], self.y - other[1], self.z - other[2]))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((self.x - other, self.y - other, self.z - other))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector3' and {type(other)}")

    def __rsub__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((other.x - self.x, other.y - self.y, other.z - self.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((other[0] - self.x, other[1] - self.y, other[2] - self.z))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((other - self.x, other - self.y, other - self.z))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector3' and {type(other)}")

    def __mul__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((self.x * other.x, self.y * other.y, self.z * other.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((self.x * other[0], self.y * other[1], self.z * other[2]))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((self.x * other, self.y * other, self.z * other))
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector3' and {type(other)}")

    def __rmul__(self, other: not str) -> 'Vector3':
        return self.__mul__(other)

    def __truediv__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((self.x / other.x, self.y / other.y, self.z / other.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((self.x / other[0], self.y / other[1], self.z / other[2]))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((self.x / other, self.y / other, self.z / other))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector3' and {type(other)}")

    def __rtruediv__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3((other.x / self.x, other.y / self.y, other.z / self.z))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                return Vector3((other[0] / self.x, other[1] / self.y, other[2] / self.z))
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3((other / self.x, other / self.y, other / self.z))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector3' and {type(other)}")

    def __iadd__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                self.x += other[0]
                self.y += other[1]
                self.z += other[2]
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x += other
            self.y += other
            self.z += other
        else:
            raise TypeError(f"Unsupported operand type(s) for +=: 'Vector3' and {type(other)}")
        self.update_subsets()
        return self

    def __isub__(self, other: not str) -> 'Vector3':
        if isinstance(other, Vector3):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 3:
                self.x -= other[0]
                self.y -= other[1]
                self.z -= other[2]
            else:
                raise ValueError(f"Length of {type(other)} must be 3, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x -= other
            self.y -= other
            self.z -= other
        else:
            raise TypeError(f"Unsupported operand type(s) for -=: 'Vector3' and {type(other)}")
        self.update_subsets()
        return self

    def __imul__(self, other: not str) -> 'Vector3':
        if isinstance(other, float) or isinstance(other, int):
            self.x *= other
            self.y *= other
            self.z *= other
            self.update_subsets()
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector3' and {type(other)}")

    def __itruediv__(self, other: not str) -> 'Vector3':
        if isinstance(other, float) or isinstance(other, int):
            self.x /= other
            self.y /= other
            self.z /= other
            self.update_subsets()
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector3' and {type(other)}")

    def set(self, key: str, value: any) -> 'Vector3':
        if key == 'x':
            self.x = value
        elif key == 'y':
            self.y = value
        elif key == 'z':
            self.z = value
        elif key == 'vector':
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def update(self, key: str, value: any) -> 'Vector3':
        if key == 'x':
            self.x += value
        elif key == 'y':
            self.y += value
        elif key == 'z':
            self.z += value
        elif key == 'vector':
            self.x += value[0]
            self.y += value[1]
            self.z += value[2]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def get(self, key: str) -> any:
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'z':
            return self.z
        elif key == 'vector':
            return self.x, self.y, self.z
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def magnitude(self) -> float or int:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unit_vector(self) -> 'Vector3':
        return Vector3((self.x / self.magnitude(), self.y / self.magnitude(), self.z / self.magnitude()))

    def angle(self, other: 'Vector3') -> float:
        return math.acos((self.x * other.x + self.y * other.y + self.z * other.z) /
                         (self.magnitude() * other.magnitude()))

    def dot_product(self, other: 'Vector3') -> float or int:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross_product(self, other: 'Vector3') -> 'Vector3':
        return Vector3((self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x))

    def normal(self) -> 'Vector3':
        return Vector3((-self.y, self.x, self.z))

    def homogenize(self) -> 'Vector3':
        return Vector3((self.x / self.z, self.y / self.z, 1))

    def project(self, other: 'Vector3') -> 'Vector3':
        return other.unit_vector() * self.dot_product(other.unit_vector())

    def reject(self, other: 'Vector3') -> 'Vector3':
        return self - self.project(other)

    def reflect(self, other: 'Vector3') -> 'Vector3':
        return self - self.project(other) * 2

    def refract(self, other: 'Vector3', n1: float, n2: float) -> 'Vector3':
        cos_theta = self.unit_vector().dot_product(other.unit_vector())
        sin_theta = math.sqrt(1 - cos_theta**2)
        sin_phi = n1 / n2 * sin_theta
        cos_phi = math.sqrt(1 - sin_phi**2)
        return self.unit_vector() * n1 - other.unit_vector() * (n1 / n2 * cos_theta - cos_phi)

    def rotate(self, angle: float, axis: 'Vector3') -> 'Vector3':
        return (self * math.cos(angle) +
                self.cross_product(axis) * math.sin(angle) +
                self * (1 - math.cos(angle)) * axis.dot_product(self))

    def lerp(self, other: 'Vector3', t: float) -> 'Vector3':
        return self * (1 - t) + other * t

    def slerp(self, other: 'Vector3', t: float) -> 'Vector3':
        omega = self.angle(other)
        sin_omega = math.sin(omega)
        return self * (math.sin((1 - t) * omega) / sin_omega) + other * (math.sin(t * omega) / sin_omega)

    def nlerp(self, other: 'Vector3', t: float) -> 'Vector3':
        return self.lerp(other, t).unit_vector()

    def nslerp(self, other: 'Vector3', t: float) -> 'Vector3':
        return self.slerp(other, t).unit_vector()

    def to_vector2(self) -> 'Vector2':
        return Vector2((self.x, self.y))

    def to_vector4(self) -> 'Vector4':
        return Vector4((self.x, self.y, self.z, 0))

    def to_homogeneous(self) -> 'Vector4':
        return Vector4((self.x, self.y, self.z, 1))

    def to_list(self) -> list:
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple:
        return tuple((self.x, self.y, self.z))

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def to_json(self) -> str:
        return json.dumps({'x': self.x, 'y': self.y, 'z': self.z})

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    def to_string(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    @staticmethod
    def from_list(array: list) -> 'Vector3':
        return Vector3((array[0], array[1], array[2]))

    @staticmethod
    def from_tuple(array: tuple) -> 'Vector3':
        return Vector3((array[0], array[1], array[2]))

    @staticmethod
    def from_dict(array: dict) -> 'Vector3':
        return Vector3((array['x'], array['y'], array['z']))

    @staticmethod
    def from_bytes(array: bytes) -> 'Vector3':
        return pickle.loads(array)

    @staticmethod
    def from_vector4(array: 'Vector4') -> 'Vector3':
        return Vector3((array.x, array.y, array.z))

    @staticmethod
    def from_vector2(array: 'Vector2') -> 'Vector3':
        return Vector3((array.x, array.y, 0))

    @staticmethod
    def from_vector2_homogenous(array: 'Vector2') -> 'Vector3':
        return Vector3((array.x, array.y, 1))

class Vector4(object):
    def __init__(self, vector: not str = (0, 0, 0, 0)) -> None:
        self.xyzw = None
        self.xyz = None
        self.xyw = None
        self.xzw = None
        self.yzw = None
        self.xy = None
        self.yz = None
        self.xz = None
        self.xw = None
        self.yw = None
        self.zw = None
        if isinstance(vector, Vector4):
            self.x = vector.x
            self.y = vector.y
            self.z = vector.z
            self.w = vector.w
        elif isinstance(vector, float) or isinstance(vector, int):
            self.x = vector
            self.y = vector
            self.z = vector
            self.w = vector
        else:
            self.x = vector[0]
            self.y = vector[1]
            self.z = vector[2]
            self.w = vector[3]
        self.update_subsets()

    def update_subsets(self):
        super().__setattr__('xy', (self.x, self.y))
        super().__setattr__('yz', (self.y, self.z))
        super().__setattr__('xz', (self.x, self.z))
        super().__setattr__('xw', (self.x, self.w))
        super().__setattr__('yw', (self.y, self.w))
        super().__setattr__('zw', (self.z, self.w))
        super().__setattr__('xyz', (self.x, self.y, self.z))
        super().__setattr__('xyw', (self.x, self.y, self.w))
        super().__setattr__('xzw', (self.x, self.z, self.w))
        super().__setattr__('yzw', (self.y, self.z, self.w))
        super().__setattr__('xyzw', (self.x, self.y, self.z, self.w))

    def __repr__(self) -> str:
        return f"(x: {self.x}, y: {self.y}, z: {self.z}, w: {self.w})"

    def __get__(self, instance, owner) -> 'Vector4':
        return self

    def __set__(self, instance, value: 'Vector4') -> None:
        self.x = value.x
        self.y = value.y
        self.z = value.z
        self.w = value.w
        self.update_subsets()

    def __delete__(self, instance) -> None:
        del self

    def __call__(self, _x: any, _y: any, _z: any, _w: any) -> None:
        self.x = _x
        self.y = _y
        self.z = _z
        self.w = _w
        self.update_subsets()

    def __eq__(self, other: 'Vector4') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w

    def __ne__(self, other: 'Vector4') -> bool:
        return self.x != other.x or self.y != other.y or self.z != other.z or self.w != other.w

    def __setattr__(self, name: str, value: any) -> None:
        attrs = ['x', 'y', 'z', 'w', 'xyzw', 'xyz', 'xyw', 'xzw', 'yzw', 'xy', 'yz', 'xz', 'xw', 'yw', 'zw']
        try:
            if len(value) != len(name):
                raise ValueError(f"Length of {value} must be {len(name)}, not {len(value)}.")
        except TypeError:
            super().__setattr__(name, value)
        else:
            if isinstance(value, Vector4):
                super().__setattr__('x', value.x)
                super().__setattr__('y', value.y)
                super().__setattr__('z', value.z)
                super().__setattr__('w', value.w)
            elif name in attrs:
                for dim, val in zip(name, value):
                    super().__setattr__(dim, val)
            else:
                raise AttributeError(f"Attribute {name} not found.")
            self.update_subsets()

    def __getattr__(self, key: str) -> any:
        attrs = ['x', 'y', 'z', 'w', 'xyzw', 'xyz', 'xyw', 'xzw', 'yzw', 'xy', 'yz', 'xz', 'xw', 'yw', 'zw']
        if key not in attrs:
            raise AttributeError(f"Attribute {key} not found.")
        else:
            return super().__getattribute__(key)

    def __delattr__(self, key: str) -> None:
        if key == 'x':
            del self.x
        elif key == 'y':
            del self.y
        elif key == 'z':
            del self.z
        elif key == 'w':
            del self.w
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def __int__(self) -> 'Vector4':
        return Vector4((int(self.x), int(self.y), int(self.z), int(self.w)))

    def __float__(self) -> 'Vector4':
        return Vector4((float(self.x), float(self.y), float(self.z), float(self.w)))

    def __str__(self) -> str:
        return f"(x: {self.x}, y: {self.y}, z: {self.z}, w: {self.w})"

    def __dict__(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}

    def __lt__(self, other: 'Vector4') -> bool:
        return self.magnitude() < other.magnitude()

    def __le__(self, other: 'Vector4') -> bool:
        return self.magnitude() <= other.magnitude()

    def __gt__(self, other: 'Vector4') -> bool:
        return self.magnitude() > other.magnitude()

    def __ge__(self, other: 'Vector4') -> bool:
        return self.magnitude() >= other.magnitude()

    def __add__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((self.x + other[0], self.y + other[1], self.z + other[2], self.w + other[3]))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((self.x + other, self.y + other, self.z + other, self.w + other))
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector4' and {type(other)}")

    def __radd__(self, other: not str) -> 'Vector4':
        return self.__add__(other)

    def __sub__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((self.x - other[0], self.y - other[1], self.z - other[2], self.w - other[3]))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((self.x - other, self.y - other, self.z - other, self.w - other))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector4' and {type(other)}")

    def __rsub__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((other.x - self.x, other.y - self.y, other.z - self.z, other.w - self.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((other[0] - self.x, other[1] - self.y, other[2] - self.z, other[3] - self.w))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((other - self.x, other - self.y, other - self.z, other - self.w))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector4' and {type(other)}")

    def __mul__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((self.x * other[0], self.y * other[1], self.z * other[2], self.w * other[3]))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((self.x * other, self.y * other, self.z * other, self.w * other))
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector4' and {type(other)}")

    def __rmul__(self, other: not str) -> 'Vector4':
        return self.__mul__(other)

    def __truediv__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((self.x / other[0], self.y / other[1], self.z / other[2], self.w / other[3]))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((self.x / other, self.y / other, self.z / other, self.w / other))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector4' and {type(other)}")

    def __rtruediv__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            return Vector4((other.x / self.x, other.y / self.y, other.z / self.z, other.w / self.w))
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                return Vector4((other[0] / self.x, other[1] / self.y, other[2] / self.z, other[3] / self.w))
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            return Vector4((other / self.x, other / self.y, other / self.z, other / self.w))
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector4' and {type(other)}")

    def __iadd__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            self.w += other.w
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                self.x += other[0]
                self.y += other[1]
                self.z += other[2]
                self.w += other[3]
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x += other
            self.y += other
            self.z += other
            self.w += other
        else:
            raise TypeError(f"Unsupported operand type(s) for +=: 'Vector4' and {type(other)}")
        self.update_subsets()
        return self

    def __isub__(self, other: not str) -> 'Vector4':
        if isinstance(other, Vector4):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
            self.w -= other.w
        elif isinstance(other, list) or isinstance(other, tuple):
            if len(other) == 4:
                self.x -= other[0]
                self.y -= other[1]
                self.z -= other[2]
                self.w -= other[3]
            else:
                raise ValueError(f"Length of {type(other)} must be 4, not {len(other)}.")
        elif isinstance(other, float) or isinstance(other, int):
            self.x -= other
            self.y -= other
            self.z -= other
            self.w -= other
        else:
            raise TypeError(f"Unsupported operand type(s) for -=: 'Vector4' and {type(other)}")
        self.update_subsets()
        return self

    def __imul__(self, other: not str) -> 'Vector4':
        if isinstance(other, float) or isinstance(other, int):
            self.x *= other
            self.y *= other
            self.z *= other
            self.w *= other
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector4' and {type(other)}")
        self.update_subsets()
        return self

    def __itruediv__(self, other: not str) -> 'Vector4':
        if isinstance(other, float) or isinstance(other, int):
            self.x /= other
            self.y /= other
            self.z /= other
            self.w /= other
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector4' and {type(other)}")
        self.update_subsets()
        return self

    def set(self, key: str, value: any) -> 'Vector4':
        if key == 'x':
            self.x = value
        elif key == 'y':
            self.y = value
        elif key == 'z':
            self.z = value
        elif key == 'w':
            self.w = value
        elif key == 'vector':
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
            self.w = value[3]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def update(self, key: str, value: any) -> 'Vector4':
        if key == 'x':
            self.x += value
        elif key == 'y':
            self.y += value
        elif key == 'z':
            self.z += value
        elif key == 'w':
            self.w += value
        elif key == 'vector':
            self.x += value[0]
            self.y += value[1]
            self.z += value[2]
            self.w += value[3]
        else:
            raise AttributeError(f"Attribute {key} not found.")
        self.update_subsets()
        return self

    def get(self, key: str) -> any:
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'z':
            return self.z
        elif key == 'w':
            return self.w
        elif key == 'vector':
            return self.x, self.y, self.z, self.w
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def magnitude(self) -> float or int:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def unit_vector(self) -> 'Vector4':
        return Vector4((self.x / self.magnitude(),
                        self.y / self.magnitude(),
                        self.z / self.magnitude(),
                        self.w / self.magnitude()))

    def angle(self, other: 'Vector4') -> float:
        return math.acos((self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w) /
                         (self.magnitude() * other.magnitude()))

    def dot_product(self, other: 'Vector4') -> float or int:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    # def cross_product(self, other1: 'Vector4', other2: 'Vector4') -> 'Vector4':
    #     return Vector4(())

    def normal(self) -> 'Vector4':
        return Vector4((-self.y, self.x, self.z, self.w))

    def homogenize(self) -> 'Vector4':
        return Vector4((self.x / self.w, self.y / self.w, self.z / self.w, 1))

    def project(self, other: 'Vector4') -> 'Vector4':
        return other.unit_vector() * self.dot_product(other.unit_vector())

    def reject(self, other: 'Vector4') -> 'Vector4':
        return self - self.project(other)

    def reflect(self, other: 'Vector4') -> 'Vector4':
        return self - self.project(other) * 2

    def refract(self, other: 'Vector4', n1: float, n2: float) -> 'Vector4':
        cos_theta = self.unit_vector().dot_product(other.unit_vector())
        sin_theta = math.sqrt(1 - cos_theta**2)
        sin_phi = n1 / n2 * sin_theta
        cos_phi = math.sqrt(1 - sin_phi**2)
        return self.unit_vector() * n1 - other.unit_vector() * (n1 / n2 * cos_theta - cos_phi)

    def rotate(self, angle: float, axis: 'Vector4') -> 'Vector4':
        return (self * math.cos(angle) +
                self.cross_product(axis) * math.sin(angle) +
                self * (1 - math.cos(angle)) * axis.dot_product(self))

    def lerp(self, other: 'Vector4', t: float) -> 'Vector4':
        return self * (1 - t) + other * t

    def slerp(self, other: 'Vector4', t: float) -> 'Vector4':
        omega = self.angle(other)
        sin_omega = math.sin(omega)
        return self * (math.sin((1 - t) * omega) / sin_omega) + other * (math.sin(t * omega) / sin_omega)

    def nlerp(self, other: 'Vector4', t: float) -> 'Vector4':
        return self.lerp(other, t).unit_vector()

    def nslerp(self, other: 'Vector4', t: float) -> 'Vector4':
        return self.slerp(other, t).unit_vector()

    def to_vector2(self) -> 'Vector2':
        return Vector2((self.x, self.y))

    def to_vector3(self) -> 'Vector3':
        return Vector3((self.x, self.y, self.z))

    def to_homogeneous(self) -> 'Vector4':
        return Vector4((self.x, self.y, self.z, 1))

    def to_list(self) -> list:
        return [self.x, self.y, self.z, self.w]

    def to_tuple(self) -> tuple:
        return tuple((self.x, self.y, self.z, self.w))

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}

    def to_json(self) -> str:
        return json.dumps({'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    def to_string(self) -> str:
        return f"Vector4({self.x}, {self.y}, {self.z}, {self.w})"

    @staticmethod
    def from_list(array: list) -> 'Vector4':
        return Vector4((array[0], array[1], array[2], array[3]))

    @staticmethod
    def from_tuple(array: tuple) -> 'Vector4':
        return Vector4((array[0], array[1], array[2], array[3]))

    @staticmethod
    def from_dict(array: dict) -> 'Vector4':
        return Vector4((array['x'], array['y'], array['z'], array['w']))

    @staticmethod
    def from_bytes(array: bytes) -> 'Vector4':
        return pickle.loads(array)

    @staticmethod
    def from_vector3(array: 'Vector3') -> 'Vector4':
        return Vector4((array.x, array.y, array.z, 0))

    @staticmethod
    def from_vector2(array: 'Vector2') -> 'Vector4':
        return Vector4((array.x, array.y, 0, 0))

    @staticmethod
    def from_vector2_homogenous(array: 'Vector2') -> 'Vector4':
        return Vector4((array.x, array.y, 1, 1))

    @staticmethod
    def from_vector3_homogenous(array: 'Vector3') -> 'Vector4':
        return Vector4((array.x, array.y, array.z, 1))
