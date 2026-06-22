class DenseVector:
    def __init__(self, values: tuple[int | float, ...]):
        self.n: int = len(values)
        self.values = values

    def get(self, index: int):
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        return self.values[index - 1]

    def set(self, index: int, value: int | float):
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        self.values = self.values[:index - 1] + (value,) + self.values[index:]

    def __str__(self):
        return f"DenseVector(size={self.n}, values={self.values})"


class SparseVector:
    def __init__(self, size: int, values: tuple[tuple[int, int | float], ...]):
        self.n: int = size
        self.values = {}
        for index, value in values:
            if index < 1 or index > self.n:
                raise IndexError("Index out of bounds")
            if value == 0:
                continue
            self.values[index] = value

    def fill(self, indices: tuple[int, ...], values: tuple[int | float, ...]):
        if len(indices) != len(values):
            raise ValueError("Indices and values must have the same length")
        for index, value in zip(indices, values):
            self.set(index, value)

    def get(self, index: int):
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        return self.values.get(index, 0)

    def set(self, index: int, value: int | float):
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        if value != 0:
            self.values[index] = value
        elif index in self.values:
            del self.values[index]

    def to_dense(self):
        dense_vector = DenseVector(tuple(0 for _ in range(self.n)))
        for index, value in self.values.items():
            dense_vector.set(index, value)
        return dense_vector

    def __str__(self):
        return f"SparseVector(size={self.n}, values={self.values})"


class SparseSeededTriangularMatrix:
    def __init__(self, size: int, seed: tuple[tuple[int, int | float], ...]):
        self.n: int = size
        self.seed: tuple[tuple[int, int | float], ...] = seed

        self.left_dense_lambda = self.construct_left_dense_lambda()
        self.left_sparse_lambda = self.construct_left_sparse_lambda()
        self.right_dense_lambda = self.construct_right_dense_lambda()
        self.right_sparse_lambda = self.construct_right_sparse_lambda()

    def construct_left_dense_lambda(self):
        func = lambda row_idx, v: sum(x * v.values[(i + row_idx) % self.n] for i, x in self.seed)
        return func

    def construct_left_sparse_lambda(self):
        func = lambda row_idx, v: sum(x * v.get((i + row_idx + 1) % (self.n + 1)) for i, x in self.seed)
        return func

    def construct_right_dense_lambda(self):
        func = lambda row_idx, v: sum(x * v.values[(i + row_idx) % self.n] for i, x in self.seed)
        return func

    def construct_right_sparse_lambda(self):
        func = lambda row_idx, v: sum(x * v.get((i + row_idx + 1) % (self.n + 1)) for i, x in self.seed)
        return func

    def dot_product(self, other, left=True):
        if self.n != other.n:
            raise ValueError("Vector and Matrix sizes do not match")
        if left:
            if isinstance(other, DenseVector):
                return tuple(self.left_dense_lambda(row_idx, other) for row_idx in range(self.n))
            elif isinstance(other, SparseVector):
                return tuple(self.left_sparse_lambda(row_idx, other) for row_idx in range(self.n))
            else:
                raise TypeError("Unsupported vector type")
        else:
            if isinstance(other, DenseVector):
                return DenseVector(self.right_dense_lambda(row_idx, other) for row_idx in range(self.n))
            elif isinstance(other, SparseVector):
                return SparseVector(self.n, tuple((row_idx + 1, self.right_sparse_lambda(row_idx, other) for row_idx in range(self.n))))
            else:
                raise TypeError("Unsupported vector type")

    def __mul__(self, other):
        if isinstance(other, (DenseVector, SparseVector)):
            return self.dot_product(other, left=True)
        elif isinstance(other, (int, float)):
            new_seed = tuple((i, x * other) for i, x in self.seed)
            return SparseSeededTriangularMatrix(self.n, new_seed)
        elif isinstance(other, SparseSeededTriangularMatrix):
            new_seed = tuple((i, x * y) for (i, x), (j, y) in zip(self.seed, other.seed) if i == j)
            return SparseSeededTriangularMatrix(self.n, new_seed)
        else:
            raise TypeError("Unsupported multiplication type")

    def __str__(self):
        return f"SparseTriangularMatrix(size={self.n}, seed={self.seed})"