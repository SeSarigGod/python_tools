class DenseVector:
    """A simple dense vector backed by an immutable tuple of numeric values.

    Notes:
    - Public API uses 1-based indexing (get/set) to match the rest of the codebase.
    - Internally, the tuple is zero-indexed.
    """

    def __init__(self, values: tuple[int | float, ...]):
        self.n: int = len(values)
        self.values: tuple[int | float, ...] = tuple(values)

    def get(self, index: int) -> int | float:
        """Return the value at 1-based index.

        Raises IndexError if index out of range.
        """
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        return self.values[index - 1]

    def set(self, index: int, value: int | float) -> None:
        """Replace the value at 1-based index. Keeps the underlying representation as a tuple.

        This creates a new tuple with the replaced element.
        """
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        # Build a new tuple that replaces the element at index-1
        self.values = self.values[: index - 1] + (value,) + self.values[index:]

    def __str__(self) -> str:
        return f"DenseVector(size={self.n}, values={self.values})"


class SparseVector:
    """A sparse vector implemented as a mapping from 1-based index -> value.

    Only non-zero values are stored in the internal dict.
    """

    def __init__(self, size: int, values: tuple[tuple[int, int | float], ...]):
        self.n: int = int(size)
        self.values: dict[int, int | float] = {}
        for index, value in values:
            if index < 1 or index > self.n:
                raise IndexError("Index out of bounds")
            if value == 0:
                continue
            self.values[int(index)] = value

    def fill(self, indices: tuple[int, ...], values: tuple[int | float, ...]) -> None:
        if len(indices) != len(values):
            raise ValueError("Indices and values must have the same length")
        for index, value in zip(indices, values):
            self.set(index, value)

    def get(self, index: int) -> int | float:
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        return self.values.get(index, 0)

    def set(self, index: int, value: int | float) -> None:
        if index < 1 or index > self.n:
            raise IndexError("Index out of bounds")
        if value != 0:
            self.values[index] = value
        elif index in self.values:
            del self.values[index]

    def to_dense(self) -> DenseVector:
        dense_vector = DenseVector(tuple(0 for _ in range(self.n)))
        for index, value in self.values.items():
            dense_vector.set(index, value)
        return dense_vector

    def __str__(self) -> str:
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
        """Construct a function that computes a single entry of M * v when v is dense.

        The returned function takes a 0-based row_idx and a DenseVector
        and returns the numeric dot product for that row.
        """
        func = lambda row_idx, v: sum(
            x * v.values[(i + row_idx) % self.n] for i, x in self.seed
        )
        return func

    def construct_left_sparse_lambda(self):
        """Construct a function that computes a single entry of M * v when v is sparse.

        This converts the internal 0-based computation into the SparseVector's
        1-based API when calling get(...).
        """
        func = lambda row_idx, v: sum(
            x * v.get(((i + row_idx) % self.n) + 1) for i, x in self.seed
        )
        return func

    def construct_right_dense_lambda(self):
        """Same as left_dense but kept separate for clarity/extensibility."""
        func = lambda row_idx, v: sum(
            x * v.values[(i + row_idx) % self.n] for i, x in self.seed
        )
        return func

    def construct_right_sparse_lambda(self):
        func = lambda row_idx, v: sum(
            x * v.get(((i + row_idx) % self.n) + 1) for i, x in self.seed
        )
        return func

    def dot_product(self, other, left=True):
        if self.n != other.n:
            raise ValueError("Vector and Matrix sizes do not match")
        if left:
            if isinstance(other, DenseVector):
                # Return a DenseVector result for matrix * dense_vector
                return DenseVector(tuple(self.left_dense_lambda(row_idx, other) for row_idx in range(self.n)))
            elif isinstance(other, SparseVector):
                # Build a sparse result containing only non-zero entries
                entries = tuple(
                    (row_idx + 1, val)
                    for row_idx in range(self.n)
                    if (val := self.left_sparse_lambda(row_idx, other)) != 0
                )
                return SparseVector(self.n, entries)
            else:
                raise TypeError("Unsupported vector type")
        else:
            if isinstance(other, DenseVector):
                return DenseVector(tuple(self.right_dense_lambda(row_idx, other) for row_idx in range(self.n)))
            elif isinstance(other, SparseVector):
                entries = tuple(
                    (row_idx + 1, val)
                    for row_idx in range(self.n)
                    if (val := self.right_sparse_lambda(row_idx, other)) != 0
                )
                return SparseVector(self.n, entries)
            else:
                raise TypeError("Unsupported vector type")

    def __mul__(self, other):
        if isinstance(other, (DenseVector, SparseVector)):
            return self.dot_product(other, left=True)
        elif isinstance(other, (int, float)):
            new_seed = tuple((i, x * other) for i, x in self.seed)
            return SparseSeededTriangularMatrix(self.n, new_seed)
        elif isinstance(other, SparseSeededTriangularMatrix):
            # Multiply seeds element-wise by matching seed indices.
            a = dict(self.seed)
            b = dict(other.seed)
            common = sorted(set(a.keys()) & set(b.keys()))
            new_seed = tuple((i, a[i] * b[i]) for i in common)
            return SparseSeededTriangularMatrix(self.n, new_seed)
        else:
            raise TypeError("Unsupported multiplication type")

    def __str__(self):
        return f"SparseTriangularMatrix(size={self.n}, seed={self.seed})"