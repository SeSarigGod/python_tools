import numpy as np


class SparseVector:
    def __init__(self, n: int, values: tuple[tuple[int, int | float], ...]):
        self.n: int = n
        self.values = {}
        for idx, val in values:
            if self.check_index(idx): raise IndexError(f"Index out of bounds: {idx}.")
            if val == 0: continue
            self.values[idx] = val
        self.dense_dot_product_lambda = self.construct_dense_dot_product_lambda()
        self.sparse_dot_product_lambda = self.construct_sparse_dot_product_lambda()

    def check_index(self, index: int) -> bool:
        return index < 1 or index > self.n

    def construct_dense_dot_product_lambda(self):
        func = lambda v: sum(v[idx] * x for idx, x in self.values.items())
        return func

    def construct_sparse_dot_product_lambda(self):
        func = lambda v: sum(v[idx] * x for idx, x in self.values.items())
        return func

    def dot_product(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector', left=True):
        if isinstance(other, SeededSparseTriangularMatrix):
            if left:
                return SparseVector(self.n, tuple((i, self.sparse_dot_product_lambda(other.get_column(i))) for i in range(1, self.n + 1)))
            else:
                return SparseVector(self.n, tuple((i, self.sparse_dot_product_lambda(other.get_row(i))) for i in range(1, self.n + 1)))
        elif isinstance(other, DenseVector):
            return self.dense_dot_product_lambda(other)
        elif isinstance(other, SparseVector):
            return self.sparse_dot_product_lambda(other)
        else:
            raise TypeError("Unsupported type for dot product.")

    def get(self, idx: int) -> int | float:
        return self.values.get(idx, 0)

    def norm(self, order: int = 2) -> int | float:
        if order == 0:
            return len(self.values)
        elif order == 1:
            return sum(abs(val) for val in self.values.values())
        elif order >= 2:
            return sum(val ** order for val in self.values.values()) ** (1 / order)
        else:
            raise ValueError("Unsupported norm order.")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, val + other) for idx, val in self.values.items()))
        elif isinstance(other, DenseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return DenseVector(tuple(self.get(idx) + other[idx] for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return SparseVector(self.n, tuple((idx, self.get(idx) + other[idx]) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for addition.")

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, other + val) for idx, val in self.values.items()))
        elif isinstance(other, DenseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return DenseVector(tuple(other[idx] + self.get(idx) for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return SparseVector(self.n, tuple((idx, other[idx] + self.get(idx)) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for addition.")

    def __getitem__(self, idx: int) -> int | float:
        return self.get(idx)

    def __mul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, val * other) for idx, val in self.values.items()))
        elif isinstance(other, (SeededSparseTriangularMatrix, DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {self.n} * {other.n}.")
            return self.dot_product(other)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __rmul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, val * other) for idx, val in self.values.items()))
        elif isinstance(other, (SeededSparseTriangularMatrix, DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {other.n} * {self.n}.")
            return self.dot_product(other, left=False)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __neg__(self):
        return SparseVector(self.n, tuple((idx, -val) for idx, val in self.values.items()))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, val - other) for idx, val in self.values.items()))
        elif isinstance(other, DenseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return DenseVector(tuple(self.get(idx) - other[idx] for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return SparseVector(self.n, tuple((idx, self.get(idx) - other[idx]) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for subtraction.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return SparseVector(self.n, tuple((idx, other - val) for idx, val in self.values.items()))
        elif isinstance(other, DenseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return DenseVector(tuple(other[idx] - self.get(idx) for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return SparseVector(self.n, tuple((idx, other[idx] - self.get(idx)) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for subtraction.")

    def __repr__(self):
        return f"SparseVector[{self.values}]"

class DenseVector:
    def __init__(self, values: tuple[int | float, ...]):
        self.values: tuple[int | float, ...] = values
        self.n: int = len(values)
        self.dense_dot_product_lambda = self.construct_dense_dot_product_lambda()
        self.sparse_dot_product_lambda = self.construct_sparse_dot_product_lambda()

    def check_index(self, idx: int) -> bool:
        return idx < 1 or idx > self.n

    def construct_dense_dot_product_lambda(self):
        func = lambda v: sum(self.get(idx) * v[idx] for idx in range(1, self.n + 1))
        return func

    def construct_sparse_dot_product_lambda(self):
        func = lambda v: sum(self.get(idx) * x for idx, x in v.values.items())
        return func

    def dot_product(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector', left=True):
        if isinstance(other, SeededSparseTriangularMatrix):
            if left:
                return DenseVector(tuple(self.sparse_dot_product_lambda(other.get_column(i)) for i in range(1, self.n + 1)))
            else:
                return DenseVector(tuple(self.sparse_dot_product_lambda(other.get_row(i)) for i in range(1, self.n + 1)))
        elif isinstance(other, DenseVector):
            return self.dense_dot_product_lambda(other)
        elif isinstance(other, SparseVector):
            return self.sparse_dot_product_lambda(other)
        else:
            raise TypeError("Unsupported type for dot product.")

    def get(self, idx: int) -> int | float:
        return self.values[idx - 1]

    def norm(self, order: int = 2) -> int | float:
        if order == 0:
            return len(self.values)
        elif order == 1:
            return sum(abs(val) for val in self.values)
        elif order >= 2:
            return sum(val ** order for val in self.values) ** (1 / order)
        else:
            raise ValueError("Unsupported norm order.")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return DenseVector(tuple(val + other for val in self.values))
        elif isinstance(other, (DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return DenseVector(tuple(self.get(idx) + other.get(idx) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for addition.")

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return DenseVector(tuple(other + val for val in self.values))
        elif isinstance(other, (DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for addition: {self.n} + {other.n}.")
            return DenseVector(tuple(other.get(idx) + self.get(idx) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for addition.")

    def __getitem__(self, idx: int) -> int | float:
        return self.values[idx - 1]

    def __mul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return DenseVector(tuple(x * other for x in self.values))
        elif isinstance(other, (SeededSparseTriangularMatrix, DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {self.n} * {other.n}.")
            return self.dot_product(other, left=True)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __rmul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return DenseVector(tuple(x * other for x in self.values))
        elif isinstance(other, (SeededSparseTriangularMatrix, DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {other.n} * {self.n}.")
            return self.dot_product(other, left=False)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __neg__(self):
        return DenseVector(tuple(-val for val in self.values))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return DenseVector(tuple(val - other for val in self.values))
        elif isinstance(other, DenseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return DenseVector(tuple(self.get(idx) - other[idx] for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return SparseVector(self.n, tuple((idx, self.get(idx) - other[idx]) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for subtraction.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return DenseVector(tuple(other - val for val in self.values))
        elif isinstance(other, (DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for subtraction: {self.n} - {other.n}.")
            return DenseVector(tuple(other[idx] - self.get(idx) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for subtraction.")

    def __repr__(self):
        return f"DenseVector{self.values}"


class SeededSparseTriangularMatrix:
    def __init__(self, seed: SparseVector, upper = False, lower = False):
        self.n: int = seed.n
        self._first_row = seed.values
        self._upper = upper
        self._lower = lower
        if self._upper and self._lower:
            self._upper = False
            self._lower = False

    def check_index(self, idx: int) -> bool:
        return idx < 1 or idx > self.n

    def dot_product(self, other: DenseVector | SparseVector, left=True):
        if isinstance(other, DenseVector):
            if left:
                return DenseVector(tuple(self.get_row(idx) * other for idx in range(1, self.n + 1)))
            else:
                return DenseVector(tuple(other * self.get_column(idx) for idx in range(1, self.n + 1)))
        elif isinstance(other, SparseVector):
            if left:
                return SparseVector(self.n, tuple((idx, self.get_row(idx) * other) for idx in range(1, self.n + 1)))
            else:
                return SparseVector(self.n, tuple((idx, other * self.get_column(idx)) for idx in range(1, self.n + 1)))
        else:
            raise TypeError("Unsupported type for dot product.")

    def get_column(self, idx: int) -> SparseVector:
        if self.check_index(idx): raise IndexError("Index out of bounds.")
        vector_tuple = None
        if self._upper:
            vector_tuple = tuple(((self.n - i + idx) % self.n + 1, self._first_row[i] if ((self.n - i + idx) % self.n + 1) <= idx else 0) for i in self._first_row)
        elif self._lower:
            vector_tuple = tuple(((self.n - i + idx) % self.n + 1, self._first_row[i] if ((self.n - i + idx) % self.n + 1) >= idx else 0) for i in self._first_row)
        else:
            vector_tuple = tuple(((self.n - i + idx) % self.n + 1, self._first_row[i]) for i in self._first_row)
        return SparseVector(self.n, vector_tuple)

    def get_row(self, idx: int) -> SparseVector:
        if self.check_index(idx): raise IndexError("Index out of bounds.")
        vector_tuple = None
        if self._upper:
            vector_tuple = tuple(((i + idx - 2) % self.n + 1, self._first_row[i] if ((i + idx - 2) % self.n + 1) >= idx else 0) for i in self._first_row)
        elif self._lower:
            vector_tuple = tuple(((i + idx - 2) % self.n + 1, self._first_row[i] if ((i + idx - 2) % self.n + 1) <= idx else 0) for i in self._first_row)
        else:
            vector_tuple = tuple(((i + idx - 2) % self.n + 1, self._first_row[i]) for i in self._first_row)
        return SparseVector(self.n, vector_tuple)

    def internal_product(self, other: 'SeededSparseTriangularMatrix', left=True) -> 'SeededSparseTriangularMatrix':
        if left:
            new_seed = SparseVector(self.n, tuple((idx, self.get_row(1) * other.get_column(idx)) for idx in range(1, self.n + 1) if abs(other.get_row(1) * self.get_column(idx)) > 1e-12))
        else:
            new_seed = SparseVector(self.n, tuple((idx, other.get_row(1) * self.get_column(idx)) for idx in range(1, self.n + 1) if abs(other.get_row(1) * self.get_column(idx)) > 1e-12))
        return SeededSparseTriangularMatrix(new_seed, self._upper or other._upper, self._lower or other._lower)

    def inv(self):
        matrix = []
        for i in range(1, self.n + 1):
            matrix.append([self.get_row(i)[j] for j in range(1, self.n + 1)])
        matrix = np.array(matrix)
        determinant = np.linalg.det(matrix)
        inverse_matrix = np.linalg.inv(matrix)
        inverse_determinant = 1 / determinant
        inv_seed = SparseVector(self.n, tuple((idx + 1, inverse_matrix[0][idx]) for idx in range(self.n) if abs(inverse_matrix[0][idx]) > abs(inverse_determinant)))
        return SeededSparseTriangularMatrix(inv_seed, self._upper, self._lower)

    def scalar_product(self, scalar: int | float) -> 'SeededSparseTriangularMatrix':
        return SeededSparseTriangularMatrix(SparseVector(self.n, tuple((idx, val * scalar) for idx, val in self._first_row.items())))

    def transpose(self) -> 'SeededSparseTriangularMatrix':
        return SeededSparseTriangularMatrix(self.get_column(1))

    def __mul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return self.scalar_product(other)
        elif isinstance(other, SeededSparseTriangularMatrix):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {self.n} * {other.n}.")
            return self.internal_product(other, True)
        elif isinstance(other, (DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {self.n} * {other.n}.")
            return self.dot_product(other, True)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __rmul__(self, other: 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float') -> 'SeededSparseTriangularMatrix | DenseVector | SparseVector | int | float':
        if isinstance(other, (int, float)):
            return self.scalar_product(other)
        elif isinstance(other, SeededSparseTriangularMatrix):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {other.n} * {self.n}.")
            return self.internal_product(other, False)
        elif isinstance(other, (DenseVector, SparseVector)):
            if other.n != self.n: raise ValueError(f"Dimension mismatch for multiplication: {other.n} * {self.n}.")
            return self.dot_product(other, False)
        else:
            raise TypeError("Unsupported type for multiplication.")

    def __repr__(self):
        return f"SSTM(first_row={self._first_row})"


class ConjugateGradientSolver:
    def __init__(self, H: SeededSparseTriangularMatrix, b: DenseVector, x0: DenseVector = None, tol: float = 1e-6, max_iter: int = 1000):
        self.H = H
        self.b = b
        self.x = [0] * (max_iter + 1)
        self.x[0] = x0 if x0 is not None else DenseVector(tuple(0 for _ in range(H.n)))
        self.tol = tol
        self.max_iter = max_iter
        self.k = 0
        self.g = [0] * (max_iter + 1)
        self.g[0] = self.H * self.x[0] - self.b
        self.p = [0] * (max_iter + 1)
        self.p[0] = -self.g[0]
        self.h = [0] * (max_iter + 1)
        self.alpha = [0] * (max_iter + 1)
        self.beta = [0] * (max_iter + 1)
        self.relative_errors = [0] * (max_iter + 1)

    def iterate(self):
        self.h[self.k] = self.H * self.p[self.k]
        self.alpha[self.k] = (self.g[self.k] * self.g[self.k]) / (self.p[self.k] * self.h[self.k])
        self.x[self.k + 1] = self.x[self.k] + self.alpha[self.k] * self.p[self.k]
        self.g[self.k + 1] = (self.g[self.k] + self.alpha[self.k] * self.h[self.k])
        self.beta[self.k] = (self.g[self.k + 1] * self.g[self.k + 1]) / (self.g[self.k] * self.g[self.k])
        self.p[self.k + 1] = -self.g[self.k + 1] + self.beta[self.k] * self.p[self.k]
        self.k += 1


class BroydenClass:
    def __init__(self, H, x0, B0, b, mu_k = 0, tol=1e-6, max_iter=1000):
        self.H = H
        self.x = [np.array(None)] * (max_iter + 1)
        self.x[0] = x0
        self.B = [np.array(None)] * (max_iter + 1)
        self.B[0] = B0
        self.b = b
        self.mu_k = mu_k
        self.tol = tol
        self.max_iter = max_iter
        self.g = [np.array(None)] * (max_iter + 1)
        self.g[0] = np.matvec(self.H, self.x[0]) - self.b
        self.k = 0
        self.p = [np.array(None)] * (max_iter + 1)
        self.h = [np.array(None)] * (max_iter + 1)
        self.alpha = [0] * (max_iter + 1)
        self.s = [np.array(None)] * (max_iter + 1)
        self.y = [np.array(None)] * (max_iter + 1)
        self.w = [np.array(None)] * (max_iter + 1)
        self.relative_errors = [0] * (max_iter + 1)

    def BFGS_update(self):
        return self.B[self.k] - (1 / np.linalg.multi_dot((self.s[self.k], self.B[self.k], self.s[self.k]))) * np.linalg.outer(np.matvec(self.B[self.k], self.s[self.k]), np.matvec(self.B[self.k], self.s[self.k])) + (1 / np.linalg.vecdot(self.y[self.k], self.s[self.k])) * np.linalg.outer(self.y[self.k], self.y[self.k])

    def iterate(self):
        self.p[self.k] = np.linalg.solve(self.B[self.k], -self.g[self.k])
        self.h[self.k] = np.matvec(self.H, self.p[self.k])
        self.alpha[self.k] = -np.linalg.vecdot(self.g[self.k], self.p[self.k]) / np.linalg.vecdot(self.h[self.k], self.p[self.k])
        self.s[self.k] = self.alpha[self.k] * self.p[self.k]
        self.y[self.k] = self.alpha[self.k] * self.h[self.k]
        self.x[self.k + 1] = self.x[self.k] + self.s[self.k]
        self.g[self.k + 1] = self.g[self.k] + self.y[self.k]
        self.B[self.k + 1] = self.BFGS_update()
        if self.mu_k > 0:
            self.w[self.k] = (1 / np.linalg.vecdot(self.y[self.k], self.s[self.k])) * self.y[self.k] - (1 / np.linalg.multi_dot((self.s[self.k], self.B[self.k], self.s[self.k]))) * np.matvec(self.B[self.k], self.s[self.k])
            self.B[self.k + 1] = self.B[self.k + 1] + self.mu_k * np.linalg.multi_dot((self.s[self.k], self.B[self.k], self.s[self.k])) * np.linalg.outer(self.w[self.k], self.w[self.k])
        self.k += 1


if __name__ == "__main__":
    size = 100
    print("Problem 2.22:")
    hessian = SeededSparseTriangularMatrix(seed=SparseVector(size, ((1, 4), (2, 1), (size, 1))))
    x_star = DenseVector(tuple(((-1) ** i) * i for i in range(1, size + 1)))
    b1 = hessian * x_star
    solver = ConjugateGradientSolver(H=hessian, b=b1, max_iter=20)
    relative_error = (x_star - solver.x[0]).norm(1) / x_star.norm(1)
    solver.relative_errors[0] = relative_error
    print(f"Iteration 0: Relative Error = {relative_error:.6e}")
    for _ in range(solver.max_iter):
        solver.iterate()
        relative_error = (x_star - solver.x[solver.k]).norm(1) / x_star.norm(1)
        solver.relative_errors[solver.k] = relative_error
        print(f"Iteration {solver.k}: Relative Error = {relative_error:.6e}")

    print("\n\nProblem 2.23:")
    a = (2 + (3**0.5))**0.5
    bb = 1 / a
    S = SeededSparseTriangularMatrix(seed=SparseVector(size, ((1, a), (2, bb), (size, bb))), upper=True)
    S_t = S.transpose()
    S_inv = S.inv()
    S_inv_t = S_inv.transpose()
    M = S_inv_t * hessian * S_inv
    c = S_inv_t * b1
    solver2 = ConjugateGradientSolver(H=M, b=c, max_iter=20)
    relative_error2 = (x_star - (S_inv * solver2.x[0])).norm(1) / x_star.norm(1)
    solver2.relative_errors[0] = relative_error2
    print(f"Iteration 0: Relative Error = {relative_error2:.6e}")
    for _ in range(solver2.max_iter):
        solver2.iterate()
        relative_error2 = (x_star - (S_inv * solver2.x[solver2.k])).norm(1) / x_star.norm(1)
        solver2.relative_errors[solver2.k] = relative_error2
        print(f"Iteration {solver2.k}: Relative Error = {relative_error2:.6e}")

    print("Problem 2.23 Comment: The preconditioned conjugate gradient method converges much faster than the standard conjugate gradient method, however it also seems to have converged to a very different solution.")

    f = lambda x1, x2: x1 - 0.75 * x2 + (4 / 9) * (x1 ** 2) - 2 * x1 * x2 + 3 * (x2 ** 2)
    df = lambda x: np.array([1 + (8 / 9) * x[0] - 2 * x[1], -0.75 - 2 * x[0] + 6 * x[1]])

    Hess = np.array([[8 / 9, -2], [-2, 6]])
    initial_guess = np.array([-1, 4])
    initial_matrix = np.array([[2, 1], [1, 3]])
    initial_b = np.array([-1, 0.75])

"""
Output 23:35 23/03/26
Problem 2.22:
Iteration 0: Relative Error = 1.000000e+00
Iteration 1: Relative Error = 7.600729e-02
Iteration 2: Relative Error = 1.013036e-02
Iteration 3: Relative Error = 2.640674e-03
Iteration 4: Relative Error = 7.191056e-04
Iteration 5: Relative Error = 1.937730e-04
Iteration 6: Relative Error = 5.181010e-05
Iteration 7: Relative Error = 1.383818e-05
Iteration 8: Relative Error = 3.694611e-06
Iteration 9: Relative Error = 9.860684e-07
Iteration 10: Relative Error = 2.630837e-07
Iteration 11: Relative Error = 7.016863e-08
Iteration 12: Relative Error = 1.871018e-08
Iteration 13: Relative Error = 4.987248e-09
Iteration 14: Relative Error = 1.328896e-09
Iteration 15: Relative Error = 3.539706e-10
Iteration 16: Relative Error = 9.425165e-11
Iteration 17: Relative Error = 2.508744e-11
Iteration 18: Relative Error = 6.675277e-12
Iteration 19: Relative Error = 1.775531e-12
Iteration 20: Relative Error = 4.721022e-13


Problem 2.23:
Iteration 0: Relative Error = 1.000000e+00
Iteration 1: Relative Error = 3.110062e-01
Iteration 2: Relative Error = 3.115159e-01
Iteration 3: Relative Error = 3.115301e-01
Iteration 4: Relative Error = 3.115300e-01
Iteration 5: Relative Error = 3.115300e-01
Iteration 6: Relative Error = 3.115300e-01
Iteration 7: Relative Error = 3.115300e-01
Iteration 8: Relative Error = 3.115300e-01
Iteration 9: Relative Error = 3.115300e-01
Iteration 10: Relative Error = 3.115300e-01
Iteration 11: Relative Error = 3.115300e-01
Iteration 12: Relative Error = 3.115300e-01
Iteration 13: Relative Error = 3.115300e-01
Iteration 14: Relative Error = 3.115300e-01
Iteration 15: Relative Error = 3.115300e-01
Iteration 16: Relative Error = 3.115300e-01
Iteration 17: Relative Error = 3.115300e-01
Iteration 18: Relative Error = 3.115300e-01
Iteration 19: Relative Error = 3.115300e-01
Iteration 20: Relative Error = 3.115300e-01
Problem 2.23 Comment: The preconditioned conjugate gradient method converges much faster than the standard conjugate gradient method, however it also seems to have converged to a very different solution.

Process finished with exit code 0
"""
