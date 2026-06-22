import numpy as np
from particle_properties import *


class WaveFunction:
    def __init__(self):
        self.coefficients = []
        self.particle_states = []
        self.wave_function = []
        self.normalization = 1

    def add_component(self, coefficient: int | float, particle_state: tuple):
        if len(self.particle_states) != 0:
            if len(particle_state) != len(self.particle_states[-1]):
                raise ValueError("All particle states must have the same number of particles.")

        if particle_state in self.particle_states:
            idx = self.particle_states.index(particle_state)
            self.coefficients[idx] += coefficient
            if self.coefficients[idx] == 0:
                del self.coefficients[idx]
                del self.particle_states[idx]
                del self.wave_function[idx]
            else:
                self.wave_function[idx] = (self.coefficients[idx], self.particle_states[idx])
        else:
            self.coefficients.append(coefficient)
            self.particle_states.append(particle_state)
            self.wave_function.append((coefficient, particle_state))

    def update_normalization(self):
        norm = sum(np.power(coef, 2) for coef in self.coefficients)
        if norm > 0:
            self.normalization = np.sqrt(norm)
        else:
            raise ValueError(f"Normalization must be positive, instead has value: {norm}.")

    def __mul__(self, other):
        if isinstance(other, WaveFunction):
            new_wave_function = WaveFunction()
            for coef1, state1 in self.wave_function:
                for coef2, state2 in other.wave_function:
                    new_coef = (coef1 / self.normalization) * (coef2 / other.normalization)
                    new_state = (state1, state2)
                    new_wave_function.add_component(new_coef, new_state)
            return new_wave_function
        elif isinstance(other, (int, float)):
            new_wave_function = WaveFunction()
            for coef, state in self.wave_function:
                new_wave_function.add_component(coef * other, state)
            return new_wave_function
        else:
            raise ValueError("Multiplication is only supported between WaveFunction instances or with scalars.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other != 0:
                new_wave_function = WaveFunction()
                for coef, state in self.wave_function:
                    new_wave_function.add_component(coef / other, state)
                return new_wave_function
            else:
                raise ValueError("Division by zero is not allowed.")
        else:
            raise ValueError("Division is only supported with scalars.")

    def __rtruediv__(self, other):
        raise ValueError("Right division is not supported for WaveFunction instances.")

    def __add__(self, other):
        if isinstance(other, WaveFunction):
            new_wave_function = WaveFunction()
            for coef, state in self.wave_function:
                new_wave_function.add_component(coef / self.normalization, state)
            for coef, state in other.wave_function:
                new_wave_function.add_component(coef / other.normalization, state)
            return new_wave_function
        else:
            raise ValueError("Addition is only supported between WaveFunction instances.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, WaveFunction):
            for coef, state in other.wave_function:
                self.add_component((coef / other.normalization) * self.normalization, state)
            return self
        else:
            raise ValueError("In-place addition is only supported between WaveFunction instances.")

    def __repr__(self):
        return f"WaveFunction(Normalization={self.normalization}, States={self.wave_function})"


def construct_triple_antisymmetrical_state(basis: tuple):
    """Construct the completely antisymmetrical state for a given set of three."""
    similarity = determine_similarity(basis)

    wave_function = WaveFunction()
    if similarity == 0:
        for i in range(3):
            wave_function.add_component(1, (basis[i], basis[(i + 1) % 3], basis[(i + 2) % 3]))
            wave_function.add_component(-1, (basis[i], basis[(i + 2) % 3], basis[(i + 1) % 3]))
    else:
        raise ValueError("Completely antisymmetrical states can only be constructed for three distinct items.")

    wave_function.update_normalization()
    return wave_function


def construct_mixed_state(basis: tuple, asymmetry: int, triplet: bool = False):
    """Construct the mixed symmetry state for a given set of three."""
    similarity = determine_similarity(basis)

    basis_zero = None
    basis_up = None
    basis_down = None
    idx = None
    idx_up = None
    idx_down = None
    match similarity:
        case 0:
            triplet_idx = basis.index('s') if 's' in basis else basis.index('c') if 'c' in basis else basis.index(
                'b') if 'b' in basis else basis.index('t') if 't' in basis else None
            if triplet_idx is None:
                raise ValueError("Triplet states can only be constructed for three distinct items.")
            basis_zero = basis[triplet_idx]
            basis_up = basis[(triplet_idx + 1) % 3]
            basis_down = basis[(triplet_idx + 2) % 3]
        case 1:
            idx = 2
            idx_up = (idx + 1) % 3
            idx_down = (idx + 2) % 3
        case 2:
            idx = 1
            idx_up = (idx + 1) % 3
            idx_down = (idx + 2) % 3
        case 3:
            idx = 0
            idx_up = (idx + 1) % 3
            idx_down = (idx + 2) % 3
        case 4:
            raise ValueError("Mixed symmetry states cannot be constructed for three identical items.")

    wave_function = WaveFunction()
    if asymmetry == 0:
        if similarity == 0:
            if triplet:
                wave_function.add_component(1, (basis_up, basis_zero, basis_down))
                wave_function.add_component(-1, (basis_zero, basis_up, basis_down))
                wave_function.add_component(1, (basis_down, basis_zero, basis_up))
                wave_function.add_component(-1, (basis_zero, basis_down, basis_up))
            elif not triplet:
                wave_function.add_component(2, (basis_up, basis_down, basis_zero))
                wave_function.add_component(-2, (basis_down, basis_up, basis_zero))
                wave_function.add_component(1, (basis_up, basis_zero, basis_down))
                wave_function.add_component(-1, (basis_zero, basis_up, basis_down))
                wave_function.add_component(-1, (basis_down, basis_zero, basis_up))
                wave_function.add_component(1, (basis_zero, basis_down, basis_up))
            else:
                raise ValueError(f"Something is wrong with the value of triplet (bool). The value is {triplet}.")
        else:
            wave_function.add_component(1, (basis[idx_down], basis[idx], basis[idx_up]))
            wave_function.add_component(-1, (basis[idx], basis[idx_down], basis[idx_up]))
    elif asymmetry == 1:
        if similarity == 0:
            if triplet:
                wave_function.add_component(1, (basis_up, basis_down, basis_zero))
                wave_function.add_component(-1, (basis_zero, basis_down, basis_up))
                wave_function.add_component(1, (basis_down, basis_up, basis_zero))
                wave_function.add_component(-1, (basis_zero, basis_up, basis_down))
            elif not triplet:
                wave_function.add_component(2, (basis_up, basis_zero, basis_down))
                wave_function.add_component(-2, (basis_down, basis_zero, basis_up))
                wave_function.add_component(1, (basis_up, basis_down, basis_zero))
                wave_function.add_component(-1, (basis_zero, basis_down, basis_up))
                wave_function.add_component(-1, (basis_down, basis_up, basis_zero))
                wave_function.add_component(1, (basis_zero, basis_up, basis_down))
            else:
                raise ValueError(f"Something is wrong with the value of triplet (bool). The value is {triplet}.")
        else:
            wave_function.add_component(1, (basis[idx_down], basis[idx_up], basis[idx]))
            wave_function.add_component(-1, (basis[idx], basis[idx_up], basis[idx_down]))
    elif asymmetry == 2:
        if similarity == 0:
            if triplet:
                wave_function.add_component(1, (basis_down, basis_up, basis_zero))
                wave_function.add_component(-1, (basis_down, basis_zero, basis_up))
                wave_function.add_component(1, (basis_up, basis_down, basis_zero))
                wave_function.add_component(-1, (basis_up, basis_zero, basis_down))
            elif not triplet:
                wave_function.add_component(2, (basis_zero, basis_up, basis_down))
                wave_function.add_component(-2, (basis_zero, basis_down, basis_up))
                wave_function.add_component(1, (basis_down, basis_up, basis_zero))
                wave_function.add_component(-1, (basis_down, basis_zero, basis_up))
                wave_function.add_component(-1, (basis_up, basis_down, basis_zero))
                wave_function.add_component(1, (basis_up, basis_zero, basis_down))
            else:
                raise ValueError(f"Something is wrong with the value of triplet (bool). The value is {triplet}.")
        else:
            wave_function.add_component(1, (basis[idx_down], basis[idx_up], basis[idx]))
            wave_function.add_component(-1, (basis[idx_down], basis[idx], basis[idx_up]))
    else:
        raise ValueError(f"Asymmetry can only be 0, 1, or 2; not {asymmetry}.")

    wave_function.update_normalization()
    return wave_function


def construct_triple_symmetrical_state(basis: tuple):
    """Construct the completely symmetrical state for a given set of three."""
    similarity = determine_similarity(basis)

    wave_function = WaveFunction()
    if similarity == 4:
        wave_function.add_component(1, basis)
    elif similarity in [1, 2, 3]:
        for i in range(3):
            wave_function.add_component(1, (basis[i], basis[(i + 1) % 3], basis[(i + 2) % 3]))
    elif similarity == 0:
        for i in range(6):
            wave_function.add_component(1, (basis[i // 2], basis[(i // 2 + 1) % 3], basis[(i // 2 + 2) % 3]))

    wave_function.update_normalization()
    return wave_function


class Baryon:
    def __init__(self, flavors: tuple, baryon_isospin: float, baryon_spin: tuple[float, float]):
        self.flavors = flavors
        self.flavor_similarity = determine_similarity(flavors)
        self.isospin = baryon_isospin
        self.spin = baryon_spin[0]
        self.spin_value = baryon_spin[1]
        self.spins = self.determine_spins()
        self.strangeness = -flavors.count('s')
        self.charmness = flavors.count('c')
        self.bottomness = -flavors.count('b')
        self.topness = flavors.count('t')
        self.baryon_type = self.determine_baryon_type()

        self.wave_function_flavor_components = []
        self.wave_function_spin_components = []
        self.wave_function = None

        self.construct_wave_function()

    def construct_wave_function_spin_components(self):
        match self.spin:
            case 0.5:
                for i in range(3):
                    self.wave_function_spin_components.append(construct_mixed_state(self.spins, asymmetry=i))
            case 1.5:
                if self.baryon_type == "lambda":
                    self.wave_function_spin_components.append(construct_triple_antisymmetrical_state(self.spins))
                else:
                    self.wave_function_spin_components.append(construct_triple_symmetrical_state(self.spins))
            case _:
                raise ValueError(f"Invalid spin value (for now) for baryon: {self.spin}.")

    def construct_wave_function_flavor_components(self):
        match self.baryon_type:
            case "nucleon":
                for i in range(3):
                    self.wave_function_flavor_components.append(construct_mixed_state(self.flavors, asymmetry=i))
            case "delta":
                self.wave_function_flavor_components.append(construct_triple_symmetrical_state(self.flavors))
            case "lambda":
                if self.spin == 0.5:
                    for i in range(3):
                        self.wave_function_flavor_components.append(construct_mixed_state(self.flavors, asymmetry=i, triplet=False))
                else:
                    self.wave_function_flavor_components.append(construct_triple_antisymmetrical_state(self.flavors))
            case "sigma":
                if self.spin == 0.5:
                    for i in range(3):
                        self.wave_function_flavor_components.append(construct_mixed_state(self.flavors, asymmetry=i, triplet=True))
                else:
                    self.wave_function_flavor_components.append(construct_triple_symmetrical_state(self.flavors))
            case "xi":
                for i in range(3):
                    self.wave_function_flavor_components.append(construct_mixed_state(self.flavors, asymmetry=i))
            case "omega":
                self.wave_function_flavor_components.append(construct_triple_symmetrical_state(self.flavors))
            case _:
                raise ValueError(f"Invalid baryon type: {self.baryon_type}.")

    def construct_wave_function(self):
        self.construct_wave_function_flavor_components()
        self.construct_wave_function_spin_components()

        combined_wave_function = WaveFunction()
        for i, j in enumerate(self.wave_function_flavor_components):
            combined_wave_function += j * self.wave_function_spin_components[i]

        self.wave_function = combined_wave_function
        self.wave_function.update_normalization()

    def determine_charge(self):
        charge = 0
        for flavor in self.flavors:
            match flavor:
                case 'u': charge += 2
                case 'd': charge -= 1
                case 'c': charge += 2
                case 's': charge -= 1
                case 't': charge += 2
                case 'b': charge -= 1
                case _: raise ValueError(f"Invalid flavor: {flavor}.")
        return charge / 3

    def determine_baryon_type(self):
        if self.strangeness == 0:
            if self.isospin == 0.5:
                return "nucleon"
            elif self.isospin == 1.5:
                return "delta"
            else:
                raise ValueError(f"Invalid isospin value for strangeness 0 baryon: {self.isospin}.")
        elif self.strangeness == -1:
            if self.isospin == 0.0:
                return "lambda"
            elif self.isospin == 1.0:
                return "sigma"
            else:
                raise ValueError(f"Invalid isospin value for strangeness -1 baryon: {self.isospin}.")
        elif self.strangeness == -2:
            if self.isospin == 0.5:
                return "xi"
            else:
                raise ValueError(f"Invalid isospin value for strangeness -2 baryon: {self.isospin}.")
        elif self.strangeness == -3:
            if self.isospin == 0.0:
                return "omega"
            else:
                raise ValueError(f"Invalid isospin value for strangeness -3 baryon: {self.isospin}.")
        else:
            raise ValueError(f"Invalid strangeness value for baryon: {self.strangeness}.")

    def determine_spins(self):
        match self.spin_value:
            case -1.5: return -0.5, -0.5, -0.5
            case -0.5: return 0.5, -0.5, -0.5
            case 0.5: return 0.5, 0.5, -0.5
            case 1.5: return 0.5, 0.5, 0.5
            case _:
                raise ValueError(f"Invalid spin value (for now) for baryon: {self.spin}.")
