import numpy as np

quark_charges = {
    'u': 2/3,  # Up quark charge
    'd': -1/3, # Down quark charge
    'c': 2/3,  # Charm quark charge
    's': -1/3, # Strange quark charge
    't': 2/3,  # Top quark charge
    'b': -1/3, # Bottom quark charge
}

quark_spins = {
    'u': 0.5,  # Up quark spin
    'd': 0.5,  # Down quark spin
    'c': 0.5,  # Charm quark spin
    's': 0.5,  # Strange quark spin
    't': 0.5,  # Top quark spin
    'b': 0.5,  # Bottom quark spin
}

quark_isospins = {
    'u': 0.5,  # Up quark isospin
    'd': -0.5, # Down quark isospin
    'c': 0.0,  # Charm quark isospin
    's': 0.0,  # Strange quark isospin
    't': 0.0,  # Top quark isospin
    'b': 0.0,  # Bottom quark isospin
}

baryon_types = {
    'N': {'I': 0.5, 'S':  0, 'Flavor Symmetry': 'M'},
    'D': {'I': 1.5, 'S':  0, 'Flavor Symmetry': 'S'},
    'L': {'I': 0.0, 'S': -1, 'Flavor Symmetry': 'A'},
    'S': {'I': 1.0, 'S': -1, 'Flavor Symmetry': 'S'},
    'X': {'I': 0.5, 'S': -2, 'Flavor Symmetry': 'M'},
    'O': {'I': 0.0, 'S': -3, 'Flavor Symmetry': 'S'},
}

# proton_wave_function = [
#     (1, ('u', 'u', 'd'), (0.5, 0.5, -0.5)), (-1, ('u', 'u', 'd'), (0.5, -0.5, 0.5)), (-1, ('u', 'u', 'd'), (-0.5, 0.5, 0.5)),
#     (1, ('u', 'u', 'd'), (0.5, 0.5, -0.5)), (-1, ('u', 'u', 'd'), (0.5, -0.5, 0.5)), (-1, ('u', 'u', 'd'), (0.5, -0.5, 0.5)),
#     (1, ('u', 'd', 'u'), (0.5, -0.5, 0.5)), (-1, ('u', 'd', 'u'), (-0.5, 0.5, 0.5)), (-1, ('u', 'd', 'u'), (0.5, 0.5, -0.5)),
#     (1, ('u', 'd', 'u'), (0.5, -0.5, 0.5)), (-1, ('u', 'd', 'u'), (-0.5, 0.5, 0.5)), (-1, ('u', 'd', 'u'), (0.5, 0.5, -0.5)),
#     (1, ('d', 'u', 'u'), (-0.5, 0.5, 0.5)), (-1, ('d', 'u', 'u'), (0.5, 0.5, -0.5)), (-1, ('d', 'u', 'u'), (0.5, -0.5, 0.5)),
#     (1, ('d', 'u', 'u'), (-0.5, 0.5, 0.5)), (-1, ('d', 'u', 'u'), (0.5, 0.5, -0.5)), (-1, ('d', 'u', 'u'), (0.5, -0.5, 0.5))
# ]


def determine_similarity_generalization(basis: tuple):
    """Calculate the equality for a given set of three items."""
    similarity = []
    processed = []

    for i, j in enumerate(basis):
        if j in processed:
            continue
        processed.append(j)

        indices = tuple(k for k, l in enumerate(basis) if l == j and k != i)
        similarity.append((i, indices))

    return similarity


def determine_similarity(basis: tuple):
    """Calculate the equality for a given set of three items."""
    if basis[0] == basis[1] == basis[2]:
        return 4
    elif basis[0] == basis[1]:
        return 1
    elif basis[0] == basis[2]:
        return 2
    elif basis[1] == basis[2]:
        return 3
    else:
        return 0


def spin_dot_spin(total_spin, spin1, spin2):
    """Calculate the spin dot product given the total spin."""
    return 0.5 * (total_spin * (total_spin + 1) - spin1 * (spin1 + 1) - spin2 * (spin2 + 1))


# def create_baryon_wave_function(self):
#     if len(self.flavors) != 3 or len(self.spins) != 3:
#         raise ValueError("Baryons must have 3 flavors and 3 spins.")
#     # Generate all possible combinations of flavors and spins for the baryon wave function
#     # Wavefunction is of the form |flavor1, spin1; flavor2, spin2; flavor3, spin3>, coded as (flavor, spin) tuples
#     # e.g. +|u, up; u, up; d, down> would be represented as (1, (u, u, d), (up, up, down))
#     if self.flavors[0] == self.flavors[1] == self.flavors[2]:
#         self.flavor_equality = 4
#     elif self.flavors[0] == self.flavors[1]:
#         self.flavor_equality = 1
#     elif self.flavors[0] == self.flavors[2]:
#         self.flavor_equality = 2
#     elif self.flavors[1] == self.flavors[2]:
#         self.flavor_equality = 3
#     else:
#         self.flavor_equality = 0
#     if self.spins[0] == self.spins[1] == self.spins[2]:
#         self.spin_equality = 4
#     elif self.spins[0] == self.spins[1]:
#         self.spin_equality = 1
#     elif self.spins[0] == self.spins[2]:
#         self.spin_equality = 2
#     elif self.spins[1] == self.spins[2]:
#         self.spin_equality = 3
#     else:
#         self.spin_equality = 0
#     combinations = []
#     baryon_wave_function = None
#     if self.spin_equality < 4 and 4 > self.flavor_equality > 0:
#         for i in range(3):
#             for j in range(3):
#                 flavors1 = (self.flavors[-j], self.flavors[1-j], self.flavors[2-j])
#                 spins1 = (self.spins[-i], self.spins[1-i], self.spins[2-i])
#                 flavors2 = None
#                 if self.flavor_equality == 1:
#                     flavors2 = (flavors1[1], flavors1[0], flavors1[2])
#                 elif self.flavor_equality == 2:
#                     flavors2 = (flavors1[2], flavors1[1], flavors1[0])
#                 elif self.flavor_equality == 3:
#                     flavors2 = (flavors1[0], flavors1[2], flavors1[1])
#                 spins2 = None
#                 if self.spin_equality == 1:
#                     spins2 = (spins1[1], spins1[0], spins1[2])
#                 elif self.spin_equality == 2:
#                     spins2 = (spins1[2], spins1[1], spins1[0])
#                 elif self.spin_equality == 3:
#                     spins2 = (spins1[0], spins1[2], spins1[1])
#                 combination1 = (1 if i == j else -1, flavors1, spins1)
#                 combination2 = (1 if i == j else -1, flavors2, spins2)
#                 combinations.append(combination1)
#                 combinations.append(combination2)
#         baryon_wave_function = combinations
#     elif self.spin_equality == 4 and self.flavor_equality == 4:
#         for j in range(3):
#             flavors1 = (self.flavors[-j], self.flavors[1-j], self.flavors[2-j])
#             flavors2 = (self.flavors[1-j], self.flavors[-j], self.flavors[2-j])
#             spins = (self.spins[0], self.spins[1], self.spins[2])
#             combination1 = (1, flavors1, spins)
#             combination2 = (1, flavors2, spins)
#             combinations.append(combination1)
#             combinations.append(combination2)
#         baryon_wave_function = combinations
#     self.normalization = 1 / np.sqrt(len(baryon_wave_function))
#     return baryon_wave_function