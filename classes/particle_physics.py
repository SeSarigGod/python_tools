import numpy as np
from wave_function import *
from particle_properties import *


def calculate_meson_mass(tspin, q1, q2):
    """Calculate the mass of a meson given its spin and the masses of its constituent quark and antiquark."""
    # Masses for quarks (in MeV/c^2)
    quark_masses = {
        'u': 308,  # Up quark mass
        'd': 308,  # Down quark mass
        's': 483,  # Strange quark mass
    }

    m1 = quark_masses[q1]
    m2 = quark_masses[q2]

    A = np.power(2 * quark_masses['u'], 2) * 160  # A constant for spin-spin interaction

    s1dots2 = spin_dot_spin(tspin, 0.5, 0.5)
    mass = m1 + m2 + (A * s1dots2) / (m1 * m2)  # Adding a spin-dependent term for mass

    return mass


def calculate_baryon_mass(tspin, q1, q2, q3, spin=(1, 1, 1)):
    """Calculate the mass of a baryon given its spin and the masses of its constituent quarks."""
    # Masses for quarks (in MeV/c^2)
    quark_masses = {
        'u': 363,  # Up quark mass
        'd': 363,  # Down quark mass
        's': 538,  # Strange quark mass
    }

    m1 = quark_masses[q1]
    m2 = quark_masses[q2]
    m3 = quark_masses[q3]

    Aprime = np.power(2 * quark_masses['u'], 2) * 50  # A constant for spin-spin interaction
    m123 = m1 + m2 + m3
    mass = 0

    if tspin == 0.5:
        totals = (1, 1, 0)

        s1dots2 = spin_dot_spin(totals[0], 0.5, 0.5)
        s1dots3 = spin_dot_spin(totals[1], 0.5, 0.5)
        s2dots3 = spin_dot_spin(totals[2], 0.5, 0.5)

        mass1 = m123 + Aprime * ((s1dots2 / (m1 * m2)) + (s1dots3 / (m1 * m3)) + (s2dots3 / (m2 * m3)))

        s1dots2 = spin_dot_spin(totals[1], 0.5, 0.5)
        s1dots3 = spin_dot_spin(totals[2], 0.5, 0.5)
        s2dots3 = spin_dot_spin(totals[0], 0.5, 0.5)

        mass2 = m123 + Aprime * ((s1dots2 / (m1 * m2)) + (s1dots3 / (m1 * m3)) + (s2dots3 / (m2 * m3)))

        s1dots2 = spin_dot_spin(totals[2], 0.5, 0.5)
        s1dots3 = spin_dot_spin(totals[0], 0.5, 0.5)
        s2dots3 = spin_dot_spin(totals[1], 0.5, 0.5)

        mass3 = m123 + Aprime * ((s1dots2 / (m1 * m2)) + (s1dots3 / (m1 * m3)) + (s2dots3 / (m2 * m3)))

        mass = (1 / 3) * (mass1 + mass2 + mass3)
    elif tspin == 1.5:
        totals = (1, 1, 1)

        s1dots2 = spin_dot_spin(totals[0], 0.5, 0.5)
        s1dots3 = spin_dot_spin(totals[1], 0.5, 0.5)
        s2dots3 = spin_dot_spin(totals[2], 0.5, 0.5)

        mass = m1 + m2 + m3 + Aprime * ((s1dots2 / (m1 * m2)) + (s1dots3 / (m1 * m3)) + (s2dots3 / (m2 * m3)))
    else:
        raise ValueError("Total spin must (currently) be either 0.5 or 1.5 for baryons.")

    return mass


def calculate_baryon_mass_wf(wavefunction):
    """Calculate the mass of a baryon given its spin and wave function."""
    # Masses for quarks (in MeV/c^2)
    quark_masses = {
        'u': 363,  # Up quark mass
        'd': 363,  # Down quark mass
        's': 538,  # Strange quark mass
    }

    m123 = quark_masses[wavefunction.flavors[0]] + quark_masses[wavefunction.flavors[1]] + quark_masses[wavefunction.flavors[2]]
    mass = 0
    spins = None
    Aprime = np.power(2 * quark_masses['u'], 2) * 50  # A constant for spin-spin interaction
    for coeff, combinations in wavefunction.wave_function.wave_function:
        q1 = combinations[0][0]
        q2 = combinations[0][1]
        q3 = combinations[0][2]
        s1 = combinations[1][0]
        s2 = combinations[1][1]
        s3 = combinations[1][2]

        m1 = quark_masses[q1]
        m2 = quark_masses[q2]
        m3 = quark_masses[q3]

        similarity = determine_similarity(combinations[1])
        match similarity:
            case 0: spins = (0, 0, 0)
            case 1: spins = (1, 0, 0)
            case 2: spins = (0, 1, 0)
            case 3: spins = (0, 0, 1)
            case 4: spins = (1, 1, 1)
        s1dots2 = spin_dot_spin(spins[0], s1, s2)
        s1dots3 = spin_dot_spin(spins[1], s1, s3)
        s2dots3 = spin_dot_spin(spins[2], s2, s3)

        mass += np.power(coeff, 2) * (m123 + Aprime * ((s1dots2 / (m1 * m2)) + (s1dots3 / (m1 * m3)) + (s2dots3 / (m2 * m3))))

    return mass / np.power(wavefunction.wave_function.normalization, 2)


if __name__ == "__main__":
    # Calculating rho(770) mass
    total_spin = 1
    quark1 = 'u'
    quark2 = 'd'

    meson_mass = calculate_meson_mass(total_spin, quark1, quark2)
    print(
        f"The mass of the meson with quarks {quark1} and {quark2} and total spin {total_spin} is approximately {meson_mass:.2f} MeV/c^2.")

    # Calculating K*(892)0 mass
    total_spin = 1
    quark1 = 'd'
    quark2 = 's'

    meson_mass = calculate_meson_mass(total_spin, quark1, quark2)
    print(
        f"The mass of the meson with quarks {quark1} and {quark2} and total spin {total_spin} is approximately {meson_mass:.2f} MeV/c^2.")

    # Calculating sigma0 mass
    total_spin = 0.5
    quark1 = 'u'
    quark2 = 'd'
    quark3 = 's'

    baryon_mass = calculate_baryon_mass(total_spin, quark1, quark2, quark3)
    print(
        f"The mass of the baryon with quarks {quark1}, {quark2}, and {quark3} and total spin {total_spin} is approximately {baryon_mass:.2f} MeV/c^2.")

    # Calculating omega mass
    total_spin = 1.5
    quark1 = 's'
    quark2 = 's'
    quark3 = 's'

    baryon_mass = calculate_baryon_mass(total_spin, quark1, quark2, quark3)
    print(
        f"The mass of the baryon with quarks {quark1}, {quark2}, and {quark3} and total spin {total_spin} is approximately {baryon_mass:.2f} MeV/c^2.")

    total_spin = 1.5
    quark1 = 'u'
    quark2 = 'u'
    quark3 = 'u'

    baryon_mass = calculate_baryon_mass(total_spin, quark1, quark2, quark3)
    print(
        f"The mass of the baryon with quarks {quark1}, {quark2}, and {quark3} and total spin {total_spin} is approximately {baryon_mass:.2f} MeV/c^2.")

    baryon = Baryon(flavors=('u', 'u', 'd'), baryon_isospin=0.5, baryon_spin=(0.5, 0.5))
    print(f"{baryon.baryon_type} wave function: {baryon.wave_function}")
    baryon_mass = calculate_baryon_mass_wf(baryon)
    print(f"The mass of the {baryon.baryon_type} is approximately {baryon_mass:.2f} MeV/c^2.")
