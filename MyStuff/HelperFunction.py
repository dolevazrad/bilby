def calculate_chirp_mass_and_ratio(m1, m2):
    chirp_mass = (m1 * m2) ** (3./5.) / (m1 + m2) ** (1./5.)
    mass_ratio = m2 / m1
    return chirp_mass, mass_ratio
def calculate_m1_m2_from_chirp_mass_and_ratio(chirp_mass, mass_ratio):
    # Since (m1 * m2) = (chirp_mass * (m1 + m2)^(1/5))^5 / (m1 * m2)^(3/5)
    # and mass_ratio = m2 / m1, we can solve the quadratic equation for m1:
    # m1^2 - (chirp_mass * (1 + mass_ratio)^(1/5))^5 / mass_ratio^(3/5) * m1 + chirp_mass^5 = 0
    A = 1
    B = -(chirp_mass * (1 + mass_ratio) ** (1./5.)) ** 5 / mass_ratio ** (3./5.)
    C = chirp_mass ** 5
    m1 = (-B + (B**2 - 4 * A * C)**0.5) / (2 * A)
    m2 = mass_ratio * m1
    return m1, m2