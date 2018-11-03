from numpy import pi

ATEN  = 1.17728057e-9    # surface tension parameter
G_THD = 8e-11             # thermodynamic parameter (Ghan et al., 2011, Eqn A4)
                         # for T=280 K, p=1000 mb. Could be calculated from
                         # first principles. Note error in Ghan's 2013 paper where
                         # exponent is given as -9.
ALPHA = 5.53786731e-4    # (Ghan et al., 2011, Eqn A7) for T=280 K, p=1000 mb
GAMMA = 3374173.8        # gammastar (Ghan et al., 2011, Eqn A8) for T=280 K,
                         # p=1000 mb. Should be unitless
#RHO_AIR = 1.2e-3         # air density, g cm**-3, approx. SLP and temperature
RHO_AIR = 1.2            # air density, kg m**-3, approx. SLP and temperature
RVOLTOEFF = 0.8          # Martin et al., JAS 1994
A_WOOD = 2.4e-6         # Wood, personal communication, at T=290 K. [what are units? paper says 2.4e-3 g m**-4 = 2.4e-6 kg m**4]

R_EARTH = 6.371e6        # m
A_EARTH   = 4 * pi * R_EARTH**2  # cm**2
DAYS_IN_YEAR = 365.25

BETA = 4.7               # m**3 kg**-1 s**-1
RHO_WATER = 1000.        # density of water (kg m/3)

R_EMB = 22.e-6           # embryo (drizzle?) radius (m)

SOLAR_CONSTANT = 1361.0  # W m**-2
