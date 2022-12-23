#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Purpose
"""
# Standard libraries.
from math import log10

# Third party libraries.
from astropy import units as u

__date__ = "2019/12/20 18:55:26 hoel"
__author__ = "Berthold Höllmann"
__copyright__ = "Copyright © 2019 by Berthold Höllmann"
__credits__ = ["Berthold Höllmann"]
__maintainer__ = "Berthold Höllmann"
__email__ = "berhoel@gmail.com"

g = 9.81 * u.m / u.s ** 2
nue = 1.1883e-6 * u.m ** 2 / u.s  # for seewater, salinity: 3.5%,
rho = 1.025 * 1000 * u.kg / u.m ** 3  # 15°C, according to [5], p. 351


def nue_calc(salin, temp):
    """calculates the kinematic viscisity of water, returns value in m^2/s

    salin = salinity of the water
    temp  = temperature of the water  in degrees centigrade
    """
    return 1.0e-6 * (0.014 * salin + (0.000645 * temp - 0.0503) * temp + 1.75)


def F_n(speed, L):
    """Froude number, according to [5], p. 323"""
    return speed / pow(L * g, 0.5)


def R_n(speed, L):
    """Reynolds number, according to [5], p. 323"""
    return speed * L / nue


def C_F(speed, L):
    """Coefficient of frictional resistance according to the ITTC-1957 formula,
    [2], p253"""
    return 0.075 / pow((log10(R_n(speed, L).value) - 2.0), 2.0)


def R_F(speed, ship):
    """frictional resitance according to the ITTC-57 formula"""
    return C_F(speed, ship.L) * 0.5 * rho * pow(speed, 2) * ship.S

# Local Variables:
# mode: python
# compile-command: "python ./setup.py test"
# time-stamp-pattern: "30/__date__ = \"%:y/%02m/%02d %02H:%02M:%02S %u\""
# End:
