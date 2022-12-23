#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Definitions to calculate ships ressitance according to Holtrop.

.. rubric:: References
.. [1] Ir. J. Holtrop, A Statistical Analysis of Performance Test Results,
       International shipbuilding progress. Vol. 24, 1977, pp. 23 ff
.. [2] J. Holtrop and G. G. J. Mennen, A Statistical Power Prediction
       Method, International shipbuilding progress. Vol. 25, 1978, pp. 253 ff
.. [3] J. Holtrop and G. G. J. Mennen, An Approximate Power Predition Method,
       International shipbuilding progress. Vol. 29, 1982, pp. 166 ff
.. [4] J. Holtrop, A Statistical Re-analysis of Resistance and Propulsion
       Data, International shipbuilding progress. Vol. 31, 1984, pp. 272 ff
.. [5] H. Schneekluth, Hydromechanik zum Schiffsentwurf: Vorlesungen. 3., verb.
       u. erw. Aufl., Herford: Koehler, 1988
"""

# Standard libraries.
import math

# Third party libraries.
from astropy import units as u

from . import ship, hydro

__date__ = "2019/12/20 18:57:01 hoel"
__author__ = "Berthold Höllmann"
__copyright__ = "Copyright © 2019 by Berthold Höllmann"
__credits__ = ["Berthold Höllmann"]
__maintainer__ = "Berthold Höllmann"
__email__ = "berhoel@gmail.com"
__scm_version__ = "$Revision$"[10:-1]


class Holtrop(ship.Ship):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # According to [4], p.  273:
        #
        # .. math::
        #
        #     d = -0.9
        self.d = -0.9

    @property
    def L_R(ship):
        r"""
        According to [4], p.  272:

        .. math::

            L_R = L \left( 1 - C_P + \frac{0.06 C_P \text{lcb}}{4 C_P - 1}\right)
        """
        return ship.L * (
            1.0 - ship.C_P + 0.06 * ship.C_P * ship.lcb / (4.0 * ship.C_P - 1.0)
        )

    @property
    def k_1(ship):
        r"""
        Calculates (1+k_1) according to Holtrop[4], page 272:
    
        .. math::
    
            1 + k_1 = 0.93 + 0.487118 c_{14} \left( \frac B L \right)^{1.06806} \left(
                      \frac T L \right)^{0.46106} \left( \frac L{L_R} \right)^{0.121563}
                      \left( \frac{L^3}∇ \right)^{0.36486} \left(1 - C_P
                      \right)^{-0.604247}
        """
        return 0.93 + 0.487118 * ship.c_14 * math.pow(
            ship.B / ship.L, 1.06806
        ) * math.pow(ship.T / ship.L, 0.46106) * math.pow(
            ship.L / ship.L_R, 0.121563
        ) * math.pow(
            pow(ship.L, 3) / ship.Nab, 0.36486
        ) * math.pow(
            1.0 - ship.C_P, -0.604247
        )

    @property
    def c_14(ship):
        r"""
        According to [4], p.  272
    
        ..  table:: C_Stern according to table: (see [4], p.  272):
    
        =================================== ==================
        Afterbody form                      $C_{\text{Stern}}$
        =================================== ==================
        Pram with gondola                   -25
        V-shaped section                    -10
        Normal section shape                0
        U-shaped sections with Hogner stern 10
        =================================== ==================
    
        .. math::
    
            c_{14} = 1 + 0.011 C_{\text{Stern}}
        """
        return 1.0 + 0.011 * ship.C_Stern

    @property
    def C_A(ship):
        if super().C_A is None:
            return ship.C_A_calc()
        else:
            return super().C_A

    def C_A_calc(ship):
        r"""
    Correlation allowance coefficient according to [3], p. 168:
    \begin{equation}
      C_A = 0.006 (L + 100)^{-0.16} - 0.00205 + 0.003 \sqrt{\frac L{7.5}} C_B^4
            c_2 (0.04 - c_4)
    \end{equation}
    """
        return (
            0.006 * math.pow((ship.L.value + 100.0), -0.16)
            - 0.00205
            + 0.003
            * math.pow(ship.L.value / 7.5, 0.5)
            * math.pow(ship.C_B, 4)
            * ship.c_2
            * (0.04 - ship.c_4)
        ).value

    @property
    def c_2(ship):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            c_2 = \exp\left(-1.89 \sqrt{c_3} \right)
        """
        return math.exp(-1.89 * math.sqrt(ship.c_3))

    @property
    def c_3(ship):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            c_3 = \frac{0.56 A_{BT}^{1.5}}{B T \left(0.31 \sqrt{A_BT} + T_F - h_B\right)}
        """
        return (
            0.56
            * math.pow(ship.A_BT.value, 1.5)
            / (
                ship.B
                * ship.T
                * (0.31 * math.sqrt(ship.A_BT.value) + ship.T_F.value - ship.h_b.value)
            )
        ).value

    @property
    def c_4(ship):
        r"""
        According to [3], p.  168:
    
        .. math::
    
            c_4 = \left\{
            \begin{array}{lll}
              \frac{T_F}L & \text{when} & \frac{T_F}L ≤ 0.04\\
              0.04        & \text{when} & \frac{T_F}L > 0.04
            \end{array}
            \right.
        """
        return max(ship.T_F / ship.L, 0.04)

    @property
    def c_17(ship):
        r"""
        According to [4], p.  272:
    
        .. math::
    
            c_17 = 6919.3 C_M^{-1.3346} \left( \frac∇{L^3}\right)^{2.00977}
                   \left( \frac L B - 2 \right)^{1.40692}
    
        \end{equation}
        """
        return (
            6919.3
            * math.pow(ship.C_M, -1.3346)
            * math.pow((ship.Nab / pow(ship.L, 3)), 2.00977)
            * math.pow((ship.L / ship.B) - 2.0, 1.40692)
        )

    @property
    def m_3(ship):
        r"""
        According to [4], p.  272:
    
        .. math::
    
            m_3 = -7.2035 \left( \frac B L \right)^{0.326869} \left( \frac T B
              \right)^{0.605375}
        """
        return (
            -7.2035
            * math.pow(ship.B / ship.L, 0.326869)
            * math.pow(ship.T / ship.B, 0.605375)
        )

    @property
    def c_5(ship):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            c_5 = 1 - 0.8 \frac{A_T}{B T C_M}
        """
        return (1.0 - 0.8 * ship.A_T / (ship.B * ship.T * ship.C_M)).value

    @property
    def lambda_(ship):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            λ = \left\{
            \begin{array}{lll}
              1.446 C_P - 0.03 \frac L B& \text{when} & \frac L B < 12\\
              1.446 C_P - 0.36          & \text{when} & \frac L B > 12
            \end{array}
            \right.
        """
        if (ship.L / ship.B) < 12.0:
            return (1.446 * ship.C_P - 0.03 * (ship.L / ship.B)).value
        else:
            return (1.446 * ship.C_P - 0.36).value

    def c_6(ship, speed):
        r"""
        According to [3], p.  168:
    
        .. math::
    
            c_6 = \left\{
            \begin{array}{lll}
              0.2 (1 - 0.2 F_{nT}) & \text{when} & F_{nT} < 5\\
              0                    & \text{when} & F_{nT} ≥ 5
            \end{array}
            \right.
        """
        if ship.F_nT(speed) < 5.0:
            return 0.2 * (1.0 - 0.2 * ship.F_nT(speed))
        else:
            return 0

    @property
    def c_15(ship):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            c_{15} = \left\{
            \begin{array}{lll}
              -1.69385                       & \text{when} & \frac{L^3}∇ < 512\\
              -1.69385 + \frac{\left(\frac L{∇^⅓} - 8\right)}{2.36}
                                             & \text{when} & 512 < \frac{L^3}∇ < 1726.91\\
              0                              & \text{when} & \frac{L^3}∇ > 1726.91
            \end{array}
            \right.
        """
        if (pow(ship.L, 3) / ship.Nab) < 512:
            return -1.69385
        elif (pow(ship.L, 3) / ship.Nab) < 1726.91:
            return -1.69385 + ((ship.L / pow(ship.Nab, 1.0 / 3.0)) - 8.0) / 2.36
        else:
            return 0.0

    @property
    def calc_i_E(ship):
        r"""
        Angle if the waterline at the bow in degrees with referende to the
        centre Plane but neglecting the local shape at the stem.  Formula
        according to [3], pp.  167.
    
        .. math::
    
            i_E = 1 + 89 \exp \left( -\left(\frac L B \right)^{0.80856} \left( 1 - C_{WP}
                  \right)^{0.30484} \left(1 - C_P - 0.0225 \text{lcb} \right)^{0.6367} \left(
                  \frac{L_R}B \right)^{0.34574} \left( \frac{100 ∇}{L^3}
                  \right)^{0.16302} \right)
        """
        return (
            1.0
            + 89.0
            * math.exp(
                -math.pow((ship.L / ship.B), 0.80856)
                * math.pow((1 - ship.C_WP), 0.30484)
                * math.pow((1 - ship.C_P - 0.0225 * ship.lcb), 0.6367)
                * math.pow((ship.L_R / ship.B), 0.34574)
                * pow((100.0 * ship.Nab / pow(ship.L, 3)), 0.16302)
            )
        ) * u.degree

    def m_4(ship, speed):
        r"""
        According to [4], p.  273:
    
        .. math::
    
            c_{15} = 0.4 \exp \left( -0.034 F_n^{-3.29} \right)
        """
        return (
            ship.c_15
            * 0.4
            * math.exp(-0.034 * math.pow(hydro.F_n(speed, ship.L), -3.29))
        )

    def R_W(ship, speed):
        r"""
        Wave ressistance formula according to [4], p.  273:
    
        .. math::
    
            R_W = \left\{
            \begin{array}{lll}
              R_{W-A}                    & \text{when} & F_n < 0.40\\
              R_{W-A_{0.4}} + \left(10
                   F_n - 4\right) \frac{
                   \left(R_{W-B_{0.55}} -
                   R_{W-A_{0.4}} \right)}
                   {0.5}                 & \text{when} & 0.40 < F_n < 0.55\\
              R_{W-B}                    & \text{when} & F_n > 0.55
            \end{array}
            \right.
    
        Here $R_{W-A_{0.4}}$ is the wave resistance prediction for $F_n =
        0.40$ and $R_{W-B_{0.55}}$ is the wave resistance prediction for
        $F_n = 055$ according to the respective formulae.
        """

        F = hydro.F_n(speed, ship.L)
        if F < 0.4:
            return ship.R_WA(speed, F)
        elif F < 0.55:
            return (
                ship.R_WA(speed, 0.4)
                + (10.0 * F - 4)
                * (ship.R_WB(speed, 0.55) - ship.R_WA(speed, 0.4))
                / 1.5
            )
        else:
            return ship.R_WB(speed, F)

    def R_WA(ship, speed, F_n):
        r"""
        Wave ressistance formula for speed range F_n < 0.4 according
        to [4], p.  273:

        .. math::

            R_{W-A} = c_1 c_2 c_5 ∇ ρ g \exp\left(m_1 F_n^d + m_4 \cos \left(
                      \lambda F_n^{-2}\right)\right)
        """
        return (
            ship.c_1
            * ship.c_2
            * ship.c_5
            * ship.Nab()
            * hydro.rho
            * hydro.g
            * math.exp(
                ship.m_1 * math.pow(F_n, ship.d)
                + ship.m_4(speed) * math.cos(ship.lambda_ * math.pow(F_n, -2.0))
            )
        )

    def R_WB(ship, speed, F_n):
        r"""
        Wave ressistance formula for speed range F_n > 0.55 according
        to [4], p.  272:

        .. math::

            R_{W-B} = c_{17} c_2 c_5 ∇ ρ g \exp\left(m_3 F_n^d + m_4 \cos \left(
                      \lambda F_n^{-2}\right)\right)
        """
        return (
            ship.c_17
            * ship.c_2
            * ship.c_5
            * ship.Nab
            * hydro.rho
            * hydro.g
            * math.exp(
                ship.m_3 * math.pow(F_n, ship.d)
                + ship.m_4(speed) * math.cos(ship.lambda_ * math.pow(F_n, -2.0))
            )
        )

    def R_app(ship, speed):
        r"""
        Appedage resistance according to [3], p.  167:
    
        .. math::
    
            R_{APP} = = 0.5 ρ V^2 S_{APP} \left 1 + k_2 \right)_{eq} C_F
        """

        def calc_k_2_eq(App):
            r"""
            Calculation of equivalent 1 + k_2 value for a combination of
            appendages:
    
            .. math::
    
                \left(1 + k_2\right) = \frac{\Sum \left(1 + k_2 \right) S_{APP}}{\Sum S_{APP}}
            """
            S_Sum = 0.0 * u.m ** 2
            k_2_Sum = 0.0 * u.m ** 2
            for (S_app, k_2) in App:
                S_Sum = S_Sum + S_app
                k_2_Sum = k_2_Sum + (S_app * k_2)
            if S_Sum.value > 0.0:
                return (S_Sum, (k_2_Sum / S_Sum))
            else:
                return (0.0 * u.m ** 2, 0.0)

        (S_app, k_2_eq) = calc_k_2_eq(ship.App)
        return (
            0.5 * hydro.rho * pow(speed, 2) * S_app * k_2_eq * hydro.C_F(speed, ship.L)
        )

    def R_B(ship, speed):
        r"""
        Additional resistance due to the presence of a bulbous bow near
        the surface, according to [3], p.  168:
    
        .. math::
    
            R_B = 0.11 \exp \left(-3 P_B^{-2} \right) \frac{F_{ni}^3 A{BT}^{1.5} ρ g}
                  {1 + F_{ni}^2}
        """
        if ship.A_BT.value > 0:
            return (
                0.11
                * math.exp(-3.0 * pow(P_B(ship), -2))
                * math.pow(ship.F_ni(speed), 3.0)
                * (ship.A_BT * pow(ship.A_BT, 0.5))
                * hydro.rho
                * hydro.g
                / (1.0 + pow(ship.F_ni(speed), 2))
            )
        else:
            return 0 * u.N

    # resistance calculation:
    def R(ship, speed):
        r"""
        Calculating of resistance of merchant ships according to the
        statistical method of J.  Holtrop, [4], p.  272:
    
        .. math::
    
            R_{\text{Total}} = hydro.R_F(1+K_1)+R_{APP}+R_W+R_B+R_{TR}+R_A
        """
        R_total = (
            hydro.R_F(speed, ship) * ship.k_1
            + ship.R_app(speed)
            + ship.R_W(speed)
            + ship.R_B(speed)
            + ship.R_TR(speed)
            + ship.R_A(speed)
        )
        return R_total

    def R_TR(ship, speed):
        r"""
        Additional pressure resistance due to immersed transom, according
        to [3], p.~168:
    
        .. math::
    
            R{TR} = 0.5 ρ V^2 A_T c_6
    
        """
        return 0.5 * hydro.rho * pow(speed, 2) * ship.A_T * ship.c_6(speed)

    def R_A(ship, speed):
        r"""
        Model-ship correlation resistance according to [3], p.  168:
    
        .. math::
    
            R_A = 0.5 ρ V^2 S C_A
    
        """
        return 0.5 * hydro.rho * pow(speed, 2) * ship.S * ship.C_A

    def F_nT(ship, speed):
        r"""
        Froude number based on the transom immersion, according to [3], p.
        168:
    
        .. math::
    
            F_{nT} = \frac V{\sqrt{\frac{2 g A_T}{B + B C_{WP}}}}}
        """
        return speed / pow(
            2.0 * hydro.g * ship.A_T / (ship.B + ship.B * ship.C_WP), 0.5
        )


def C_A_ITTC78(L, k_s):
    r"""
    Correlation allowance coefficient increase according to [3], p.
    168:

    .. math::

        C_A = \frac{0.105 k_s^{\frac 1 3} - 0.005579}{L^{\frac 1 3}}
    """
    return (0.105 * math.pow(k_s, (1.0 / 3.0)) - 0.005579) / pow(L, (1.0 / 3.0))


def C_V(speed, ship):
    r"""
    Viscous resistance coefficient according to [4], p.  274:

    .. math::

        C_V = (1 + k) C_F + C_A
    """
    return k_1(ship) * C_F(speed, ship.L()) + ship.C_A()


def c_1(ship):
    r"""
    according to [4], p.  273:

    .. math::

        c_1 = 2223105 c_7^{3.78613} \left( \frac T B \right)^{1.07961} 90 -
              i_E)^{-1.37565}
    """
    return (
        2223105.0
        * math.pow(c_7(ship), 3.78613)
        * math.pow((ship.T() / ship.B()), 1.07961)
        * math.pow((90.0 - ship.i_E().value), -1.37565)
    )


def c_7(ship):
    r"""
    According to [4], p.  273:

    .. math::

        c_7 = \left\{
        \begin{array}{lll}
          0.229577\left(\frac B L \right)^{0.33333} & \text{when} & \frac B L < 0.11\\
          \frac B L                                 & \text{when} & 0.11 < \frac B L <
                                                      0.25\\
          0.5 - 0.0625 \frac B L                    & \text{when} & \frac B L > 0.25
        \end{array}
        \right.
    """
    if (ship.B() / ship.L()) < 0.11:
        return 0.229577 * math.pow((ship.B() / ship.L()), 0.33333)
    elif (ship.B() / ship.L()) < 0.25:
        return ship.B() / ship.L()
    else:
        return 0.5 - 0.0625 * (ship.L() / ship.B())


def c_8(ship):
    r"""
    According to [4], p.  273:

    .. math::

        c_8 = \left\{
        \begin{array}{lll}
          \frac{B S}{L D T_A}                   & \text{when} & \frac B{T_A} < 5\\
          \frac{S7\frac B{T_A} - 25}
            {LD \left(\frac B{T_A} - 3 \right)} & \text{when} & \frac B{T_A} > 5
        \end{array}
        \right.
    """
    if (ship.B() / ship.T_A()) < 5.0:
        return ship.B() * ship.S() / (ship.L() * ship.D() * ship.T_A())
    else:
        return (
            ship.S()
            * (7.0 * (ship.B() / ship.T_A()) - 25.0)
            / (ship.L() * ship.D() * ((ship.B() / ship.T_A()) - 3.0))
        )


def c_9(ship):
    r"""
    According to [4], p.  273:

    .. math::

        c_9 = \left\{
        \begin{array}{lll}
          c_8                      & \text{when} & c_8 < 28\\
          32 - \frac{16}{c_8 - 24} & \text{when} & c_8 > 28
        \end{array}
        \right.
    """
    C_8 = c_8(ship)
    if C_8 < 28:
        return C_8
    else:
        return 32.0 - 16.0 / (C_8 - 24)


def c_10():
    pass


def c_11(ship):
    r"""
    According to [4], p.  273:

    .. math::

        c_{11} = \left\{
        \begin{array}{lll}
          \frac{T_A}D            & \text{when} & \frac{T_A}D < 2\\
          0.0833333 \left( \frac{T_A}D \right)^3 + 1.33333
                                 & \text{when} & \frac{T_A}D > 2
        \end{array}
        \right.
    """
    val = ship.T_A() / ship.D()
    if val < 2.0:
        return val
    else:
        retrun(0.0833333 * math.pow(val, 3.0) + 1.33333)


def c_12():
    pass


def c_13():
    pass


def c_16(ship):
    r"""
    According to [4], p.  273:

    .. math::

        c_{16} = \left\{
        \begin{array}{lll}
          8.07981 C_P - 13.8673 C_P^2 + 6.984388 C_P^3& \text{when} & C_P < 0.8\\
          1.73014 - 0.7067 C_P                        & \text{when} & C_P > 0.8
        \end{array}
        \right.
    """
    if ship.C_P() < 0.8:
        return (
            8.07981 * ship.C_P()
            - 13.8673 * math.pow(ship.C_P(), 2.0)
            + 6.984388 * math.pow(ship.C_P(), 3.0)
        )
    else:
        return 1.73014 - 0.7067 * ship.C_P()


def c_18():
    pass


def c_19(ship):
    r"""
    According to [4], p.  273-274:

    .. math::

        c_{19} = \left\{
        \begin{array}{lll}
          \frac{0.12997}{0.95 - C_B} -
            \frac{0.11056}{0.95 - C_P}   & \text{when}& C_P < 0.7\\
          \frac{ 0.18567}{1.3571 - C_M} -
            0.71276 + 0.38648 C_P        & \text{when}& C_P > 0.7
        \end{array}
        \right.
    """
    if ship.C_P() < 0.7:
        return 0.12997 / (0.95 - ship.C_B()) - 0.11056 / (0.95 - ship.C_P())
    else:
        return 0.18567 / (1.3571 - ship.C_M) - 0.71276 + 0.38648 * ship.C_P()


def c_20(ship):
    r"""
    According to [4], p.  274:

    .. math::

        c_{20} = 1 + 0.015 C_{\text{Stern}}
    """
    return 1.0 + 0.015 * ship.C_Stern()


def c_21():
    pass


def C_P1(ship):
    r"""
    According to [4], p.  274:

    .. math::

        C_{P1} = 1.45 C_P - 0.315 - 0.0225 \text{lcb}
    """
    return 1.45 * ship.C_P() - 0.315 - 0.0225 * ship.lcb()


def m_1(ship):
    r"""
    According to [4], p.  273:

    .. math::

        m_1 = 0.0140407 \frac L T - 1.75254 \frac{∇^{\frac13}}L - 4.79323
              \frac B L - c_{16}
    """
    return (
        0.0140407 * (ship.L() / ship.T())
        - 1.75254 * pow(ship.Nab(), (1.0 / 3.0)) / ship.L()
        - 4.79323 * ship.B() / ship.L()
        - c_16(ship)
    )


def m_2():
    pass


def P_B(ship):
    r"""
    A measure for the emergence of the bow, according to [3], p.  168:

    .. math::

        P_B  = \frac{0.56 \sqrt{A_{BT}}}{T_F - 1.5 h_B}

    \end{equation}
    """
    return 0.56 * ship.A_BT() ** (1 / 2) / (ship.T_F() - 1.5 * ship.h_b())


def F_ni(speed, ship):
    r"""
    Froude number based on the immersion, according to [3], p.  168:

    .. math::

        \frac V{\sqrt{g \left( T_F - h_B - 0.25 \sqrt{A_{BT}} \right) + 0.15 V^2}}
    """
    return speed / pow(
        hydro.g * (ship.T_F() - ship.h_b() - 0.25 * pow(ship.A_BT(), 0.5))
        + 0.15 * pow(speed, 2),
        0.5,
    )


# prediction of delivered power:
def w_single(speed, ship):
    r"""
    Wake prediction for single screw ships according to [4], p.  273:

    .. math::

        w = c_9 c{20} C_V \frac L{T_A} \left( 0.050776 + 0.93405 c_{11} \frac{C_V}
            {\left(1 - C_{P1} \right)} \right) + 0.27915 c_{20} \sqrt{\frac B{L\left(
            1 - C_{P1} \right)}} c_{19} c_{20}

    """
    return (
        c_9(ship)
        * c_20(ship)
        * C_V(speed, ship)
        * (ship.L() / ship.T_A())
        * (0.050776 + 0.93405 * c_11(ship) * (C_V(speed, ship) / (1 - C_P1(ship))))
    ) + (
        0.27915 * c_20(ship) * math.sqrt(ship.B() / (ship.L() * (1 - C_P1(ship))))
        + c_19(ship) * c_20(ship)
    )


def t_single(ship):
    r"""
    Thrust decuction prediction for single screw ships according to
    [4], p.  274:

    .. math::

        t = \frac{0.25014 \left(\frac B L \right)^{0.28956} \left( \frac{\sqrt{B T}}D
                  \right)^{0.2624}}
                 {\left(1 - C_P + 0.0225 \text{lcb}\right)^{0.01762}} +
            0.0015 C_{\text{stern}}

    """
    return (
        0.25014
        * math.pow(ship.B() / ship.L(), 0.28956)
        * math.pow(math.sqrt((ship.B() * ship.T()).value) / ship.D().value, 0.2624)
        / math.pow(1 - ship.C_P() + 0.0225 * ship.lcb(), 0.01762)
        + 0.0015 * ship.C_Stern()
    )


def eta_R_single(ship):
    r"""
    The relatigve-rotative efficiency prediction for single screw
    ships according according to [3], pp.  168:

    .. math::

        η_R = 0.9922 - 0.05908 \frac{A_E}{A_O} +
           0.07424 \left( C_P - 0.0225 \text{lcb} \right)
    """
    return (
        0.9922
        - 0.05908 * ship.A_E() / ship.A_O()
        + 0.07424 * (ship.C_P() - 0.0225 * ship.lcb())
    )


def w_single_open_stern(speed, ship):
    r"""
    Wake prediction for single screw ships with open stern (as
    sometimes applied on slender, fast sailing ships) according to
    [3], p.  169:

    .. math::

        w = 0.3 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{B T}}

    """
    return (
        0.3 * ship.C_B()
        + 10.0 * C_V(speed, ship) * ship.C_B()
        - 0.23 * ship.D() / math.sqrt(ship.B() * ship.T)
    )


def t_single_open_stern(speed, ship):
    r"""
    Thrust decuction prediction for single screw ships with open stern
    (as sometimes applied on slender, fast sailing ships) according to
    [3], p.  169:

    .. math::

        t = 0.10

    """
    return 0.1


def eta_R_single_open_stern(ship):
    r"""
    The relatigve-rotative efficiency prediction for single screw
    ships with open stern (as sometimes applied on slender, fast
    sailing ships) according according to [3], pp.  168:

    .. math::

        η_R = 0.98

    """
    return 0.98


def w_twin(speed, ship):
    r"""
    Wake prediction for twin screw ships according to [3], p.  169:

    .. math::

        w = 0.3095 C_B + 10 C_V C_B - 0.23 \frac D{\sqrt{B T}}

    """
    return (
        0.3095 * ship.C_B()
        + 10.0 * C_V(speed, ship) * ship.C_B()
        - 0.23 * ship.D() / math.sqrt(ship.B() * ship.T())
    )


def t_twin(ship):
    r"""
    Thrust decuction prediction for twin screw ships according to [3],
    p.  169:

    .. math::

        t = 0.325 C_B - 0.1885 \frac D {\sqrt{B T}}

    """
    return 0.325 * ship.C_B() - 0.1885 * ship.D() / math.sqrt(ship.B() * ship.T())


def eta_R_twin(ship):
    r"""
    The relatigve-rotative efficiency prediction for twin screw ships
    according according to [3], pp.  168:

    .. math::

        η_R = 0.9737 + 0.111 \left( C_P - 0.0225 \text{lcb} \right) + 0.06325 \frac P D
    """
    return (
        0.9737
        + 0.111 * (ship.C_P() - 0.0225 * ship.lcb())
        + 0.06325 * ship.P() / ship.D()
    )


# Local Variables:
# mode: python
# compile-command: "python ./setup.py test"
# time-stamp-pattern: "30/__date__ = \"%:y/%02m/%02d %02H:%02M:%02S %u\""
# End:
