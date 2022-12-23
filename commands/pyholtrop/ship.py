#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Hold ship dimensions.
"""
# Standard libraries.
import math
import collections

# Third party libraries.
import Pmw
import astropy
from astropy import units as u
from astropy.units import imperial as ui
from astropy.units.quantity import Quantity

from . import hydro

__date__ = "2019/12/20 18:23:34 hoel"
__author__ = "Berthold Höllmann"
__copyright__ = "Copyright © 2019 by Berthold Höllmann"
__credits__ = ["Berthold Höllmann"]
__maintainer__ = "Berthold Höllmann"
__email__ = "berhoel@gmail.com"


u.imperial.enable()


class Ship:
    """
class to describe a ship for resistance calculation.
"""

    __known_methods = ("S", "C_WP", "C_B", "i_E", "C_P", "C_A", "Nab")
    _S_method = "Holtrop"
    __C_WP_method = "Schneekluth_1"
    __C_B_method = "Ayre 1.06"
    __default_speed = 14 * ui.kn
    __needed_units = {
        "L": u.m,  # Quantity("1m").unit,
        "B": u.m,  # Quantity("1m").unit,
        "T": u.m,  # Quantity("1m").unit,
        "D": u.m,  # Quantity("1m").unit,
        "T_F": u.m,  # Quantity("1m").unit,
        "T_A": u.m,  # Quantity("1m").unit,
        "S": u.m ** 2,  # Quantity("1m**2").unit,
        "h_b": u.m,  # Quantity("1m").unit,
        "Nab": u.m ** 3,  # Quantity("1m**3").unit,
        "S_app": [u.m ** 2],  # Quantity("1m**2").unit],
        "k_1": None,
        "k_2": [None],
        "App": [[u.m ** 2, None]],  # Quantity("1m**2").unit, None]],
        "A_BT": u.m ** 2,  # Quantity("1m**2").unit,
        "i_E": u.degree,  # Quantity("1deg").unit,
        "lcb": None,
        "A_T": u.m ** 2,  # Quantity("1m**2").unit,
        "A_WP": u.m ** 2,  # Quantity("1m**2").unit,
        "L_WP": u.m,  # Quantity("1m").unit,
        "B_WP": u.m,  # Quantity("1m").unit,
        "C_Stern": None,
        "C_A": None,
        "C_M": None,
        "C_WP": None,
        "C_P": None,
        "C_B": None,
        "v_Probe": u.m / u.s,  # Quantity("1m/s").unit,
        "R_F": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R_app": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R_W": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R_B": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R_TR": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R_A": 1000.0 * u.N,  # Quantity("1kN").unit,
        "R": 1000.0 * u.N,  # Quantity("1kN").unit,
        "eta_R": None,
        "w": None,
        "t": None,
        "A_E_0": None,
        "c_P_D": None,
        "eta_0": None,
    }

    def __init__(self, **kw):

        self.__keys = collections.defaultdict(lambda: None)
        self.__keys.update(kw)
        # copy.deepcopy(kw)
        # attach variables given on constructor line as attributes and methods
        # to class
        # for key in self.__keys.keys():
        #     self[key] = self.__keys[key]

        # attach those attributes and methods, we don't have given values for on
        # the contructor line, but can be calculated easily
        if "T" not in self.__keys and "T_F" in self.__keys and "T_A" in self.__keys:
            self.T = 0.5 * (self.T_F + self.T_A)
        if "T" in self.__keys and "T_F" not in self.__keys:
            self.T_F = self.T
        if "T" in self.__keys and "T_A" not in self.__keys:
            self.T_A = self.T
        if (
            "C_WP" not in self.__keys
            and "A_WP" in self.__keys
            and "L_WP" in self.__keys
            and "B_WP" in self.__keys
        ):
            self.C_WP = (self.A_WP / (self.L_WP * self.B_WP)).value
        if "App" not in self.__keys:
            App = []
            if "S_app" in self.__keys and "k_2" in self.__keys:
                if len(self.__keys["S_app"]) != len(self.__keys["k_2"]):
                    raise ValueError
                for i in range(len(self.__keys["S_app"])):
                    App.append((self.__keys["S_app"][i], self.__keys["k_2"][i]))
            self.App = App

        # For some parameters we know approximate methods. If there were no
        # values given for these parameters, we will attach the standard methods
        # for these.
        # for method in Ship.__known_methods:
        #     if method not in self.__keys:
        #         setattr(self, method, eval("self." + method + "_calc"))
        return

    @property
    def L(self):
        return self.__keys["L"]

    @L.setter
    def L(self, value):
        self.__keys["L"] = value

    @property
    def h_b(self):
        return self.__keys["h_b"]

    @h_b.setter
    def h_b(self, value):
        self.__keys["h_b"] = value

    @property
    def Nab(self):
        if self.__keys["Nab"] is None:
            return self.Nab_calc()
        else:
            return self.__keys["Nab"]

    @Nab.setter
    def Nab(self, value):
        self.__keys["Nab"] = value

    @property
    def T_F(self):
        return self.__keys["T_F"]

    @T_F.setter
    def T_F(self, value):
        self.__keys["T_F"] = value

    @property
    def T_A(self):
        return self.__keys["T_A"]

    @T_A.setter
    def T_A(self, value):
        self.__keys["T_A"] = value

    @property
    def T(self):
        return self.__keys["T"]

    @T.setter
    def T(self, value):
        self.__keys["T"] = value

    @property
    def B(self):
        return self.__keys["B"]

    @B.setter
    def B(self, value):
        self.__keys["B"] = value

    @property
    def S(self):
        if self.__keys["S"] is None:
            return self.S_calc()
        else:
            return self.__keys["S"]

    @S.setter
    def S(self, value):
        self.__keys["S"] = value

    @property
    def C_Stern(self):
        return self.__keys["C_Stern"]

    @C_Stern.setter
    def C_Stern(self, value):
        self.__keys["C_Stern"] = value

    @property
    def C_WP(self):
        if self.__keys["C_WP"] is None:
            return self.C_WP_calc()
        else:
            return self.__keys["C_WP"]

    @C_WP.setter
    def C_WP(self, value):
        self.__keys["C_WP"] = value

    @property
    def C_B(self):
        if self.__keys["C_B"] is None:
            return self.C_B_calc()
        else:
            return self.__keys["C_B"]

    @C_B.setter
    def C_B(self, value):
        self.__keys["C_B"] = value

    @property
    def i_E(self):
        if self.__keys["i_E"] is None:
            return self.i_E_calc()
        else:
            return self.__keys["i_E"]

    @i_E.setter
    def i_E(self, value):
        self.__keys["i_E"] = value

    @property
    def C_P(self):
        if self.__keys["C_P"] is None:
            return self.C_P_calc()
        else:
            return self.__keys["C_P"]

    @C_P.setter
    def C_P(self, value):
        self.__keys["C_P"] = value

    @property
    def C_A(self):
        return self.__keys["C_A"]

    @C_A.setter
    def C_A(self, value):
        self.__keys["C_A"] = value

    @property
    def App(self):
        return self.__keys["App"]

    @App.setter
    def App(self, value):
        self.__keys["App"] = value

    @property
    def C_M(self):
        return self.__keys["C_M"]

    @C_M.setter
    def C_M(self, value):
        self.__keys["C_M"] = value

    @property
    def A_T(self):
        return self.__keys["A_T"]

    @A_T.setter
    def A_T(self, value):
        self.__keys["A_T"] = value

    @property
    def A_WP(self):
        return self.__keys["A_WP"]

    @A_WP.setter
    def A_WP(self, value):
        self.__keys["A_WP"] = value

    @property
    def L_WP(self):
        return self.__keys["L_WP"]

    @L_WP.setter
    def L_WP(self, value):
        self.__keys["L_WP"] = value

    @property
    def B_WP(self):
        return self.__keys["B_WP"]

    @B_WP.setter
    def B_WP(self, value):
        self.__keys["B_WP"] = value

    @property
    def A_BT(self):
        return self.__keys["A_BT"]

    @A_BT.setter
    def A_BT(self, value):
        self.__keys["A_BT"] = value

    @property
    def lcb(self):
        return self.__keys["lcb"]

    @lcb.setter
    def lcb(self, value):
        self.__keys["lcb"] = value

    def __call__(self):
        return self.__keys

    def __repr__(self):
        out = self.__class__.__name__ + "("
        sep = ""
        for i in self.__keys.keys():
            out = out + sep + i + " = " + repr(self.__keys[i])
            sep = ", "
        return out + ")"

    def S_calc(self, method=None):
        """S_calc(method = 'Holtrop'):
        Calculates the wetted area of the hull. Known methods are:
        Holtrop : Approximated wetted area of a ship hull, according to J.
                  Holtrop and G. G. J. Mennen, An Approximate Power Predition
                  Method, International shipbuilding progress. Vol. 29, 1982,
                  p. 166
        Schenzle:
        Default method is Holtrop.
        """
        if method == None:
            method = self._S_method
        if method == "Holtrop":
            return (
                self.L
                * (2.0 * self.T + self.B)
                * math.pow(self.C_M, 0.5)
                * (
                    0.453
                    + 0.4425 * self.C_B
                    - 0.2862 * self.C_M
                    - 0.003467 * self.B / self.T
                    + 0.3696 * self.C_WP
                )
                + 2.38 * self.A_BT / self.C_B
            )
        elif method == "Schenzle":
            B = self.C_WP * self.B / self.T
            C = self.L / self.B / self.C_M
            A1 = (1.0 + B / 2.0 - math.pow(1.0 + B * B / 4.0, 0.5)) * 2.0 / B
            A2 = 1.0 + C - math.pow(1.0 + C * C, 0.5)
            CN1 = 0.8 + 0.2 * B
            CN2 = 1.15 + 0.2833 * C
            CPX = self.C_B / self.C_M
            CPZ = self.C_B / self.C_WP
            C1 = 1.0 - A1 * math.pow(1.0 - (2.0 * CPZ - 1.0) ** CN1, 0.5)
            C2 = 1.0 - A2 * math.pow(1.0 - (2.0 * CPX - 1.0) ** CN2, 0.5)
            return (2.0 + C1 * B + 2.0 * C2 / C) * self.L * self.T
        else:
            raise KeyError
        raise ShipInternalError()

    def Set_S_Method(self, method):
        self._S_method = method
        return

    def C_WP_calc(self, method=None):
        """C_WP_calc(method)
        Calculates the waterplane coefficient of the hull. Known methods are:
        Schneekluth_1: For ships with U-shaped sections, with not sweeping
                       stern lines:
                        C_WP = 0.95 C_P + 0.17 (1 - C_P)^(1/3)
        Schneekluth_2: For medium shape forms:
                        C_WP = (1 + 2 C_B) / 3
        Schneekluth_3: V-shaped sections, also for sweeping stern lines:
                        C_WP = sqrt(C_B) - 0.025
        Schneekluth_4: For shapes as Schneekluth_3
                        C_WP = C_P^(2/3)
        Schneekluth_5: For shapes as Schneekluth_3
                        C_WP = (1 + 2 (C_B / sqrt(C_M))) / 3
        All formulas according to: H. Schneekluth, Entwerfen von Schiffen:
        Vorlesungen. 3., verb. u. erw. Aufl., Herford: Koehler, 1985
        """

        if method == None:
            method = self.__C_WP_method
        if method == "Schneekluth_1":
            # For ships with U-shaped sections, with not sweeping stern lines:
            return 0.95 * self.C_P + 0.17 * math.pow((1.0 - self.C_P), (1.0 / 3.0))
        elif method == "Schneekluth_2":
            # For medium shape forms:
            return (1.0 + 2.0 * self.C_B) / 3.0
        elif method == "Schneekluth_3":
            # V-shaped sections, also for sweeping stern lines:
            return math.pow(self.C_B, 0.5) - 0.025
        elif method == "Schneekluth_4":
            # For shapes as Schneekluth_3
            return math.pow(self.C_P, (2.0 / 3.0))
        elif method == "Schneekluth_5":
            # For shapes as Schneekluth_3
            return (1.0 + 2.0 * (self.C_B / math.pow(self.C_M, 0.5))) / 3.0
        else:
            raise KeyError
        raise ShipInternalError()

    def Set_C_WP_Method(self, method):
        self.__C_WP_method = method
        return

    def C_B_calc(self, method=None, speed=None):
        if method == None:
            method = self.__C_B_method
        if speed == None:
            speed = self.__default_speed
        if method[:4] == "Ayre":
            return (float(method[4:]) - 1.68 * hydro.F_n(speed, self.L)).value
        elif method[:11] == "Schneekluth":
            F_n = hydro.F_n(speed, self.L)
            if 0.14 > F_n > 0.32:
                raise RangeError("F_n")
            if F_n > 0.3:
                F_n = 0.3
            if method[-1:] == "1":
                C_B = (0.14 / F_n) * (((self.L / self.B) + 20) / 26.0)
            elif method[-1:] == "2":
                C_B = (0.23 / math.pow(F_n, (2.0 / 3.0))) * (
                    ((self.L / self.B) + 20) / 26.0
                )
            else:
                raise KeyError("unknown method")
            if C_B < 0.48:
                return 0.48
            elif C_B > 0.85:
                return 0.85
            else:
                return C_B
        else:
            raise KeyError("unknown method")
        raise ShipInternalError()

    def Set_C_B_method(self, method):
        self.__B_N_method = method
        return

    def Set_default_speed(self, speed):
        self.__default_speed = speed
        return

    def i_E_calc(self):
        return H.calc_i_E(self)

    def C_P_calc(self):
        return self.C_B / self.C_M

    def Nab_calc(self):
        return (self.L * self.B * self.T) * self.C_B

    def load(self, file=None):
        try:

            f = open(file, "r")
        except IOError:
            return
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("#")[0].strip()
            if len(line) > 0:
                res = line.split("=")
                key, value = res[0].strip(), res[1].strip()
                out[key] = eval(value)
                self.__keys.append(key)
        return

    def save(self, file=None):
        pass

    def TkEdit(self, master):

        self.dialog = Pmw.Dialog(
            master,
            buttons=("Save", "Cancel"),
            defaultbutton="Cancel",
            title="Edit ship's properties",
        )
        # master.withdraw()
        new_master = self.dialog.interior()
        self.dialog.grid()
        self.selbox = Pmw.ComboBox(
            new_master,
            label_text="select entity:",
            labelpos="w",
            selectioncommand=self.execute,
            scrolledlist_items=self.__keys,
        )
        self.selbox.grid()
        firstKey = self.__keys[0]
        self.selbox.selectitem(firstKey)
        # master.activate()
        return

    def execute(self, entity):
        text = "entity " + entity
        print(text)


class ShipError(Exception):
    pass


class ShipTypeError(TypeError):
    pass


class ShipInternalError(ShipError):
    pass


def checkQuantity(item, needed_unit):
    """
Check, whether the item has needed_unit.
"""
    if isinstance(needed_unit, (tuple, list)):
        if isinstance(needed_unit[0], (tuple, list)):
            if isinstance(item, (tuple, list)):
                for i in item:
                    if not isinstance(i, (tuple, list)):
                        raise ShipTypeError("Wrong argument Type (need list or tuple).")
                    elif len(i) != len(needed_unit[0]):
                        raise ShipTypeError(
                            "All elemnts of list must have same length."
                        )
                    checkQuantity([i], needed_unit[0])
            else:
                raise ShipTypeError("Wrong argument Type (need list or tuple).")
        else:
            if isinstance(item, (list, tuple)):
                for i in item:
                    if isinstance(i, (list, tuple)):
                        for check in zip(i, needed_unit):
                            checkQuantity(check[0], check[1])
                    else:
                        checkQuantity(i, needed_unit[0])
            else:
                raise ShipTypeError("Wrong argument Type (need list or tuple).")

    else:
        if needed_unit is None:
            if isinstance(item, Quantity) and not item.unit == astropy.units.core.Unit(
                ""
            ):
                raise ShipTypeError("Has no dimension")
        else:
            if item.unit == astropy.units.core.Unit(""):
                raise ShipTypeError("Has no dimension")
            if not item.unit == needed_unit:
                raise ShipTypeError("Need unit compatible to " + repr(needed_unit))


# Local Variables:
# mode: python
# compile-command: "python ./setup.py test"
# time-stamp-pattern: "30/__date__ = \"%:y/%02m/%02d %02H:%02M:%02S %u\""
# End:
