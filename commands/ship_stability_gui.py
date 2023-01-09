from PySide6.QtWidgets import QApplication, QMenu, QFormLayout,QWidget,QHeaderView,QSlider,QLineEdit
from PySide6.QtWidgets import QDialog, QPushButton,QGridLayout,QVBoxLayout,QHBoxLayout,QTableView,QTextEdit,QLabel
from PySide6.QtCore import Slot,Qt,QObject
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect
from PySide6.QtGui import QColor, QPainter
#from PySide6.QtCharts import QtCharts
from PySide6 import QtCharts
from PySide6.QtCore import SIGNAL,SLOT
import openmesh as om
import numpy as np
import os
#d3v imports
from signals import Signals
from commands import Command
from iohandlers import IOHandler
#d3v-gsd
from hullformdir.hullform import HullForm
from hullformdir.shipstability import ShipStability
#from hullformdir.resistance_h&m import Holtrop_and_Mennen_resistance_prediction_ver2

class ShipStabilityGUI():
    def __init__(self,hfc):
        self._hf_command = hfc #HullFormCommand
        self._app = QApplication.instance()

        self.menuMain = QMenu("Ship &Stability")
        mb = self.mainwin.menuBar()
        mb.addMenu(self.menuMain)

        menu_result_displacement = self.menuMain.addAction("Calculate Displacement")
        menu_result_displacement.triggered.connect(self.on_calculate_displacements)
        menu_result_drought = self.menuMain.addAction("Calculate Drought")
        menu_result_drought.triggered.connect(self.on_calculate_drought)
        menu_result_drought = self.menuMain.addAction("Calculate Trim")
        menu_result_drought.triggered.connect(self.on_calculate_trim)

    @property
    def selected_hull_form(self):
        return self._hf_command.selected_hull_form

    @property
    def active_hull_form(self):
        return self._hf_command.active_hull_form

    @property
    def hull_forms(self):
        return self._hf_command.hull_forms


    def on_calculate_displacements(self):
        if isinstance(self.active_hull_form, HullForm):
            sscalc = ShipStability(self.active_hull_form)
            sscalc.calculate_displacement_and_displacementCG()


    def on_calculate_trim(self):
        if isinstance(self.active_hull_form, HullForm):
            pass

    def on_calculate_drought(self):
        if isinstance(self.active_hull_form, HullForm):
            sscalc = ShipStability(self.active_hull_form)
            sscalc.calculate_drought_horizontal_waterline()


    @property
    def app(self):
        return self._hf_command.app

    @property
    def mainwin(self):
        return self._hf_command.mainwin

    @property
    def glwin(self):
        return self._hf_command.glwin
