from PySide6.QtWidgets import QApplication, QMenu, QFormLayout,QWidget,QHeaderView,QSlider,QLineEdit
from PySide6.QtWidgets import QDialog, QPushButton,QGridLayout,QVBoxLayout,QHBoxLayout,QTableView,QTextEdit,QLabel
from PySide6.QtCore import Slot,Qt,QObject
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect
from PySide6.QtGui import QColor, QPainter,QColorConstants
from PySide6.QtCharts import QChartView,QChart,QSplineSeries,QValueAxis,QLineSeries
from PySide6.QtWidgets import QTableWidget,QTableWidgetItem,QSplitter,QInputDialog,QDialogButtonBox,QMessageBox
from PySide6.QtCore import SIGNAL,SLOT
from interactive_chart_widget import InteractiveTableChartWidget
import openmesh as om
import numpy as np
import os
#d3v imports
from signals import Signals
from commands import Command
from iohandlers import IOHandler
#d3v-gsd
from hullformdir.hullform import HullForm
from hullformdir.shipstability import ShipStability,LoadCondition,Waterline
#from hullformdir.resistance_h&m import Holtrop_and_Mennen_resistance_prediction_ver2

class ShipStabilityGUI():
    def __init__(self,hfc):
        self._hf_command = hfc #HullFormCommand
        self._app = QApplication.instance()

        self.menuMain = QMenu("Ship &Stability")
        mb = self.mainwin.menuBar()
        mb.addMenu(self.menuMain)

        menu_result = self.menuMain.addAction("Calculate Displacement")
        menu_result.triggered.connect(self.on_calculate_displacements)
        menu_result = self.menuMain.addAction("Set load condition")
        menu_result.triggered.connect(self.on_set_load_condition)
        menu_result = self.menuMain.addAction("Calculate Drought")
        menu_result.triggered.connect(self.on_calculate_drought)
        menu_result = self.menuMain.addAction("Calculate Trim")
        menu_result.triggered.connect(self.on_calculate_trim)
        menu_result = self.menuMain.addAction("Curve of Static Stability")
        menu_result.triggered.connect(self.on_calculate_css)
        self._current_load_condition:LoadCondition = None

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
            main_deck_z =self.active_hull_form.bbox.maxCoord[2]-0.01
            text, ok = QInputDialog.getText(self.mainwin, 'Input Dialog',
                                                  'Input drought for calculation:')
            if ok:
                try:
                    wl_z = float(text)
                    if wl_z <= main_deck_z:
                        sscalc = ShipStability(self.active_hull_form,main_deck_z)
                        sscalc.wl.set_plane_point_z(wl_z)
                        sscalc.calculate_displacement_and_displacementCG_example()
                    else:
                        print('ERROR: Inputed drought greater than ship height. Calculation ommitted!')
                except:
                    print('ERROR: Inputed drought is not a number. Calculation ommitted!')



    def on_calculate_drought(self):
        if isinstance(self.active_hull_form, HullForm):
            main_deck_z = self.active_hull_form.bbox.maxCoord[2] - 0.01
            sscalc = ShipStability(self.active_hull_form, main_deck_z)
            z_mid, z_max = self.active_hull_form.get_z_mid_z_max_from_mesh()
            # sscalc.wl.set_plane_point_x(self.active_hull_form.xmf)
            sscalc.wl.set_plane_point_x(50.0)
            sscalc.wl.set_plane_point_z(z_mid)
            if self._current_load_condition is not None:
                #sscalc.set_ship_G_and_CG(3160.0, 50.0, 0.0, 5.5)
                lc= self._current_load_condition
                sscalc.set_ship_G_and_CG(lc.mass, lc.CG[0], lc.CG[1], lc.CG[2])
                sscalc.determine_horizontal_waterline_for_current_weight()
            else:
                QMessageBox.warning(self.mainwin, "Warning", "Load condition not present, calculation omitted!")

    def on_calculate_trim(self):
        if isinstance(self.active_hull_form, HullForm):
            main_deck_z = self.active_hull_form.bbox.maxCoord[2]-0.01
            sscalc = ShipStability(self.active_hull_form, main_deck_z)
            z_mid, z_max = self.active_hull_form.get_z_mid_z_max_from_mesh()
            #sscalc.wl.set_plane_point_x(self.active_hull_form.xmf)
            sscalc.wl.set_plane_point_x(50.0)
            sscalc.wl.set_plane_point_z(z_mid)

            if self._current_load_condition is not None:
                #sscalc.set_ship_G_and_CG(3160.0, 50.0, 0.0, 5.5)
                lc= self._current_load_condition
                sscalc.set_ship_G_and_CG(lc.mass, lc.CG[0], lc.CG[1], lc.CG[2])
                sscalc.determine_trim_for_current_load_condition()
            else:
                QMessageBox.warning(self.mainwin, "Warning", "Load condition not present, calculation omitted!")

    def on_calculate_css(self):
        if isinstance(self.active_hull_form, HullForm):
            main_deck_z = self.active_hull_form.bbox.maxCoord[2]-0.001
            sscalc = ShipStability(self.active_hull_form, main_deck_z)
            if self._current_load_condition is not None:
                #sscalc.set_ship_G_and_CG(3160.0, 50.0, 0.0, 5.5)
                # sscalc.set_ship_G_and_CG(2000.0, 52.0, 0.0, 6.0)
                lc= self._current_load_condition
                sscalc.set_ship_G_and_CG(lc.mass, lc.CG[0], lc.CG[1], lc.CG[2])
                widget_css = StaticStabilityDiagramWidget(self.mainwin, sscalc)
                widget_css.setWindowFlag(Qt.WindowType.Window)
                widget_css.show()
            else:
                QMessageBox.warning(self.mainwin, "Warning", "Load condition not present, calculation omitted!")

    def on_set_load_condition(self):
        if isinstance(self.active_hull_form, HullForm):
            mass = 0.0
            CG = [0.0,0.0]
            if self._current_load_condition is not None:
                mass = self._current_load_condition.mass
                CG[0]= self._current_load_condition.CG[0]
                CG[1] = self._current_load_condition.CG[2]
            dlg_lc = ShipLoadConditionWidget(self.mainwin,mass,CG)
            dlgresult = dlg_lc.exec_()
            if dlgresult:
                mass,CG = dlg_lc.getDataFromDialog()
                self._current_load_condition = LoadCondition(mass,np.array([CG[0],0.0,CG[1]]))


    @property
    def app(self):
        return self._hf_command.app

    @property
    def mainwin(self):
        return self._hf_command.mainwin

    @property
    def glwin(self):
        return self._hf_command.glwin

class StaticStabilityDiagramWidget(QWidget):
    def __init__(self, parent,sscalc):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.sscalc = sscalc
        sizetxt=25
        self.mainwin = parent
        #Interactive Chart
        self.chart = InteractiveTableChartWidget()
        #self.chart.setTitle('Static stability curve plot')

        self.setWindowTitle("Static stability curve")
        self.btnGenerate = self.createButton("&Generate", self.refreshResults)

        self.txtMaxHeelAngle = QLineEdit()
        self.txtMaxHeelAngle.setFixedHeight(sizetxt)
        self.txtMaxHeelAngle.setText('90')
        self.txtMaxHeelAngle.setAlignment(Qt.AlignRight)
        self.txtHeelAngleStep = QLineEdit()
        self.txtHeelAngleStep.setFixedHeight(sizetxt)
        self.txtHeelAngleStep.setText('5')
        self.txtHeelAngleStep.setAlignment(Qt.AlignRight)


        mainLayout = QVBoxLayout()
        mainLayout.setStretch(0,1)
        mainLayout.setStretch(1, 0)

        controlLayout = QHBoxLayout()
        controlWidget = QWidget()
        controlWidget.setFixedHeight(sizetxt*3)
        controlWidget.setLayout(controlLayout)

        inputLayout = QFormLayout()
        controlLayout.addLayout(inputLayout)
        controlLayout.addWidget(self.btnGenerate)

        inputLayout.addRow("&Max. heeling angle:", self.txtMaxHeelAngle)
        inputLayout.addRow("&Heel angle step:", self.txtHeelAngleStep)

        mainLayout.addWidget(self.chart)
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(controlWidget)

        self.setLayout(mainLayout)



    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def refreshResults(self):
        max_heel_angle= float(self.txtMaxHeelAngle.text())
        step_heel_angle = float(self.txtHeelAngleStep.text())
        data =  self.sscalc.generate_static_stability_curve(max_heel_angle,step_heel_angle)
        n_rows = len(data)
        # chart
        n_cols=len(data[0])
        data_names = ['Heel,°', 'GZ, m','Trim,°', 'Displacement, t','Drought, m','Calc Heel,°']
        data_formats = ['{:.1f}', '{:.3f}', '{:.3f}', '{:.1f}', '{:.3f}', '{:.3f}']
        chart_data_pairs = [0, 1]
        self.chart.set_data(n_cols, n_rows, data_names, data_formats, data, chart_data_pairs, 'spline')

class ShipLoadConditionWidget(QDialog):
    def __init__(self, parent,mass=0.0,CG=[0.0,0.0]):
        super().__init__(parent)
        self.mainwin = parent
        sizetxt=25
        self.setWindowTitle("Enter Ship Load Condition")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.txtShipMass = QLineEdit()
        self.txtShipMass.setFixedHeight(sizetxt)
        self.txtShipMass.setText(str(mass))
        self.txtShipMass.setAlignment(Qt.AlignRight)
        self.txtCG = []
        for i in range(2):
            txtCG = QLineEdit()
            txtCG.setFixedHeight(sizetxt)
            txtCG.setText(str(CG[i]))
            txtCG.setAlignment(Qt.AlignRight)
            self.txtCG.append(txtCG)

        mainLayout = QVBoxLayout()

        inputLayout = QFormLayout()
        mainLayout.addLayout(inputLayout)

        inputLayout.addRow("&Ship mass :", self.txtShipMass)
        inputLayout.addRow("&LCG:", self.txtCG[0])
        inputLayout.addRow("&VCG:", self.txtCG[1])

        mainLayout.addLayout(inputLayout)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)

    def getDataFromDialog(self):
        mass= float(self.txtShipMass.text())
        CG = []
        for i in range(len(self.txtCG)):
            CG.append(float(self.txtCG[i].text()))
        return mass,CG
