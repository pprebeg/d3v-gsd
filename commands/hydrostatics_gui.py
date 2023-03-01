from PySide6.QtWidgets import QApplication, QMenu, QFormLayout,QWidget,QHeaderView,QSlider,QLineEdit
from PySide6.QtWidgets import QDialog, QPushButton,QGridLayout,QVBoxLayout,QHBoxLayout,QTableView,QTextEdit,QLabel
from PySide6.QtCore import Slot,Qt,QObject
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect
from PySide6.QtGui import QColor, QPainter
#from PySide6.QtCharts import QtCharts
from PySide6 import QtCharts
from PySide6.QtWidgets import QInputDialog
import openmesh as om
import numpy as np
import os
#d3v imports
from signals import Signals
from commands import Command
from iohandlers import IOHandler
#d3v-gsd
from hullformdir.hullform import HullForm
from hullformdir.hydrostatics import Hydrostatics
#from hullformdir.resistance_h&m import Holtrop_and_Mennen_resistance_prediction_ver2

class HydrostaticsGUI():
    def __init__(self,hfc):
        self._hf_command = hfc #HullFormCommand
        self._app = QApplication.instance()
        self.hf_hydroCurves = DialogHullFormHydrostaticCurves(self.mainwin)

        self.menuMain = QMenu("Hydrostatics")
        mb = self.mainwin.menuBar()
        mb.addMenu(self.menuMain)

        menuResultHydrostaticCurves = self.menuMain.addAction("Hydrostatic Curves")
        menuResultHydrostaticCurves.triggered.connect(self.onShowHydrostaticCurves)
        menuResistancePrediction = self.menuMain.addAction("Calculate Hydrostatics")
        menuResistancePrediction.triggered.connect(self.onCalculateHydrostatics)

    @property
    def selected_hull_form(self):
        return self._hf_command.selected_hull_form

    @property
    def active_hull_form(self):
        return self._hf_command.active_hull_form

    @property
    def hull_forms(self):
        return self._hf_command.hull_forms


    def onShowHydrostaticCurves(self):
        if isinstance(self.active_hull_form, HullForm):
            self.hf_hydroCurves.setCurrentHullForm(self.active_hull_form)
            self.hf_hydroCurves.exec()

    def onCalculateHydrostatics(self):
        if isinstance(self.active_hull_form, HullForm):
            text, ok = QInputDialog.getText(self.mainwin, 'Input Dialog',
                                            'Input drought for calculation:')
            if ok:
                try:
                    main_deck_z = self.active_hull_form.bbox.maxCoord[2] - 0.01
                    wl_z = float(text)
                    if wl_z <= main_deck_z:
                        hscalc = Hydrostatics(self.active_hull_form)
                        results = hscalc.get_hydrostatic_results(wl_z)
                        result_names = ['h', 'Volume', 'Awl', 'Xwl', 'KBz', 'KBx', 'Ib', 'Il', 'Swet',
                                        'KMo', 'KMl', 'JZ', 'M1', 'delta', 'Cwl', 'CB', 'CP', 'CX']
                        print(result_names)
                        print(results)
                    else:
                        print('ERROR: Inputed drought greater than ship height. Calculation ommitted!')
                except:
                    print('ERROR: Inputed drought is not a number. Calculation ommitted!')


    @property
    def app(self):
        return self._hf_command.app

    @property
    def mainwin(self):
        return self._hf_command.mainwin

    @property
    def glwin(self):
        return self._hf_command.glwin

class CustomTableModel(QAbstractTableModel):
    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.input_data = []
        self.input_names = []
        self.mapping = {}
        self.column_count = 0
        self.row_count = 0


    def setInputData(self,input_names, input_data):
        self.beginResetModel()
        self.input_data=input_data
        self.input_names= input_names
        self.column_count = len(self.input_names)
        self.row_count = len(self.input_data)
        self.endResetModel()



    def rowCount(self, parent=QModelIndex()):
        return self.row_count

    def columnCount(self, parent=QModelIndex()):
        return self.column_count

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.input_names[section]
        else:
            return "{}".format(section + 1)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.input_data[index.row()][index.column()]
        elif role == Qt.EditRole:
            return self.input_data[index.row()][index.column()]
        elif role == Qt.BackgroundRole:
            for color, rect in self.mapping.items():
                if rect.contains(index.column(), index.row()):
                    return QColor(color)
            # cell not mapped return white color
            return QColor(Qt.white);
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and role == Qt.EditRole:
            self.input_data[index.row()][index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable

    def add_mapping(self, color, area):
        self.mapping[color] = area

    def clear_mapping(self):
        self.mapping = {}

class DialogHullFormHydrostaticCurves(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        sizetxt=25
        self.mainwin = parent


        self.model = CustomTableModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.chart = QtCharts.QChart()
        self.chart.setAnimationOptions(QtCharts.QChart.AllAnimations)


        self.chart_view = QtCharts.QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumSize(640, 480)


        self.setWindowTitle("Hull Form Hydrostatic Curves")
        self.btnGenerate = self.createButton("&Generate", self.refreshResults)

        self.txtMaxWL = QTextEdit()
        self.txtMaxWL.setFixedHeight(sizetxt)
        self.txtMaxWL.setText('9.0')
        self.txtMaxWL.setAlignment(Qt.AlignRight)
        self.txtWLStep = QTextEdit()
        self.txtWLStep.setFixedHeight(sizetxt)
        self.txtWLStep.setText('0.5')
        self.txtWLStep.setAlignment(Qt.AlignRight)


        mainLayout = QVBoxLayout()
        mainLayout.setStretch(0,1)
        mainLayout.setStretch(1, 0)
        tablechartLayout = QGridLayout()


        controlLayout = QHBoxLayout()
        controlWidget = QWidget()
        controlWidget.setFixedHeight(sizetxt*3)
        controlWidget.setLayout(controlLayout)

        inputLayout = QFormLayout()
        controlLayout.addLayout(inputLayout)
        controlLayout.addWidget(self.btnGenerate)

        inputLayout.addRow("&Max. Waterline height:", self.txtMaxWL)
        inputLayout.addRow("&Waterline step:", self.txtWLStep)

        tablechartLayout.addWidget(self.table_view, 0, 0)
        tablechartLayout.addWidget(self.chart_view, 0, 1)
        mainLayout.addLayout(tablechartLayout)
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(controlWidget)

        self.setLayout(mainLayout)
        self.hscalc = None


    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def refreshResults(self):
        input_data = []
        mjerilo = [1,1/95,1/45,1,1/0.2,1,1/480,1/12220,1/0.30,1/15,1/2,1/200,1/90,1/0.008,1/0.0055,1/0.007,1/0.008]
        maxWL= float(self.txtMaxWL.toPlainText())
        stepWL = float(self.txtWLStep.toPlainText())
        h=maxWL
        while h > 0:
            result = self.hscalc.get_results_for_hydrostatic_curves(h)
            input_data.append([a*b for a,b in zip(result,mjerilo)])
            h=h-stepWL
            if h <= 0:
                result = self.hscalc.get_results_for_hydrostatic_curves(1)
                input_data.append([a*b for a,b in zip(result,mjerilo)])


        input_names = ['h', 'Volume', 'Awl', 'Xwl', 'KBz', 'KBx', 'Ib', 'Il',
                       'KMo','KMl','JZ', 'M1','delta','Cwl','CB','CP','CX']
        colors = ['aqua','maroon','blue','lime','magenta','crimson','blueviolet','orange','orchid','forestgreen','salmon','gold','slategrey','skyblue','greenyellow','moccasin']
        #input_names = ['h', 'Volume', 'Awl']
#        self.model.layoutAboutToBeChanged()
        self.model.setInputData(input_names, input_data)
        #self.model.layoutChanged()
        self.chart.removeAllSeries()
        seriesColorHex = "#000000"
        for i in range(1, len(input_names)):
            series = QtCharts.QLineSeries()
            series.setColor(QColor(str(colors[i-1])))
            series.setName(input_names[i])
            mapper = QtCharts.QVXYModelMapper(self)
            mapper.setYColumn(0)
            mapper.setXColumn(i)
            mapper.setSeries(series)
            mapper.setModel(self.model)
            self.chart.addSeries(series)
            # get the color of the series and use it for showing the mapped area
            seriesColorHex = "{}".format(series.pen().color().name())
            self.model.add_mapping(seriesColorHex, QRect(i, 0, 1, self.model.rowCount()))

        self.chart.createDefaultAxes()


    def setCurrentHullForm(self, currentHullForm):
        self.hscalc = Hydrostatics(currentHullForm)
        self.setWindowTitle("Hull Form Hydrostatic Curves")