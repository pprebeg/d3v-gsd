from PySide2.QtWidgets import QApplication, QMenu, QFormLayout,QWidget,QHeaderView,QSlider,QLineEdit
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout,QVBoxLayout,QHBoxLayout,QTableView,QTextEdit,QLabel
from PySide2.QtWidgets import QTreeView,QMainWindow
from PySide2.QtCore import Slot,Qt
from PySide2.QtCore import QAbstractTableModel, QModelIndex, QRect
from PySide2.QtGui import QColor, QPainter
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import SIGNAL,SLOT
from scipy.optimize import minimize
import openmesh as om
import numpy as np
import os
#d3v imports
from signals import Signals
from commands import Command
from iohandlers import IOHandler
#d3v-gsd
from hullformdir.hullform import HullForm

class HullFormCommand(Command):
    def __init__(self):
        super().__init__()
        app = QApplication.instance()
        app.registerIOHandler(HullFormImporter())
        self.mainwin = app.mainFrame
        self._tree: QTreeView = self.mainwin.window.findChild(QTreeView, "geometryTree")
        self._tree.hide()
        self.hf_prop = DialogHullFormModify(self.mainwin)
        self.hf_opt = DialogHullFormOptimize(self.mainwin)
        self.hf_hydroCurves = DialogHullFormHydrostaticCurves(self.mainwin)
        self.hf=0
        self.si=0

        #tools = app.mainFrame.menuTools
        mb = app.mainFrame.menuBar()

        self.menuMain = QMenu("Hull Form")

        self.menuHullFormModify = self.menuMain.addAction("&Modify form")
        self.menuHullFormModify.triggered.connect(self.onModifyForm)

        self.menuHullFormOptimize = self.menuMain.addAction("&Optimize form")
        self.menuHullFormOptimize.triggered.connect(self.onOptimizeForm)

        self.menuHullFormResults = QMenu("&Results")
        self.menu2 = QMenu("&Menu2")
        self.menuMain.addMenu(self.menuHullFormResults)
        self.menuMain.addMenu(self.menu2)



        menuResultHydrostaticCurves = self.menuHullFormResults.addAction("Hydrostatic Curves")
        menuResultHydrostaticCurves.triggered.connect(self.onShowHydrostaticCurves)



        #tools.addMenu(menu)
        mb.addMenu(self.menuMain)
        #self._menuMain.setEnabled(False)


        Signals.get().geometryAdded.connect(self.registerHullForm)
        Signals.get().selectionChanged.connect(self.registerSelection)

    @Slot()
    def registerHullForm(self, hullForm):
        if isinstance(hullForm, HullForm):
            self.hf=hullForm
            self.menuMain.setEnabled(True)
            #om.write_mesh("D:\hullForm.obj", hullForm.mesh)

    @Slot()
    def registerSelection(self, si):
        self.si=si

    def onModifyForm(self):
        if isinstance(self.hf, HullForm):
            self.hf_prop.setCurrentHullForm(self.hf)
            self.hf_prop.exec()
    def onOptimizeForm(self):
        if isinstance(self.hf, HullForm):
            self.hf_opt.setCurrentHullForm(self.hf)
            self.hf_opt.exec()

    def onShowHydrostaticCurves(self):
        if isinstance(self.hf,HullForm):
            self.hf_hydroCurves.setCurrentHullForm(self.hf)
            self.hf_hydroCurves.exec()



class HullFormImporter(IOHandler):
    def __init__(self):
        super().__init__()

    def importGeometry(self, fileName):
        if len(fileName) < 1:
            return
        filename, file_extension = os.path.splitext(fileName)
        if file_extension != ".huf":
            return
        hf = HullForm(fileName)
        Signals.get().geometryImported.emit(hf)

    def exportGeometry(self, fileName, geometry2export):
        om.write_mesh(geometry2export.mesh,fileName)
        pass

    def getExportFormats(self):
       return (".obj")

    def getImportFormats(self):
        return (".huf")

class QTextSliderConnect(QLineEdit):
    def __init__(self, parent,slider):
        super().__init__(parent)
        self.slider=slider
        Signals.connect(self, SIGNAL('editingFinished()'), self, SLOT('updateSlider()'))
        self.min =0
        self.max=1
        self.value = 0
        self.isTextChangeInProgress=False

    def setminmaxval(self,min,max,val):
        self.min=min
        self.max=max
        self.value=val
        self.setText(str(self.value))
        self.updateSlider()

    def updatetxt(self,value):
        if not self.isTextChangeInProgress:
            self.value=self.min+(self.max-self.min)*(value-self.slider.minimum())/(self.slider.maximum()-self.slider.minimum())
            self.setText(str(self.value))


    def updateSlider(self):
        try:
            self.value=float(self.text())
            if self.value < self.min:
                self.value = self.min
                self.setText(str(self.value))
            elif self.value > self.max:
                self.value = self.max
                self.setText(str(self.value))
            value=self.value
            value = int(round(self.slider.minimum()+ (self.slider.maximum()-self.slider.minimum())*(value-self.slider.minimum())/(self.max-self.min)))
            self.isTextChangeInProgress = True
            self.slider.setValue(value)
            self.isTextChangeInProgress = False
        except ValueError:
            self.setText(str(self.value))
            pass


class QSliderTextConnect(QSlider):
    def __init__(self, parent):
        super().__init__(parent)


class DialogHullFormModify(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.sizerow = 25
        self.mainwin = parent

        self.mainLayout = QGridLayout()

        self.setLayout( self.mainLayout)
        self.currentHullForm=0
        self.shipdatanames=0
        self.shipdatamins =0
        self.shipdatamaxs =0
        self.shipdatatxt = {}


    def initDicts(self):
        self.shipdatanames = {"loa_val":"Overall Length (LOA), m",
                        "boa_val":"Overall Beam (BOA), m",
                        'draft_val':"Draft (T)",
                        "ms_val":"Midship Fullness",
                        "bow_val":"Bow Fullness",
                        "tr_val":"Transom Fullness",
                        "deck_val":"Forward Deck Fullness",
                        "tb_val":"Transom Beam",
                        "td_val":"Transom Draught",
                        "acu_val":"ACU",
                        "kf_val":"Forward Keel",
                        "sa_val":"Superstructure Angle, rad"}
        self.shipdatamins = {"loa_val":0,
                        "boa_val":0,
                        'draft_val':0,
                        "ms_val":0.1,
                        "bow_val":0.1,
                        "tr_val":0.1,
                        "deck_val":0.1,
                        "tb_val":0.01,
                        "td_val":0.01,
                        "acu_val":0.01,
                        "kf_val":0.51,
                        "sa_val":0}
        self.shipdatamaxs  = {"loa_val": 300,
                        "boa_val": 80,
                        'draft_val': 10,
                        "ms_val": 10,
                        "bow_val": 10,
                        "tr_val": 10,
                        "deck_val": 10,
                        "tb_val": 0.99,
                        "td_val": 0.99,
                        "acu_val": 0.49,
                        "kf_val": 0.99,
                        "sa_val": 0.8}
    def createUserInputs(self,gridLayout:QGridLayout):
        i=0
        for key,value in self.shipdatanames.items():
            lbl = QLabel(value)
            lbl.setFixedHeight(self.sizerow)
            val = self.currentHullForm.shipdata[key]
            slider = QSliderTextConnect(Qt.Horizontal)
            slider.setFixedHeight(self.sizerow)
            txt = QTextSliderConnect(self,slider)
            self.shipdatatxt[key]=txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(50)
            slider.valueChanged.connect(txt.updatetxt)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            txt.setminmaxval(self.shipdatamins[key],self.shipdatamaxs[key],val)
            gridLayout.addWidget(lbl, i, 0)
            gridLayout.addWidget(txt, i, 1)
            gridLayout.addWidget(slider, i, 2)
            i=i+1
        return i



    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def regenerateHullFormMesh(self):
        for key,txt in self.shipdatatxt.items():
            val = float(txt.text())
            self.currentHullForm.shipdata[key]=val
        self.currentHullForm.generateMesh()
        Signals.get().geometryRebuild.emit(self.currentHullForm)

    def setCurrentHullForm(self, currentHullForm):
        self.currentHullForm = currentHullForm
        self.setWindowTitle("Modify Hull Form")
        self.initDicts()
        irow = self.createUserInputs(self.mainLayout)
        btnModify = self.createButton("&Modify", self.regenerateHullFormMesh)
        btnModify.setFocusPolicy(Qt.NoFocus)
        self.mainLayout.addWidget(btnModify, irow, 0)

class DialogHullFormOptimize(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.sizerow = 25
        self.mainwin = parent

        self.mainLayout = QGridLayout()

        self.setLayout( self.mainLayout)
        self.currentHullForm=0
        self.shipdatavarnames=0
        self.shipdatamins =0
        self.shipdatamaxs =0
        self.shipdatatxt = {}
        self.shipdatatxtmin = {}
        self.shipdatatxtmax = {}
        self.shipcriterianames = {}
        self.shipcriteriaindexes = {}
        self.shipcriteriavalues = {}
        self.xkeys = []
        self.xlast = []
    def isSame(self,x):
        result= True
        for i in range(len(x)):
            if x[i] != self.xlast[i]:
                self.xlast[i]=x[i]
                result = False
        return result

    def showEvent(self, arg__1):
        self.refreshControls(True)

    def initDicts(self):
        self.shipdatavarnames = {"loa_val": "Overall Length (LOA), m",
                        "boa_val":"Overall Beam (BOA), m",
                        'draft_val':"Draft (T)",
                        "ms_val":"Midship Fullness",
                        "bow_val":"Bow Fullness",
                        "tr_val":"Transom Fullness",
                        "deck_val":"Forward Deck Fullness",
                        "tb_val":"Transom Beam",
                        "td_val":"Transom Draught",
                        "acu_val":"ACU",
                        "kf_val":"Forward Keel",
                                 "sa_val":"Superstructure Angle, rad"}
        self.shipdatamins = {"loa_val":0,
                        "boa_val":0,
                        'draft_val':0,
                        "ms_val":0.1,
                        "bow_val":0.1,
                        "tr_val":0.1,
                        "deck_val":0.1,
                        "tb_val":0.01,
                        "td_val":0.01,
                        "acu_val":0.01,
                        "kf_val":0.51,
                        "sa_val":0}
        self.shipdatamaxs  = {"loa_val": 300,
                        "boa_val": 80,
                        'draft_val': 10,
                        "ms_val": 10,
                        "bow_val": 10,
                        "tr_val": 10,
                        "deck_val": 10,
                        "tb_val": 0.99,
                        "td_val": 0.99,
                        "acu_val": 0.49,
                        "kf_val": 0.99,
                        "sa_val": 0.8}
        self.shipcriterianames = {"cb_val": "Fullness coefficient (CB)",
                              "cp_val": "Prismatic coefficient (CP)",
                              'vol_val': "Volume (V), m3"}
        # indexes in output results list
        self.shipcriteriaindexes = {"cb_val": 14,
                                "cp_val": 15,
                                'vol_val': 1}
        for key in self.shipcriterianames.keys():
            self.shipdatamins[key] = 0.0
            self.shipdatamaxs[key] = 0.0
            self.shipcriteriavalues[key] = 0.0
    def refreshControls(self,doCalc=False):
        for key in self.shipdatavarnames.keys():
            # start value
            self.shipdatatxt[key].setText(str(np.round(self.currentHullForm.shipdata[key],4)))
            #lower bound
            self.shipdatatxtmin[key].setText(str(self.shipdatamins[key]))
            #upper bound
            self.shipdatatxtmax[key].setText(str(self.shipdatamaxs[key]))
        if doCalc:
            self.calcOutputs()
        for key in self.shipcriterianames.keys():
            # start value
            self.shipdatatxt[key].setText(str(np.round(self.shipcriteriavalues[key],4)))
            #lower bound
            self.shipdatatxtmin[key].setText(str(self.shipdatamins[key]))
            #upper bound
            self.shipdatatxtmax[key].setText(str(self.shipdatamaxs[key]))
    def createUserInputs(self,gridLayout:QGridLayout):
        txtWidth=80
        i=0
        lbl = QLabel("Variables")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 0)
        lbl = QLabel("Value")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 1)
        lbl = QLabel("Min")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 2)
        lbl = QLabel("Max")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 3)
        i+=1
        for key,value in self.shipdatavarnames.items():
            #label
            lbl = QLabel(value)
            lbl.setFixedHeight(self.sizerow)
            gridLayout.addWidget(lbl, i, 0)
            # txt cur value
            val = self.currentHullForm.shipdata[key]
            txt = QLineEdit(self)
            self.shipdatatxt[key]=txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(val))
            gridLayout.addWidget(txt, i, 1)
            # txt min value
            txt = QLineEdit(self)
            self.shipdatatxtmin[key] = txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(self.shipdatamins[key]))
            gridLayout.addWidget(txt, i, 2)
            # txt max value
            txt = QLineEdit(self)
            self.shipdatatxtmax[key] = txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(self.shipdatamaxs[key]))
            gridLayout.addWidget(txt, i, 3)
            i=i+1
        # Outputs
        lbl = QLabel("Outputs")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 0)
        lbl = QLabel("Value")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 1)
        lbl = QLabel("T.Min")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 2)
        lbl = QLabel("T.Max")
        lbl.setFixedHeight(self.sizerow)
        gridLayout.addWidget(lbl, i, 3)
        i+=1
        for key,value in self.shipcriterianames.items():
            # label
            lbl = QLabel(value)
            lbl.setFixedHeight(self.sizerow)
            gridLayout.addWidget(lbl, i, 0)
            # txt cur value
            val = self.shipcriteriavalues[key]
            txt = QLineEdit(self)
            self.shipdatatxt[key] = txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(val))
            gridLayout.addWidget(txt, i, 1)
            # txt min value
            txt = QLineEdit(self)
            self.shipdatatxtmin[key] = txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(self.shipdatamins[key]))
            gridLayout.addWidget(txt, i, 2)
            # txt max value
            txt = QLineEdit(self)
            self.shipdatatxtmax[key] = txt
            txt.setAlignment(Qt.AlignRight)
            txt.setFixedHeight(self.sizerow)
            txt.setFixedWidth(txtWidth)
            txt.setText(str(self.shipdatamaxs[key]))
            gridLayout.addWidget(txt, i, 3)
            i += 1
        return i



    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def regenerateHullFormMesh(self):
        for key,txt in self.shipdatatxt.items():
            val = float(txt.text())
            self.currentHullForm.shipdata[key]=val
        self.currentHullForm.generateMesh()
        Signals.get().geometryRebuild.emit(self.currentHullForm)

    def optCalc(self,x):
        if not self.isSame(x):
            self.calcOutputs(x)
    def fun_cb(self,x):
        self.optCalc(x)
        return self.shipcriteriavalues['cb_val']
    def fun_cp_conmin(self,x):
        self.optCalc(x)
        return self.shipcriteriavalues['cp_val']-self.shipdatamins['cp_val']
    def fun_vol_conmin(self,x):
        self.optCalc(x)
        return self.shipcriteriavalues['vol_val']-self.shipdatamins['vol_val']
    def fun_vol_conmax(self,x):
        self.optCalc(x)
        return -self.shipcriteriavalues['vol_val']+self.shipdatamaxs['vol_val']

    def optimizeHullFormMesh(self):
        #prepare for opt
        x=[]

        for key in self.shipdatavarnames.keys():
            # start value
            txt=self.shipdatatxt[key]
            val = float(txt.text())
            self.xkeys.append(key)
            #x.append(val)
            #lower boun
            txtmin = self.shipdatatxtmin[key]
            lb = float(txtmin.text())
            self.shipdatamins[key] = lb
            #upper boun
            txtmax = self.shipdatatxtmax[key]
            ub = float(txtmax.text())
            self.shipdatamaxs[key] = ub
            x.append(val+(ub-val)*0.1) # for test
        #optimize
        self.xlast = [0.0]*len(x)
        bnds = []
        for i in range(len(x)):
            key = self.xkeys[i]
            bnd=(self.shipdatamins[key],self.shipdatamaxs[key])
            bnds.append(bnd)
        con1 = {'type': 'ineq', 'fun': self.fun_cp_conmin}
        con2 = {'type': 'ineq', 'fun': self.fun_vol_conmin}
        cons = [con1, con2]
        sol1 = minimize(self.fun_cb, x, constraints=cons, bounds=bnds, method='SLSQP')
        self.calcOutputs(x)

        #show optimal hulform
        # use optimal x now, not the latests (it could be different for some optimizers)
        self.calcOutputs(x)
        self.refreshControls()
        Signals.get().geometryRebuild.emit(self.currentHullForm)

    def calcOutputs(self,x=None):
        if x is not None:
            for i in range(len(x)):
                key=self.xkeys[i]
                self.currentHullForm.shipdata[key] = x[i]

        h = self.currentHullForm.shipdata['draft_val']
        self.currentHullForm.generateMesh()
        # 0   1    2    3    4    5    6   7   8    9    10 11    12    13  14  15  16
        # h, vol, Awl, Xwl, KBz, KBx, Ib, Il, KMo, KMl, JZ, M1, delta, Cwl, CB, CP, CX
        results = self.currentHullForm.getResults(h, 1.025)
        for key,index in self.shipcriteriaindexes.items():
            self.shipcriteriavalues[key]=results[index]

    def setCurrentHullForm(self, currentHullForm):
        self.currentHullForm = currentHullForm
        self.setWindowTitle("Optimize Hull Form")
        self.initDicts()
        irow = self.createUserInputs(self.mainLayout)
        btnModify = self.createButton("&Modify", self.regenerateHullFormMesh)
        btnModify.setFocusPolicy(Qt.NoFocus)
        self.mainLayout.addWidget(btnModify, irow, 0)
        btnOptimize = self.createButton("&Optimize", self.optimizeHullFormMesh)
        btnOptimize.setFocusPolicy(Qt.NoFocus)
        self.mainLayout.addWidget(btnOptimize, irow, 1)


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
        self.currentHullForm=0


    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def refreshResults(self):
        #self.currfemmdl.getResults(9, 1.025)
        #return
        input_data = []
        mjerilo = [1,1/95,1/45,1,1/0.2,1,1/480,1/12220,1/0.30,1/15,1/2,1/200,1/90,1/0.008,1/0.0055,1/0.007,1/0.008]
        maxWL= float(self.txtMaxWL.toPlainText())
        stepWL = float(self.txtWLStep.toPlainText())
        h=maxWL
        while h > 0:
            result = self.currentHullForm.getResults(h, 1.025)
            input_data.append([a*b for a,b in zip(result,mjerilo)])
            h=h-stepWL
            if h <= 0:
                result = self.currentHullForm.getResults(1, 1.025)
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
        self.currentHullForm = currentHullForm
        self.setWindowTitle("Hull Form Hydrostatic Curves")



def createCommand():
    return HullFormCommand()