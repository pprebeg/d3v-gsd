import os.path

from PySide6.QtWidgets import QApplication, QMenu, QFormLayout,QWidget,QHeaderView,QSlider,QLineEdit
from PySide6.QtWidgets import QDialog, QPushButton,QGridLayout,QVBoxLayout,QHBoxLayout,QTableView,QTextEdit,QLabel
from PySide6.QtCore import Slot,Qt,QMetaObject,QCoreApplication
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect
from PySide6.QtGui import QColor, QPainter
#from PySide6.QtCharts import QtCharts
from PySide6.QtWidgets import QFileDialog,QDialogButtonBox,QProgressDialog
from PySide6.QtCore import SIGNAL,SLOT
from typing import Dict,List
from uuid import uuid4


try:
    # d3v imports
    from signals import Signals
    from commands import Command
    from iohandlers import IOHandler
    from core import geometry_manager as manager
    from core.geometry import Geometry
    # d3v-gsd
    from hullformdir.hullform import *
    from hullformdir.hullgeneratorform import HullGeneratorForm
    from hydrostatics_gui import HydrostaticsGUI
except BaseException as error:
    print('An exception occurred: {}'.format(error))
except:
    print('Unknown exception occurred during signals connection')

class HullFormCommand(Command):
    def __init__(self):
        super().__init__()
        self._app = QApplication.instance()
        self._app.registerIOHandler(HullFormImporter())

        self.hf_prop = DialogHullGeneratorFormModify(self.mainwin)
        self.hf_opt = DialogHullGeneratorFormOptimize(self.mainwin)
        self.hull_forms:Dict[uuid4,HullForm]={}
        self.selected_hull_form=None
        self.active_hull_form = None
        self.referent_hull_form=None

        mb = self.mainwin.menuBar()
        self.menuMain = QMenu("Hull Form")
        mb.addMenu(self.menuMain)

        self.menuHullGenForm = QMenu("&Hull Generator Form")
        self.menuMain.addMenu(self.menuHullGenForm)

        menuHullGenFormModify = self.menuHullGenForm.addAction("&Modify form")
        menuHullGenFormModify.triggered.connect(self.onModifyHullGeneratorForm)

        menuHullGenFormOptimize = self.menuHullGenForm.addAction("&Optimize form")
        menuHullGenFormOptimize.triggered.connect(self.onOptimizeHullGeneratorForm)

        self.menuSelected = QMenu("&Selected")
        self.menuMain.addMenu(self.menuSelected)
        menuSetSelectedActive = self.menuSelected.addAction("Set &Active")
        menuSetSelectedActive.triggered.connect(self.onSetActiveHullForm)
        menuSetSelectedReferent = self.menuSelected.addAction("Set &Referent")
        menuSetSelectedReferent.triggered.connect(self.onSetReferentHullForm)

        menuMove = self.menuMain.addAction("&Move")
        menuMove.triggered.connect(self.onMoveHullForm)

        self.menuMain.addSeparator()

        menuImport = self.menuMain.addAction("&Import")
        menuImport.triggered.connect(self.onImportHullForm)
        menuExport = self.menuMain.addAction("&Export")
        menuExport.triggered.connect(self.onExportHullForm)


        try:
            manager.selected_geometry_changed.connect(self.onSelectedGeometryChanged)
            manager.geometry_created.connect(self.onGeometryCreated)
            manager.geometry_removed.connect(self.onGeometryRemoved)
            manager.visible_geometry_changed.connect(self.onVisibleGeometryChanged)
        except BaseException as error:
            print('An exception occurred: {}'.format(error))
        except:
            print('Unknown exception occurred during signals connection')

        self._hstat_gui = HydrostaticsGUI(self)
        self._hstat_gui.menuMain.setEnabled(False)

    @Slot()
    def onImportHullForm(self):
        fname = QFileDialog.getOpenFileName(self.mainwin,
                                            'Select hull form file for import','../../../../examples/kyrenia',
                                            "hull form files (*.hgf *.huf *.obf *.obj *.stl)")
        fname = fname[0]
        if fname != "":
            hfi = HullFormImporter(True)
            hf = hfi.importGeometry(fname)
            if hf is not None:
                hf.emit_geometry_built()

    def onExportHullForm(self):
        available_export = ""
        if isinstance(self.active_hull_form,HullGeneratorForm):
            available_export = "available hull form types (*.hgf *.obj *.stl)"
        elif isinstance(self.active_hull_form, HullGeneratorForm):
            available_export = "available hull form types (*.obj *.stl)"

        fname = QFileDialog.getSaveFileName(self.mainwin,
                                            'Export {0} form as'.format(self.active_hull_form.name),
                                            '../../../../examples/kyrenia', available_export)
        fileName = fname[0]
        self.active_hull_form.exportGeometry(fileName)


    def onMoveHullForm(self):
        if self.active_hull_form is not None:
            MoveMenu = Move_Dialog()
            move_vector = MoveMenu.run()
            if move_vector is not None:
                self.active_hull_form.translate(move_vector)
                self.active_hull_form.emit_geometries_rebuild()

    def onSetReferentHullForm(self):
        if self.referent_hull_form is self.selected_hull_form:
            return
        self.referent_hull_form=self.selected_hull_form
        self.on_referent_hull_form_changed()

    def onSetActiveHullForm(self):
        if self.active_hull_form is self.selected_hull_form:
            return
        self.active_hull_form = self.selected_hull_form
        self.on_active_hull_form_changed()

    def on_active_hull_form_changed(self):
        if self.active_hull_form is not None:
            self._hstat_gui.menuMain.setEnabled(True)
            if isinstance(self.active_hull_form,HullGeneratorForm):
                self.menuHullGenForm.setEnabled(True)
            else:
                self.menuHullGenForm.setEnabled(False)
        else:
            self._hstat_gui.menuMain.setEnabled(False)


    def on_referent_hull_form_changed(self):
        if self.referent_hull_form is not None:
            pass
        else:
            pass
    @property
    def num_hullforms(self):
        return len(self.hull_forms)

    @Slot()
    def onVisibleGeometryChanged(self, visible:List[Geometry], loaded:List[Geometry], selected:List[Geometry]):
        for g in visible:
            pass

    @Slot()
    def onSelectedGeometryChanged(self, visible: List[Geometry], loaded: List[Geometry], selected: List[Geometry]):
        if len(selected) == 1:
            if isinstance(selected[0],HullForm):
                self.selected_hull_form = selected[0]
        elif len(selected) == 0:
            self.selected_hull_form = None

    @Slot()
    def onGeometryCreated(self, geometries:List[Geometry]):
        nlast=self.num_hullforms
        for g in geometries:
            if isinstance(g,HullForm):
                self.hull_forms[g.guid]=g
        if nlast == 0 and self.num_hullforms==1:
            self.active_hull_form=geometries[0]
            self.on_active_hull_form_changed()

    @Slot()
    def onGeometryRemoved(self, geometries:List[Geometry]):
        for g in geometries:
            if isinstance(g, HullForm):
                self.hull_forms.pop(g.guid)
                if self.active_hull_form is g:
                    self.active_hull_form = None
                    self.on_active_hull_form_changed()
                if self.referent_hull_form is g:
                    self.referent_hull_form = None
                    self.on_referent_hull_form_changed()



    def onModifyHullGeneratorForm(self):
        if isinstance(self.active_hull_form, HullGeneratorForm):
            self.hf_prop.setCurrentHullForm(self.active_hull_form)
            self.hf_prop.exec()


    def onOptimizeHullGeneratorForm(self):
        if isinstance(self.active_hull_form, HullGeneratorForm):
            self.hf_opt.setCurrentHullForm(self.active_hull_form)
            self.hf_opt.exec()

    @property
    def app(self):
        return self._app

    @property
    def mainwin(self):
        return self.app.mainFrame

    @property
    def glwin(self):
        return self.mainwin.glWin

class HullFormImporter(IOHandler):
    def __init__(self,force_import=False):
        super().__init__()
        self.force_import=force_import

    def importGeometry(self, fileName):
        if len(fileName) < 1:
            return
        filename_no_ext, file_extension = os.path.splitext(os.path.basename(fileName))
        hf=None
        if file_extension == ".hgf" or file_extension == ".huf":
            hf = HullGeneratorForm(fileName,filename_no_ext)
        elif self.force_import:
            if file_extension == ".stl" or file_extension == ".obj":
                hf = HullFormFromMesh(fileName,filename_no_ext)
        if hf is not None:
            return hf

    def exportGeometry(self, fileName, geometry2export):
        if isinstance(geometry2export,HullForm):
            geometry2export.exportGeometry(fileName)
        om.write_mesh(geometry2export.mesh,fileName)
        pass

    def getExportFormats(self):
       return (".hgf",".obj",".stl")

    def getImportFormats(self):
        return (".hgf",".huf")


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

class DialogHullGeneratorFormModify(QDialog):
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
                        'depth_val':"Depth (D)",
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
                        'depth_val':0,
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
                        'depth_val': 10,
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

    def regenerateHullForm(self):
        for key,txt in self.shipdatatxt.items():
            val = float(txt.text())
            self.currentHullForm.shipdata[key]=val
        self.currentHullForm.regenerateHullHorm()
        self.currentHullForm.emit_geometries_rebuild()

    def setCurrentHullForm(self, currentHullForm):
        self.currentHullForm = currentHullForm
        self.setWindowTitle("Modify Hull Form")
        self.initDicts()
        irow = self.createUserInputs(self.mainLayout)
        btnModify = self.createButton("&Modify", self.regenerateHullForm)
        btnModify.setFocusPolicy(Qt.NoFocus)
        self.mainLayout.addWidget(btnModify, irow, 0)

class DialogHullGeneratorFormOptimize(QDialog):
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
                        'depth_val':"Draft (T)",
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
                        'depth_val':0,
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
                        'depth_val': 10,
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
        self.currentHullForm.regenerateHullHorm()
        self.currentHullForm.emit_geometries_rebuild()

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
        #sol1 = minimize(self.fun_cb, x, constraints=cons, bounds=bnds, method='SLSQP')
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

        h = self.currentHullForm.shipdata['depth_val']
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

class Move_Dialog(object):
    def setupUi(self, Dialog):
        if Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(236, 140)
        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(10, 100, 221, 41))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.verticalLayoutWidget = QWidget(Dialog)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(20, 10, 41, 80))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.MoveLabelX = QLabel(self.verticalLayoutWidget)
        self.MoveLabelX.setObjectName(u"MoveLabelX")

        self.verticalLayout.addWidget(self.MoveLabelX)

        self.MoveLabelY = QLabel(self.verticalLayoutWidget)
        self.MoveLabelY.setObjectName(u"MoveLabelY")

        self.verticalLayout.addWidget(self.MoveLabelY)

        self.MoveLabelZ = QLabel(self.verticalLayoutWidget)
        self.MoveLabelZ.setObjectName(u"MoveLabelZ")

        self.verticalLayout.addWidget(self.MoveLabelZ)

        self.verticalLayoutWidget_2 = QWidget(Dialog)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(70, 10, 160, 80))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.DeltaXInputLine = QLineEdit(self.verticalLayoutWidget_2)
        self.DeltaXInputLine.setObjectName(u"DeltaXInputLine")

        self.verticalLayout_2.addWidget(self.DeltaXInputLine)

        self.DeltaYInputLine = QLineEdit(self.verticalLayoutWidget_2)
        self.DeltaYInputLine.setObjectName(u"DeltaYInputLine")

        self.verticalLayout_2.addWidget(self.DeltaYInputLine)

        self.DeltaZInputLine = QLineEdit(self.verticalLayoutWidget_2)
        self.DeltaZInputLine.setObjectName(u"DeltaZInputLine")

        self.verticalLayout_2.addWidget(self.DeltaZInputLine)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)

    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Move", None))
        self.MoveLabelX.setText(QCoreApplication.translate("Dialog", u"Delta X:", None))
        self.MoveLabelY.setText(QCoreApplication.translate("Dialog", u"Delta Y:", None))
        self.MoveLabelZ.setText(QCoreApplication.translate("Dialog", u"Delta Z:", None))
        self.DeltaXInputLine.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.DeltaYInputLine.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.DeltaZInputLine.setText(QCoreApplication.translate("Dialog", u"1", None))

    # retranslateUi

    def getInput(self):
        return np.array([float(self.DeltaXInputLine.text()), float(self.DeltaYInputLine.text()),
                         float(self.DeltaZInputLine.text())])

    def run(self):
        Form = QDialog()  # Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
        # self = AddForm_Dialog()		#ui je sta se sve nalazi u menu
        self.setupUi(Form)  # setappa ui (ocito)
        Form.exec()  # show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
        if Form.result() == True:  # ako je pritisnut ok
            while True:
                try:
                    move_vector = self.getInput()
                except:
                    print("numbers only")
                    Form.exec()
                    if Form.result() == False:
                        break
                else:
                    print(move_vector)
                    return move_vector
                    break

def createCommand():
    return HullFormCommand()