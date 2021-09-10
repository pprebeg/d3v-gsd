from PySide2.QtWidgets import QApplication, QMenu,QMainWindow
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout,QToolTip,QLabel, QLineEdit
from PySide2.QtCore import Slot,Qt
from commands import Command
from iohandlers import IOHandler
from signals import Signals
from typing import Dict,List, Tuple
from selinfo import SelectionInfo
from hstardir.hstar import HstarProblem
import os

class HstarCommand(Command):
    def __init__(self):
        super().__init__()
        self._problem:HstarProblem = None
        self._app = QApplication.instance()
        importer = HstarImporter(self._set_problem)
        self._app.registerIOHandler(importer)
        self._prop_dialog = DialogHstarProps(self.mainwin)
        self.si = 0

        # tools = app.mainFrame.menuTools
        mb = self.mainwin.menuBar()
        self._menuMain = QMenu("Hydrostar")


        self._menuSubmenu = QMenu("&Submenu")
        self._menuMain.addMenu(self._menuSubmenu)

        self._menuModifyProblem = self._menuMain.addAction("Modify Problem")
        self._menuModifyProblem.triggered.connect(self.onModifyProblem)

        mb.addMenu(self._menuMain)

        #Signals.get().geometryImported.connect(self.registerDBB)
        #Signals.get().selectionChanged.connect(self.registerSelection)
        self.dbb = 0

    def onModifyProblem(self):
        self._prop_dialog.exec()
        #Signals.get().geometryRebuild.emit(neka_geometrija)  # refresha!!!!!!


    @property
    def app(self):
        return self._app
    @property
    def mainwin(self):
        return self.app.mainFrame

    @property
    def glwin(self):
        return self.mainwin.glWin

    def _set_problem(self, problem):
        self._problem = problem
        self._prop_dialog.set_problem(self._problem)
        Signals.get().geometryImported.emit(self._problem.hull)

class DialogHstarProps(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self._sizerow = 25
        self._problem:HstarProblem = None
        self.mainwin = parent
        self._btnOK = self.createButton("&OK", self.apply)

        mainLayout = QGridLayout()
        self.setLayout(mainLayout)


        txtWidth = 80
        i = 0
        # Label
        lbl = QLabel("Number of Fames")
        lbl.setFixedHeight(self._sizerow)
        mainLayout.addWidget(lbl, i, 0)
        #Text Box
        self._txt_num_frames = QLineEdit(self)
        self._txt_num_frames.setAlignment(Qt.AlignRight)
        self._txt_num_frames.setFixedHeight(self._sizerow)
        self._txt_num_frames.setFixedWidth(txtWidth)
        if self._problem is not None:
            self._txt_num_frames.setText(str(self._problem.num_frames))
        mainLayout.addWidget(self._txt_num_frames, i, 1)
        i+=1
        mainLayout.addWidget(self._btnOK, i, 0)

    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def apply(self,n):
        if (self._problem is not None):
            self._problem.num_frames = int(self._txt_num_frames.text())
            self.close()
            #Signals.get().geometryRebuild.emit(self.neka_gemetrija)		#refresha?
    def set_problem(self,problem:HstarProblem):
        self._problem = problem
        if self._problem is not None:
            self._txt_num_frames.setText(str(self._problem.num_frames))

class HstarImporter(IOHandler):
    def __init__(self,func_set_problem):
        super().__init__()
        self._fun_set_problem=func_set_problem #function for seting the problem of parent object

    def importGeometry(self, fileName):
        if len(fileName) < 1:
            return
        filename, file_extension = os.path.splitext(fileName)
        if file_extension != ".hsr":
            return
        self._fun_set_problem(HstarProblem(fileName))


    def getImportFormats(self):
        return [".hsr"]

def createCommand():
    return HstarCommand()