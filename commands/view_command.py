from PySide6.QtWidgets import QApplication, QMenu, QMessageBox
from PySide6.QtGui import QMatrix4x4, QVector3D, QQuaternion
from commands import Command
from scipy.spatial.transform import Rotation as R
import openmesh as om
import numpy as np
from core import geometry_manager as manager
from core.geometry import Geometry
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFileDialog
from typing import Dict,List

class ViewCommand(Command):
    def __init__(self):
        super().__init__()
        self._app = QApplication.instance()
        tools = self.app.mainFrame.menuTools
        manager.selected_geometry_changed.connect(self.onSelectedGeometryChanged)
        menu = QMenu("View ...")

        top = menu.addAction(" ... Top")
        btm = menu.addAction(" ... Bottom")
        left = menu.addAction(" ... Left")
        right = menu.addAction(" ... Right")
        front = menu.addAction(" ... Front")
        back = menu.addAction(" ... Back")
        iso = menu.addAction(" ... Isometric")
        isoneg = menu.addAction(" ... - Isometric")
        #save = menu.addAction(" ... Save")
        #load = menu.addAction(" ... Load")

        top.triggered.connect(self.onTop)
        btm.triggered.connect(self.onBtm)
        left.triggered.connect(self.onLeft)
        right.triggered.connect(self.onRight)
        front.triggered.connect(self.onFront)
        back.triggered.connect(self.onBack)
        iso.triggered.connect(self.onIso)
        isoneg.triggered.connect(self.onIsoNeg)
        #save.triggered.connect(self.onSave)
        #load.triggered.connect(self.onLoad)

        tools.addMenu(menu)

        menu = QMenu("Export ...")
        top = menu.addAction(" ... Selected Geometry mesh")
        top.triggered.connect(self.onExportSelectedGeoMesh)
        tools.addMenu(menu)
        self._selected_geo :Geometry= None

    @property
    def app(self):
        return self._app

    @property
    def mainwin(self):
        return self.app.mainFrame

    @property
    def glwin(self):
        return self.mainwin.glWin

    @Slot()
    def onSelectedGeometryChanged(self, visible: List[Geometry], loaded: List[Geometry], selected: List[Geometry]):
        if len(selected) == 1:
            self._selected_geo  = selected[0]
        elif len(selected) == 0:
            self._selected_geo = None

    def onExportSelectedGeoMesh(self):
        if self._selected_geo is not None:
            available_export = "available hull form types (*.obj *.stl)"
            fname = QFileDialog.getSaveFileName(self.mainwin,
                                               'Export {0} form as'.format(self._selected_geo.name),
                                               '', available_export)
            if fname != None:
                fileName = fname[0]
                om.write_mesh(fileName,self._selected_geo.mesh)

    def onBack(self):
        rot = QQuaternion.fromEulerAngles(-90.0, 0.0, 180.0)
        self.setView(rot)

    def onFront(self):
        rot = QQuaternion.fromEulerAngles(-90.0, 0.0, 0.0)
        self.setView(rot)

    def onLeft(self):
        rot = QQuaternion.fromEulerAngles(-90.0, 90.0, 0.0)
        self.setView(rot)

    def onRight(self):
        rot = QQuaternion.fromEulerAngles(-90.0, -90.0, 0.0)
        self.setView(rot)

    def onTop(self):
        rot = QQuaternion.fromEulerAngles(0.0, 0.0, 0.0)
        self.setView(rot)

    def onBtm(self):
        rot = QQuaternion.fromEulerAngles(180.0, 0.0, 0.0)
        self.setView(rot)
    def onIso(self):
        mv = QMatrix4x4(0.628121, -0.777908, -0.0179731, -34.2965, 0.224517, 0.159073, 0.961399, -24.0275, -0.745021, -0.60791, 0.27457, 2.20723e-07, 0, 0, 0, 1)
        #rot = QQuaternion.fromEulerAngles(30.0, 30.0, 30.0)
        #mv.rotate(rot)
        #rot = QQuaternion.fromEulerAngles(-90.0, 0.0, 0.0)
        #mv.rotate(rot)
        self.glwin.mv = mv
        self.glwin.update()
    def onIsoNeg(self):
        mv = QMatrix4x4(-0.684249, 0.724864, -0.0798536, 32.2494, -0.352872, -0.233278, 0.906125, 9.49791, 0.638189, 0.648193, 0.415404, -2.00351e-06, 0, 0, 0, 1)
        #rot = QQuaternion.fromEulerAngles(30.0, 30.0, 30.0)
        #mv.rotate(rot)
        #rot = QQuaternion.fromEulerAngles(-90.0, 0.0, 0.0)
        #mv.rotate(rot)
        self.glwin.mv = mv
        self.glwin.update()
    def onSave(self):
        mv = QMatrix4x4()
        print(self.glwin.mv)

    def onLoad(self):
        mv = QMatrix4x4()
        self.glwin.mv = mv
        self.glwin.update()

    def setView(self,rot:QQuaternion):
        mv = QMatrix4x4()
        #mv.translate(trans)
        mv.rotate(rot)
        self.glwin.mv = mv
        self.glwin.update()


def createCommand():
    return ViewCommand()