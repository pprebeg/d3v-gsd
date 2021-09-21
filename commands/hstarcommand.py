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
import openmesh as om
from geometry import Geometry
import numpy as np

class HstarCommand(Command):
    def __init__(self):
        super().__init__()
        self._problem:HstarProblem = None
        self._app = QApplication.instance()
        importer = HstarImporter(self._set_problem)
        self._app.registerIOHandler(importer)
        self._prop_dialog = DialogHstarProps(self.mainwin)
        self._gen_hul=DialogGenerateHUL_file(self.mainwin) #proba
        #self._read_hst=Read_HST_file()
        #p=self._read_hst.points

        self.si = 0

        # tools = app.mainFrame.menuTools
        mb = self.mainwin.menuBar()
        self._menuMain = QMenu("Hydrostar")


        self._menuInput = QMenu("&Generate input files")
        self._menuMain.addMenu(self._menuInput)

        menu_HUL_file=self._menuInput.addAction(".hul")
        menu_HUL_file.triggered.connect(self.onGenerate_hul)
        menu_MRI_file = self._menuInput.addAction(".mri")
        menu_MRI_file.triggered.connect(self.onGenerate_hul)


        self._menuModifyProblem = self._menuMain.addAction("Modify Problem")
        self._menuModifyProblem.triggered.connect(self.onModifyProblem)

        self._menuReadHST = self._menuMain.addAction("Read .hst")
        self._menuReadHST.triggered.connect(self.onRead_hst)

        self._menuGenerateHST = self._menuMain.addAction("Generate HSMSH")
        self._menuGenerateHST.triggered.connect(self.onGenerate_hst)









        #self.menuGenerate_input = self._menuMain.addAction("&Input files")
        #self.menuGenerate_input.triggered.connect(self.Generate_hul)

        #menuinput=self.menuGenerate_input.a

        mb.addMenu(self._menuMain)


        #Signals.get().geometryImported.connect(self.registerDBB)
        #Signals.get().selectionChanged.connect(self.registerSelection)
        self.dbb = 0

    def onModifyProblem(self):
        self._prop_dialog.exec()
        #Signals.get().geometryRebuild.emit(neka_geometrija)  # refresha!!!!!!
    def onGenerate_hul(self):
        self._gen_hul.exec()
    def onRead_hst(self):
        self._read_hst=Read_HST_file()
        hf=HSMSH()
        #hf=self._problem.read_problem()
        #mesh=hf.mesh
        Signals.get().geometryRebuild.emit(hf)

    def onGenerate_hst(self):
        hshf=self._problem.read_problem()
        hsmsh=hshf.mesh
        Signals.get().geometryRebuild.emit(hshf)


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
        self._gen_hul.set_problem(self._problem)
        #self._read_hst.set_problem(self._problem)
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
        lbl = QLabel("Number of Frames")
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

class DialogGenerateHUL_file(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self._sizerow = 25
        self._problem:HstarProblem = None
        self.mainwin = parent
        self._btnOK = self.createButton("&Generate", self.writeHUL_File)

        mainLayout = QGridLayout()
        self.setLayout(mainLayout)


        txtWidth = 80
        i = 0
        # Label
        lbl = QLabel("Number of frames")
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

    def writeHUL_File(self):
        hshf=self._problem.read_problem()
        loa=hshf.LOA
        bmax=hshf.BOA
        frames=hshf.getNewPointsonWL()
        filename = "d3v"
        extension=".hul"
        nsec=len(frames)-1
        HULfilename = filename+extension
        os.chdir("C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\hydrostar")
        with open(HULfilename, "w") as file_object:
            file_object.write("cpb\n")
            file_object.write("{} {} {}\n".format(loa,bmax,nsec))
            i=0
            for frame in frames:
                i=0
                for item in frame:
                    if i == 0:
                        file_object.write("{} {}\n".format(item[0], item[1]))
                    else:
                        file_object.write("{} {} {}\n".format(item[0],item[1],item[2]))
                    i+=1
        return filename

    def writeMRI_File(self,filename):
        ext1=".mri"
        ext2=".hul"
        MRIfilename=filename+ext1
        HULfilename=filename+ext2
        os.chdir("C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\hydrostar")
        with open(MRIfilename, "w") as file_object:
            file_object.write("cpb\n")
            file_object.write("IXDEF\n")
            file_object.write("{}\n".format(1))
            file_object.write("HULF\n")
            file_object.write("{}\n".format(4))
            file_object.write("{}\n".format(HULfilename))
            file_object.write("TYPF\n")
            file_object.write("{}\n".format(0))
            file_object.write("TYPA\n")
            file_object.write("{}\n".format(2))
            file_object.write("DRAUGHT\n")
            file_object.write("{}\n".format(self.shipdata["draft_val"]))
            file_object.write("TRIM\n")
            file_object.write("{}\n".format(0))
            file_object.write("HEEL\n")
            file_object.write("{}\n".format(0))
            file_object.write("ALFK\n")
            file_object.write("{}\n".format(0))
            file_object.write("MPAR\n")
            file_object.write("{} {}\n".format(80,21))
            file_object.write("ENDMR\n")
        pass



    def set_problem(self,problem:HstarProblem):
        self._problem = problem
        if self._problem is not None:
            self._txt_num_frames.setText(str(self._problem.num_frames))

class Read_HST_file():
    def __init__(self):
        filename="d3v.hst"
        fvs,points=self.Read_mesh_file(filename)
        self.hsmsh=self.gen_hsmsh(fvs,points)
        #Signals.get().

    def Read_mesh_file(self,filename):
        os.chdir("C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\hydrostar")
        file=open(filename,"r")
        k=False
        b=False
        points=[]
        fvs=[]
        for line in file:
            splitdata = line.split()
            if splitdata[0] == "COORDINATES":
                k = True
                continue
            elif splitdata[0] == "ENDCOORDINATES":
                k = False
                continue
            if k:
                xp = float(splitdata[1])
                yp = float(splitdata[2])
                zp = float(splitdata[3])
                point = [xp, yp, zp]
                points.append(point)
            if splitdata[0] == "PANEL":
                b = True
                continue
            elif splitdata[0] == "ENDPANEL":
                b = False
                break
            if b:
                p1 = float(splitdata[0]) - 1
                p2 = float(splitdata[1]) - 1
                p3 = float(splitdata[2]) - 1
                p4 = float(splitdata[3]) - 1
                face = [p1, p2, p3, p4]
                fvs.append(face)
            pNeg=np.asarray(points)
            pN=pNeg*(-1)


        return fvs,points

    def gen_hsmsh(self,fvs,points):
        mesh=om.PolyMesh()
        mesh.add_vertices(points)
        mesh.add_faces(fvs)
        return mesh

class HSMSH(Geometry):
    def __init__(self):
        super().__init__()
        hst=Read_HST_file()
        self.mesh=hst.hsmsh

        #Signals.get().geometryHST.emit()

    def genHSMSH(self,lines):
        mesh=om.PolyMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        wlinesPos.reverse()
        wlinesNeg.reverse()

        whsPos = []
        whsNeg = []
        whsi = []
        whsPos.append(whsi)
        whsNeg.append(whsi)
        for p in wlKeel:
            whsi.append(mesh.add_vertex(p))


        for wl in wlinesPos:
            whsi = []
            whsPos.append(whsi)
            for p in wl:
                whsi.append(mesh.add_vertex(p))
        for wl in wlinesNeg:
            whsi = []
            whsNeg.append(whsi)
            for p in wl:
                whsi.append(mesh.add_vertex(p))

        self._genFaces(mesh,whsPos,True)
        self._genFaces(mesh, whsNeg,False)

        return mesh


    def _genFaces(self,mesh:om.PolyMesh,whs:list, doReverse:bool):
        nl=len(whs)
        npt=len(whs[0])
        for iL in range(1, nl):
            npt_iL = len(whs[iL])
            npt_iL_1 = len(whs[iL-1])
            dip=0
            if npt_iL > npt_iL_1:
                if doReverse:
                    mesh.add_face(whs[iL][0], whs[iL][1], whs[iL - 1][0])
                else:
                    mesh.add_face(whs[iL][1], whs[iL][0], whs[iL - 1][0])
                dip = 1
            for ipL_1 in range(1,npt_iL_1):
                ip = ipL_1+dip
                if doReverse:
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip], whs[iL - 1][ipL_1])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip - 1], whs[iL][ip])
                else:
                    mesh.add_face(whs[iL - 1][ipL_1-1],   whs[iL - 1][ipL_1],whs[iL ][ip])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip],    whs[iL][ip-1])



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