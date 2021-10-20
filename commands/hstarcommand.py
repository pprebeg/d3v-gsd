from PySide2.QtWidgets import QApplication, QMenu,QMainWindow
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout,QToolTip,QLabel, QLineEdit,QDialogButtonBox,QTabWidget,QVBoxLayout,QWidget,QHBoxLayout,QRadioButton,QCheckBox,QGroupBox
from PySide2.QtCore import Slot,Qt
from commands import Command
from iohandlers import IOHandler
from signals import Signals
from typing import Dict,List, Tuple
from selinfo import SelectionInfo
from hstardir.hstar import HstarProblem,HstarHullForm
import os
import subprocess
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
        self._hull_mod=DialogHullFormModify(self.mainwin)
        self._calc_dialog=DialogSeakeepngCalculation(self.mainwin)
        #self._read_hst=Read_HST_file()
        #p=self._read_hst.points

        self.hf=0
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
        menu_HST_file = self._menuInput.addAction(".hst")
        menu_HST_file.triggered.connect(self.onWrit_hst)



        self._menuModifyProblem = self._menuMain.addAction("Modify Problem")
        self._menuModifyProblem.triggered.connect(self.onModifyProblem)
        separator=self._menuMain.addSeparator()
        self._menuReadHST = self._menuMain.addAction("Seakeeping calculation")
        self._menuReadHST.triggered.connect(self.onSeaKeepingCalculation)



        mb.addMenu(self._menuMain)

        #Signals.get().geometryImported.connect(self.registerDBB)
        #Signals.get().selectionChanged.connect(self.registerSelection)
        self.dbb = 0

        Signals.get().geometryAdded.connect(self.registerHullForm)
        Signals.get().selectionChanged.connect(self.registerSelection)


    @Slot()
    def registerHullForm(self, hullForm):
        if isinstance(hullForm, HstarHullForm):
            self.hf=hullForm
            #self.menuMain.setEnabled(True)
            #om.write_mesh("D:\hullForm.obj", hullForm.mesh)

    @Slot()
    def registerSelection(self, si):
        self.si=si


    def onModifyProblem(self):
        self._hull_mod.setCurrentHullForm(self.hf)
        self._hull_mod.exec()
        #Signals.get().geometryRebuild.emit(neka_geometrija)  # refresha!!!!!!

    def onSeaKeepingCalculation(self):
        if isinstance(self.hf,HstarHullForm):
            self._calc_dialog.setCurrentHullForm(self.hf)
            self._calc_dialog.exec()

    def onGenerate_hul(self):
        self._gen_hul.exec()

    def onRead_hst(self):
        pass

    def onWrit_hst(self):
        fvs=self._problem.read_problem().mesh.fv_indices().tolist()
        points=self._problem.read_problem().mesh.points().tolist()
        org_point=[0.0,0.0,-self._problem.hull.T]
        points.insert(0,org_point)
        self._write_hst=Read_Write_HST_file().Write_hst_file(fvs,points)


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

class DialogSeakeepngCalculation(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self._sizerow = 25
        self._problem:HstarProblem = None
        self.mainwin = parent
        self.resize(400,400)

        self.currentHullForm=0
        self.gen_data= {}
        self.mass_prop={"mass":0,"dens":0,"grav":0,"lvd":0,"gyr":[],"cog":[],"ref":[]}
        self.calc_set={}
        #self._btnOK = self.createButton("&OK", self.apply)

        self._btnOK=self.createButtonBox()
        self._tabWdiget=self.createTabWidget()


        mainLayout = QVBoxLayout()
        #mainLayout=QGridLayout()
        self.setLayout(mainLayout)

        mainLayout.addWidget(self._tabWdiget)
        mainLayout.addWidget(self._btnOK)

    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def createButtonBox(self):
        btnBox=QGroupBox()
        btnBox.layout=QHBoxLayout()
        btnBox.setLayout(btnBox.layout)
        width=80

        show_resultsButton=QPushButton("&Show Results")
        show_resultsButton.clicked.connect(self.close)
        show_resultsButton.setFixedWidth(width)
        btnBox.layout.addWidget(show_resultsButton)

        #calcButton=QPushButton("&Calculate",self.apply)
        calcButton=QPushButton("&Calculate")
        calcButton.clicked.connect(self.calculation)
        calcButton.setFixedWidth(width)
        btnBox.layout.addWidget(calcButton)

        closeButton=QPushButton("&Close")
        closeButton.clicked.connect(self.close)
        closeButton.setFixedWidth(width)
        btnBox.layout.addWidget(closeButton)

        return btnBox

    def createTabWidget(self):
        tabWidget=QTabWidget()

        # Initialize tab screen
        #self.tabs = QTabWidget()
        tab_gen = QWidget()
        tab_mass = QWidget()
        tab_calc=QWidget()
        tabWidget.resize(300, 200)

        # Add tabs
        tabWidget.addTab(tab_gen, "General")
        tabWidget.addTab(tab_mass, "Mass properties")
        tabWidget.addTab(tab_calc, "Calculation settings")


        # Create tabs
        tab_gen=self.createTabGen(tab_gen)
        tab_mass=self.createTabMass(tab_mass)
        tab_calc=self.createTabCalc(tab_calc)


        return tabWidget
    
    def createTabGen(self,tab_gen:QWidget):

        tab_gen.layout = QVBoxLayout()
        tab_gen.setLayout(tab_gen.layout)

        lbl_name = QLabel("Project name:")
        lbl_name.setFixedHeight(self._sizerow)

        txt_proj_name = QLineEdit(self)
        txt_proj_name.setAlignment(Qt.AlignRight)
        txt_proj_name.setFixedHeight(self._sizerow)
        txt_proj_name.setFixedWidth(150)
        #txt_proj_name.setText("mod_00")
        txt_proj_name.setPlaceholderText("mod_00")

        proj = QWidget()
        proj.layout = QHBoxLayout()
        proj.setLayout(proj.layout)
        proj.layout.addWidget(lbl_name)
        proj.layout.addWidget(txt_proj_name)
        proj.setFixedWidth(260)
        proj.setFixedHeight(80)
        tab_gen.layout.addWidget(proj)

        msh=QGroupBox()
        msh.setLayout(QGridLayout())
        nlng_lbl=QLabel("N_long:")
        nlng_txt=QLineEdit()
        ntr_lbl=QLabel("N_tocz:")
        ntr_txt=QLineEdit()
        msh.layout().addWidget(QLabel("Mesh grid"), 0, 0)
        msh.layout().addWidget(nlng_lbl,1,0)
        msh.layout().addWidget(nlng_txt, 1, 1)
        msh.layout().addWidget(ntr_lbl, 2, 0)
        msh.layout().addWidget(ntr_txt, 2, 1)
        msh.setFixedHeight(100)
        msh.setFixedWidth(150)
        tab_gen.layout.addWidget(msh)

        amg = QGroupBox()
        amg.setLayout(QVBoxLayout())
        btn_hul = QCheckBox("hul_file")
        btn_hul.setChecked(True)
        btn_mri = QCheckBox("mri_file")
        btn_mri.setChecked(True)

        #btn_hul.set
        lbl1=QLabel("Input files")
        amg.layout().addWidget(lbl1)
        amg.layout().addWidget(btn_hul)
        amg.layout().addWidget(btn_mri)
        amg.setFixedHeight(100)
        amg.setFixedWidth(150)
        tab_gen.layout.addWidget(amg)

        btnModify = QPushButton("&Modify Form")
        btnModify.setFixedWidth(120)
        #btnModify.connect()
        tab_gen.layout.addWidget(btnModify)
        self.gen_data= {"proj_name":txt_proj_name.text(),"mshq":[nlng_txt.text(),ntr_txt.text()],"add_inp":[btn_hul.text(),btn_mri.text()]}
        return tab_gen

    def createTabMass(self,tab_mass:QWidget):

        tab_mass.layout=QGridLayout()
        tab_mass.setLayout(tab_mass.layout)

        #mass input
        i=0
        mass_lbl=QLabel("Mass, kg:")
        mass_txt=QLineEdit()
        mass_enbl=QRadioButton("&Enable")
        tab_mass.layout.addWidget(mass_lbl,i,0)
        tab_mass.layout.addWidget(mass_txt, i, 1)
        tab_mass.layout.addWidget(mass_enbl, i, 2)
        i+=1

        dens_lbl=QLabel("Water density, kg/m^3:") #water density
        dens_txt=QLineEdit("1025")
        tab_mass.layout.addWidget(dens_lbl,i,0)
        tab_mass.layout.addWidget(dens_txt, i, 1)
        i+=1

        grav_lbl=QLabel("Gravity, kg/ms^2:") #gravity acceleration
        grav_txt=QLineEdit("9.81")
        tab_mass.layout.addWidget(grav_lbl,i,0)
        tab_mass.layout.addWidget(grav_txt, i, 1)
        i+=1

        lvd_lbl=QLabel("Viscous dumping (lin), %:") #linear viscous dumping
        lvd_txt=QLineEdit("5")
        tab_mass.layout.addWidget(lvd_lbl,i,0)
        tab_mass.layout.addWidget(lvd_txt, i, 1)
        i+=1

        xos = QLabel("Xos")
        xos.setFixedHeight(self._sizerow)
        yos = QLabel("Yos")
        yos.setFixedHeight(self._sizerow)
        zos = QLabel("Zos")
        zos.setFixedHeight(self._sizerow)
        tab_mass.layout.addWidget(xos, i, 1)
        tab_mass.layout.addWidget(yos, i, 2)
        tab_mass.layout.addWidget(zos, i, 3)
        tab_mass.layout.setVerticalSpacing(0)
        i += 1
        gyr_lbl=QLabel("Gyradius (GR), kgm^2:")
        gyr_txt_x=QLineEdit()
        gyr_txt_y=QLineEdit()
        gyr_txt_z = QLineEdit()
        tab_mass.layout.addWidget(gyr_lbl,i,0)
        tab_mass.layout.addWidget(gyr_txt_x, i, 1)
        tab_mass.layout.addWidget(gyr_txt_y, i, 2)
        tab_mass.layout.addWidget(gyr_txt_z, i, 3)
        i+=1

        cog_lbl=QLabel("Center of Gravtiy (CoG), m:")
        cog_txt_x=QLineEdit()
        cog_txt_y=QLineEdit()
        cog_txt_z = QLineEdit()
        tab_mass.layout.addWidget(cog_lbl,i,0)
        tab_mass.layout.addWidget(cog_txt_x, i, 1)
        tab_mass.layout.addWidget(cog_txt_y, i, 2)
        tab_mass.layout.addWidget(cog_txt_z, i, 3)
        i+=1

        ref_lbl=QLabel("Referent point (RP), m:")
        ref_txt_x=QLineEdit()
        ref_txt_y=QLineEdit()
        ref_txt_z = QLineEdit()
        tab_mass.layout.addWidget(ref_lbl,i,0)
        tab_mass.layout.addWidget(ref_txt_x, i, 1)
        tab_mass.layout.addWidget(ref_txt_y, i, 2)
        tab_mass.layout.addWidget(ref_txt_z, i, 3)
        i+=1

        mass_prop=[mass_txt,dens_txt,grav_txt,lvd_txt,[gyr_txt_x,gyr_txt_y,gyr_txt_z],[cog_txt_x,cog_txt_y,cog_txt_z],[ref_txt_x,ref_txt_y,ref_txt_z]]
        i=0
        for key in self.mass_prop:
            self.mass_prop[key]=mass_prop[i].text()

    def createTabCalc(self,tab_calc:QWidget):

        tab_calc.layout=QVBoxLayout()
        tab_calc.setLayout(tab_calc.layout)

        lbl_calc_draught= QLabel("Calculation draught (T), m:")
        lbl_calc_draught.setFixedHeight(self._sizerow)

        txt_calc_draught = QLineEdit()
        txt_calc_draught.setAlignment(Qt.AlignRight)
        txt_calc_draught.setFixedHeight(self._sizerow)
        txt_calc_draught.setFixedWidth(100)
        txt_calc_draught.setText("9.0")

        darught = QWidget()
        darught.layout = QHBoxLayout()
        darught.setLayout(darught.layout)
        darught.layout.addWidget(lbl_calc_draught)
        darught.layout.addWidget(txt_calc_draught)
        darught.setFixedWidth(260)
        darught.setFixedHeight(80)
        tab_calc.layout.addWidget(darught)



        om=QGroupBox()
        om.layout=QGridLayout()
        om.setLayout(om.layout)
        om.setFixedHeight(100)
        #om.setFixedWidth(280)
        om.layout.addWidget(QLabel("Frequences input box:"))
        om.layout.addWidget(QLabel("min."),0,1)
        om.layout.addWidget(QLabel("max."), 0, 2)
        om.layout.addWidget(QLabel("step."), 0, 3)
        om_lbl=QLabel("w, rad/s:")
        om_min=QLineEdit()
        om_max = QLineEdit()
        om_step = QLineEdit()
        om.layout.addWidget(om_lbl,1,0)
        om.layout.addWidget(om_min,1,1)
        om.layout.addWidget(om_max, 1, 2)
        om.layout.addWidget(om_step, 1, 3)
        tab_calc.layout.addWidget(om)

        hed = QGroupBox()
        hed.layout = QGridLayout()
        hed.setLayout(hed.layout)
        hed.setFixedHeight(100)
        #hed.setFixedWidth(280)
        hed.layout.addWidget(QLabel("Headings input box:"))
        hed.layout.addWidget(QLabel("min."), 0, 1)
        hed.layout.addWidget(QLabel("max."), 0, 2)
        hed.layout.addWidget(QLabel("step."), 0, 3)
        hed_lbl = QLabel("Heading angle, deg:")
        hed_lbl.setFixedWidth(150)
        hed_min = QLineEdit()
        hed_max = QLineEdit()
        hed_step = QLineEdit()
        hed.layout.addWidget(hed_lbl, 1, 0)
        hed.layout.addWidget(hed_min, 1, 1)
        hed.layout.addWidget(hed_max, 1, 2)
        hed.layout.addWidget(hed_step, 1, 3)
        tab_calc.layout.addWidget(hed)

        spd=QGroupBox()
        spd.layout=QGridLayout()
        spd.setLayout(spd.layout)
        spd.setFixedHeight(100)
        spd.layout.addWidget(QLabel("Speed input box:"))
        spd_lbl = QLabel("Speed (v), m/s:")
        spd_val = QLineEdit()
        #spd_val.setFixedWidth(80)
        spd.layout.addWidget(spd_lbl, 1, 0)
        spd.layout.addWidget(spd_val, 1, 1)
        tab_calc.layout.addWidget(spd)
        self.calc_set={"calc_draught":txt_calc_draught.text(),"frqn":[om_min.text(),om_max.text(),om_step.text()],"head":[hed_min.text(),hed_max.text(),hed_step.text()],"speed":spd_val.text()}

    def setCurrentHullForm(self, currentHullForm):
        self.currentHullForm = currentHullForm

    def getHydroStaticResults(self,currentHullForm:HstarHullForm):
        hstat_results=self.currentHullForm._hullform.getResults()



    def connectHydrostar(self,proj_name):
        path="C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\calculation\Projects"
        name="mod_00"
        path=path+"\_" + proj_name
        os.system('cmd /k "cd  "C:\Program Files\Bureau Veritas\Hstar4Experts\v8.2.0\bin" "')
        os.system('cmd /k "Hstar4Expert_v8.2.0.bat" "')
        cmd="cd"+path+"\_"+proj_name
        os.system('cmd /k "cd "C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\calculation\Projects\_mod00""')

        proj_cmd="cmd\k proj"+proj_name
        os.system(proj_cmd)
        hslec_cmd="cmd /k hslec " +proj_name+".hst"
        os.system(hslec_cmd)
        haschk_cmd="cmd /k hschk"
        os.system(haschk_cmd)
        hastat_cmd="cmd /k hstat"
        os.system(hastat_cmd)
        hsrdf_cmd="cmd /k hsrdf -n 4 " + proj_name+".rdf"
        os.system(hsrdf_cmd)
        hsmasn_cmd="cmd /k hsmcn " +proj_name+".mec"
        os.system(hsmasn_cmd)
        hsrao_cmd="cmd /k hsrao " +proj_name+".rao"
        os.system('cmd /c "hsrao mod00.rao"')
        pass


    def calculation(self):
        """if (self._problem is not None):
            self._problem.num_frames = int(self._txt_proj_name.text())
            self.close()"""
        if isinstance(self.currentHullForm,HstarHullForm):
            self.write_func=writingCalculationData(self.gen_data,self.mass_prop,self.calc_set,self.currentHullForm)
            self.connect(self.proj_name)
            #Signals.get().geometryRebuild.emit(self.neka_gemetrija)		#refresha?

    def set_problem(self,problem:HstarProblem):
        self._problem = problem
        if self._problem is not None:
            self._txt_proj_name.setText(str(self._problem.num_frames))


class writingCalculationData():

    def __init__(self,gen_data:dict,mass_prop:dict,calc_set:dict,hshf:HstarHullForm):

        self.proj_name=gen_data["proj_name"]
        self.nlng=gen_data["mshq"][0]
        self.ntr = gen_data["mshq"][1]
        hul=gen_data["add_inp"][0]
        mri=gen_data["add_inp"][1]
        self.calc_draught=float(calc_set["calc_draught"])


        self.hshf=hshf

        path="C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\calculation\Projects"
        self.path=path+ self.proj_name
        os.chdir(path)
        os.mkdir(path)

        if hul:
            self.write_hul_file(self.proj_name)
        if mri:
            self.write_mri_file(self.proj_name)

        self.write_hst_file(self.proj_name)
        self.write_mec_file(self.proj_name,mass_prop)
        self.write_rdf_file(self.proj_name,calc_set)
        self.write_rao_file(self.proj_name)


    def write_hul_file(self,proj_name):
        #self.hshf=hshf
        loa=self.hshf.LOA
        bmax=self.hshf.BOA
        T=self.calc_draught

        xPos = self.hshf.getXpositions()
        #xPos = list(filter(lambda x: x > self.x0, xPos))

        wlPos = self.hshf.getWLPositions(T)
        lines = self.hshf._hullform.hullGen(self.hshf.shipdata, wlPos, 20, xPos)
        frames = self.hshf.genFramesfromWL(lines, xPos)

        nsec=len(frames)-1

        extension=".hul"
        HULfilename = proj_name+extension

        with open(HULfilename, "w") as file_object:
            file_object.write("d3v_gsd: hul file\n")
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

    def write_mri_file(self,proj_name):
        ext=".mri"
        MRIfilename=proj_name+ext
        HULfilename=proj_name+".hul"
        draught=self.calc_draught

        with open(MRIfilename, "w") as file_object:
            file_object.write("d3v_gsd: mri_file\n")
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
            file_object.write("{}\n".format(draught))
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

    def write_hst_file(self,proj_name,fvs:list,points:list):
        filename=proj_name+".hst"
        fvs=self.hshf.mesh.fv_indices().tolist()
        points=self.hshf.mesh.points().tolist()

        with open(filename, "w") as file_object:
            file_object.write("Mesh by d3v_gsd:\n")
            file_object.write(" SYMMETRY_BODY {} {}\n".format(1,1))
            file_object.write(" NUMPANEL {} {} {}\n".format(1, 1,len(fvs)))
            file_object.write(" COORDINATES TYPE {}\n".format(0))
            #i=0
            #file_object.write(" {}  {}  {}  {}\n".format(i, 0.0, 0.0, -9.4))
            i=1
            for p in points:
                xp="{:.5e}".format(p[0])
                yp = "{:.5e}".format(p[1])
                zp = "{:.5e}".format(p[2])
                file_object.write(" {}  {}  {}  {}\n".format(i, xp, yp,zp))
                i+=1
            file_object.write(" ENDCOORDINATES\n")
            file_object.write(" PANEL TYPE {}\n".format(0))
            for f in fvs:
                f3=f[3]
                if f3==-1:
                    f3=f[2]
                file_object.write(" {} {} {} {}\n".format(f[0]+2, f[1]+2, f[2]+2, f3+2))
            file_object.write(" ENDPANEL\n")
            file_object.write(" ENDFILE\n")

        pass

    def write_mec_file(self,proj_name,mass_prop:dict): #self.mass_prop={"mass":0,"dens":0,"grav":0,"lvd":0,"gyr":[],"cog":[],"ref":[]}
        filename = proj_name + ".mec"
        mass=mass_prop["mass"]
        density=mass_prop["dens"]
        grav=mass_prop["grav"]
        lvd=mass_prop["lvd"]
        gyr=mass_prop["gyr"]
        cog = mass_prop["cog"]
        ref = mass_prop["ref"]
        with open(filename, "w") as file_object:
            file_object.write("FILENAME A1\n\n")
            file_object.write("{} {} {}\n\n".format("MASS_BODY",1,mass))
            file_object.write("{} {} {} {} {}\n\n".format("GYRADIUS_BODY", 1, gyr[0],gyr[1],gyr[2]))
            file_object.write("{} {} {} {} {}\n\n".format("COGPOINT_BODY", 1, cog[0], cog[1], cog[2]))
            file_object.write("{} {} {} {} {}\n\n".format("REFPOINT_BODY", 1, ref[0], ref[1], ref[2]))
            file_object.write("{} {} {} {}\n\n\n".format("LINVISCOUSDAMPING", 1, lvd,"%"))
            file_object.write(" ENDFILE\n")
        pass

    def write_rdf_file(self,proj_name,calc_set:dict): #self.calc_set={"calc_draught":txt_calc_draught,"frqn":[om_min,om_max,om_step],"head":[hed_min,hed_max,hed_step],"speed":spd_val}
        filename = proj_name + ".mec"
        om=calc_set["frqn"]
        head=calc_set["head"]
        headings=np.linspace(head[0],head[1],head[2],endpoint=True)
        spd=calc_set["speed"]
        with open(filename, "w") as file_object:
            file_object.write("FILENAME A1\n\n")
            file_object.write("{} {}\n".format("FREQUENCES	TYPE",2))
            file_object.write(" {} {}\n".format("WMIN", om[0]))
            file_object.write(" {} {}\n".format("WMAX", om[1]))
            file_object.write(" {} {}\n".format("WSTP", om[2]))
            file_object.write("ENDFREQUENCES\n\n\n")

            file_object.write("{} {}\n".format("HEADINGS	TYPE",0))
            i=1
            for ang in headings:
                file_object.write("{} {}\n".format(i, ang))
                i+=1
            file_object.write("ENDHEADINGS\n\n\n")

            file_object.write("{} {}\n".format("SPEEDS	TYPE",0))
            file_object.write("{} {}\n".format(1, spd))
            file_object.write("ENDSPEEDS\n\n\n")

            file_object.write("ELIMIRREG YES\n\n\n")
            file_object.write("ELIMIRREG YES\n")

        pass

    def write_rao_file(self,proj_name):
        filename = proj_name + ".rao"
        with open(filename, "w") as file_object:
            file_object.write("\n")
            file_object.write("{} {}\n".format("GSURGE	FILE","01_gsurge.rao"))
            file_object.write("{} {}\n".format("GSWAY	FILE", "02_gsway.rao"))
            file_object.write("{} {}\n".format("GHEAVE	FILE", "03_gheave.rao"))
            file_object.write("{} {}\n".format("GROLL	FILE", "04_groll.rao"))
            file_object.write("{} {}\n".format("GPITCH	FILE", "05_gpitch.rao"))
            file_object.write("{} {}\n".format("GYAW	FILE", "06_gyaw.rao"))

        pass



class Read_Write_HST_file():
    def __init__(self):
        #os.chdir("C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples")
        filename="d3v.hst"
        fvs,points=self.Read_mesh_file(filename)
        self.hsmsh=self.gen_hsmsh(fvs,points)
        #self._problem=HstarProblem(filename)
       #self.hshf=self._problem.read_problem()
        #self.mesh=self.hshf.mesh
        #fvs = self.mesh.fv_indices().tolist()
        #points = self.mesh.points().tolist()
        write=self.Write_hst_file(fvs,points)

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
                p1 = int(splitdata[0]) - 1
                p2 = int(splitdata[1]) - 1
                p3 = int(splitdata[2]) - 1
                p4 = int(splitdata[3]) - 1
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

    def Write_hst_file(self,fvs,points):
        #fvs = self.mesh.fv_indices().tolist()
        #points = self.mesh.points().tolist()
        #org_point=
        filename="hey2.hst"
        path="C:\Work\Ivan_Orec\iodipl\python_d3v\d3v-gsd\examples\hydrostar"
        newdir="\hey1"
        dir=path+newdir
        #dir = os.path.join("C:\\", "hydrostar", "googo")
        if not os.path.exists(dir):
            os.mkdir(dir)

        os.chdir(dir)


        with open(filename, "w") as file_object:
            file_object.write("Mesh by d3v:\n")
            file_object.write(" SYMMETRY_BODY {} {}\n".format(1,1))
            file_object.write(" NUMPANEL {} {} {}\n".format(1, 1,len(fvs)))
            file_object.write(" COORDINATES TYPE {}\n".format(0))
            #i=0
            #file_object.write(" {}  {}  {}  {}\n".format(i, 0.0, 0.0, -9.4))
            i=1
            for p in points:
                xp="{:.5e}".format(p[0])
                yp = "{:.5e}".format(p[1])
                zp = "{:.5e}".format(p[2])
                file_object.write(" {}  {}  {}  {}\n".format(i, xp, yp,zp))
                i+=1
            file_object.write(" ENDCOORDINATES\n")
            file_object.write(" PANEL TYPE {}\n".format(0))
            for f in fvs:
                f3=f[3]
                if f3==-1:
                    f3=f[2]
                file_object.write(" {} {} {} {}\n".format(f[0]+2, f[1]+2, f[2]+2, f3+2))
            file_object.write(" ENDPANEL\n")
            file_object.write(" ENDFILE\n")

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