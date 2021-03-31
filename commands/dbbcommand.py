from PySide2.QtWidgets import QApplication, QMenu
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout
from commands import Command
from iohandlers import IOHandler
from signals import Signals
from dbbdir.dbb import DBBProblem,DBB,DBBHullForm, DBBDeck
import os
from PySide2.QtCore import Slot
from dbbdir import dbbmenus as mm


class DBBCommand(Command):
	def __init__(self):
		super().__init__()
		app = QApplication.instance()
		importer=DBBImporter()
		importer.fsetproblem=self.setProblem
		app.registerIOHandler(importer)
		self.mainwin = app.mainFrame
		self.dbbprop = DialogDBBProps(self.mainwin)
		self.dbbproblem=0
		self.si=0

		#tools = app.mainFrame.menuTools
		mb = app.mainFrame.menuBar()

		self.menuMain = QMenu("DBB")
		
		self.menuImportFromCsv = self.menuMain.addAction("Import From Csv")
		self.menuImportFromCsv.triggered.connect(self.onImportFromCsv)
		
		self.menuInitTestProblem = self.menuMain.addAction("TestProblem")
		self.menuInitTestProblem.triggered.connect(self.onGenerateTestProblerm)
		
		self.menuModifyBlock = QMenu("&Modify block")
		self.menu2 = QMenu("&Menu2")
		self.menuMain.addMenu(self.menuModifyBlock)
		self.menuMain.addMenu(self.menu2)



		menuMove = self.menuModifyBlock.addAction("Move")
		menuMove.triggered.connect(self.onMoveDBB)

		menuSetPosition = self.menuModifyBlock.addAction("Set Position")
		menuSetPosition.triggered.connect(self.onSetPosition)
				
		self.menuDeleteSelected = self.menuModifyBlock.addAction("Delete Selected")
		self.menuDeleteSelected.triggered.connect(self.onDeleteSelected)
		
		menuCutBlock = self.menuModifyBlock.addAction("Cut Block")
		menuCutBlock.triggered.connect(self.onCutBlock)
		
		menuMeshVolume = self.menu2.addAction("Mesh Volume")
		menuMeshVolume.triggered.connect(self.onMeshVolume)
		
		menuIsClosed = self.menu2.addAction("Is Closed?")
		menuIsClosed.triggered.connect(self.onIsClosed)
		#tools.addMenu(menu)
		mb.addMenu(self.menuMain)
		#self._menuMain.setEnabled(False)


		Signals.get().geometryAdded.connect(self.registerDBB)
		Signals.get().selectionChanged.connect(self.registerSelection)
		self.dbb = 0

	def setProblem(self,dbbproblem):
		self.dbbproblem=dbbproblem

	@Slot()
	def registerDBB(self, dbbproblem):
		if isinstance(dbbproblem,DBBProblem):
			self.dbbproblem=dbbproblem
			self.menuMain.setEnabled(True)

	@Slot()
	def registerSelection(self, si):
		self.si=si

	def onGenerateTestProblerm(self):
		FormMenu = mm.AddForm_Dialog()
		scale = FormMenu.run()
		
		BlockMenu = mm.AddBlock_Dialog()
		block_dims = BlockMenu.run()
		
		if block_dims is not None and scale is not None:	#ako niti jedan od menia nije cancelan
			self.dbbproblem = DBBProblem("")
			self.dbbproblem.testProblem(scale, block_dims)
			Signals.get().geometryImported.emit(self.dbbproblem.hull)
			for deck in self.dbbproblem.decks:
				Signals.get().geometryImported.emit(deck)
			for dbb in self.dbbproblem.dbbs:
				Signals.get().geometryImported.emit(dbb)
			self.menuInitTestProblem.setEnabled(False)

	def onDeleteSelected(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			print(self.dbbproblem)
			if isinstance(currDBB, DBB) or isinstance(currDBB, DBBHullForm) or isinstance(currDBB, DBBDeck):
				self.dbbproblem.dbbs.remove(currDBB)
				Signals.get().geometryRemoved.emit(currDBB)	#refresha!!!!!!
	
	def onMoveDBB(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			if isinstance(currDBB, DBB) or isinstance(currDBB, DBBHullForm) or isinstance(currDBB, DBBDeck):
				MoveMenu = mm.Move_Dialog()
				move_vector = MoveMenu.run()
				if move_vector is not None:
					currDBB.move(move_vector)
					Signals.get().geometryRebuild.emit(currDBB)	#refresha!!!!!!
					#self.dbbprop.setCurrentDBB(currDBB)
					#self.dbbprop.moveCurrentDBB()
					##

	def onSetPosition(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			if isinstance(currDBB,DBB) or isinstance(currDBB,DBBHullForm) or isinstance(currDBB,DBBDeck):
				SetPositionMenu = mm.SetPosition_Dialog()
				new_position = SetPositionMenu.run()
				if new_position is not None:
					currDBB.setPosition(new_position)
					Signals.get().geometryRebuild.emit(currDBB)	#refresha!!!!!!
		
	def onCutBlock(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			if isinstance(currDBB,DBB):
				currDBB.cutMesh()
				Signals.get().geometryRebuild.emit(currDBB)	#refresha!!!!!!
		
	def onMeshVolume(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			if isinstance(currDBB,DBB):
				currDBB.calcVolume()
	
	
	def onIsClosed(self):
		if self.si.haveSelection():
			currDBB=self.si.getGeometry()
			if isinstance(currDBB,DBB):
				currDBB.IsClosed()
	
	def onImportFromCsv(self):
		ImportFromCsvMenu = mm.ImportFromCsvMenu_Dialog()
		folder_path = ImportFromCsvMenu.run()	#dodaj jos uvijet da se u folderu nalaze prave datoteke

		
		self.dbbproblem = DBBProblem("")
		self.dbbproblem.readProblem(folder_path)
		#make form from huf
		#with open(huf_path, "_r") as csv:
		
		#tu ih emmita u vizualizaciju
		Signals.get().geometryImported.emit(self.dbbproblem.hull)
		for deck in self.dbbproblem.decks:
			Signals.get().geometryImported.emit(deck)
		for dbb in self.dbbproblem.dbbs:
			Signals.get().geometryImported.emit(dbb)
		#self.menuInitTestProblem.setEnabled(False)

		
		
		
		
		
		
		
		
		
class DBBImporter(IOHandler):
	def __init__(self):
		super().__init__()
		self.fsetproblem=0

	def importGeometry(self, fileName):
		if len(fileName) < 1:
			return
		filename, file_extension = os.path.splitext(fileName)
		if file_extension != ".dbb":
			return
		self.dbbproblem = DBBProblem(fileName)
		self.fsetproblem(self.dbbproblem)
		print(self.dbbproblem)
		Signals.get().geometryImported.emit(self.dbbproblem.hull)
		for deck in self.dbbproblem.decks:
			Signals.get().geometryImported.emit(deck)
		for dbb in self.dbbproblem.dbbs:
			Signals.get().geometryImported.emit(dbb)

	def getImportFormats(self):
		return [".dbb"]

class DialogDBBProps(QDialog):
	def __init__(self, parent):
		super().__init__(parent)
		self.mainwin = parent
		self.btnMove = self.createButton("&Move", self.moveCurrentDBB)

		mainLayout = QGridLayout()
		mainLayout.addWidget(self.btnMove, 0, 0)
		self.setLayout(mainLayout)
		self.currentDBB=0


	def createButton(self, text, member):
		button = QPushButton(text)
		button.clicked.connect(member)
		return button

	def moveCurrentDBB(self):
		self.currentDBB.move(1, 0, 0)
		Signals.get().geometryRebuild.emit(self.currentDBB)		#refresha?

	def setCurrentDBB(self, currentDBB):
		self.currentDBB = currentDBB
		self.setWindowTitle("Move DBB")

def createCommand():
	return DBBCommand()