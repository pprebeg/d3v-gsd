from PySide2.QtWidgets import QApplication, QMenu
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout,QToolTip
from PySide2.QtWidgets import QTreeView,QMainWindow
from PySide2.QtWidgets import QTreeWidget,QTreeWidgetItem,QDockWidget
from dbbdir.dbb import DBBProblem,DBB,DBBHullForm, DBBDeck
import os
from PySide2.QtCore import Slot,Qt,QPoint
import dbbdir.dbbmenus as mm
#d3v imports
from commands import Command
from iohandlers import IOHandler
from signals import Signals

class QTreeWidgetGeometryItem(QTreeWidgetItem):
	def __init__(self,parent,geo):
		super().__init__(parent)
		self._geo = geo
	@property
	def geometry(self):
	    return self._geo

class DBBCommand(Command):
	def __init__(self):
		super().__init__()
		app = QApplication.instance()
		importer=DBBImporter(self.setProblem)
		app.registerIOHandler(importer)
		self.mainwin:QMainWindow = app.mainFrame
		self._tree: QTreeView = self.mainwin.window.findChild(QTreeView, "geometryTree")
		self._tree.hide()
		self.dbbprop = DialogDBBProps(self.mainwin)
		self.dbbproblem:DBBProblem=0
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
		self._model_tree_dock,self._model_tree = self._init_treeveiew()

		Signals.get().geometryAdded.connect(self.registerDBB)
		Signals.get().selectionChanged.connect(self.registerSelection)
		self.dbb = 0

	def _handleItemChanged(self, item:QTreeWidgetItem, column):
		if isinstance(item,QTreeWidgetGeometryItem):
			do_show = (item.checkState(column) == Qt.Checked)
			if do_show:
				Signals.get().geometryAdded.emit(item.geometry)
			else:
				Signals.get().geometryRemoved.emit(item.geometry)


	def _init_treeveiew(self):
		tree_dock = QDockWidget('Model Tree View',self.mainwin)
		tree = QTreeWidget()
		tree_dock.setWidget(tree)
		tree_dock.setFloating(False)
		self.mainwin.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)
		tree.show()
		tree_dock.hide()
		return tree_dock,tree
	def _populate_treeveiew(self,dbbproblem:DBBProblem):
		self._model_tree.clear()
		#Hull Form
		child = QTreeWidgetGeometryItem(self._model_tree, dbbproblem.hull)
		child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
		child.setText(0, "Hull Form")
		child.setCheckState(0, Qt.Checked)
		#Hull Form Decks
		parent = QTreeWidgetItem(self._model_tree)
		parent.setText(0, "Hull Deck Segments")
		parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
		for deck in dbbproblem.decks:
			node_1 = QTreeWidgetItem(parent)
			node_1.setFlags(node_1.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
			node_1.setText(0, "Deck Segment {}".format(deck.id))
			#node_1.setCheckState(0, Qt.Checked)
			#Deck
			child = QTreeWidgetGeometryItem(node_1,deck)
			child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
			child.setText(0, "Deck{}".format(deck.id))
			child.setCheckState(0, Qt.Checked)

			#Hull Form segment from deck to Upper deck


		#
		parent = QTreeWidgetItem(self._model_tree)
		parent.setText(0, "Building Blocks")
		parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

		for key,dbblist in dbbproblem.segment_blocks.items():
			node_1 = QTreeWidgetItem(parent)
			node_1.setFlags(node_1.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
			node_1.setText(0, "Segment {}".format(key))
			for dbb in dbblist:
				child = QTreeWidgetGeometryItem(node_1, dbb)
				child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
				child.setText(0, "Block {}".format(dbb.id))
				child.setCheckState(0, Qt.Checked)
		self._model_tree.itemChanged.connect(self._handleItemChanged)
		self._model_tree_dock.show()

	def setProblem(self,dbbproblem):
		self.dbbproblem=dbbproblem
		self._populate_treeveiew(dbbproblem)

	@Slot()
	def registerDBB(self, dbbproblem):
		if isinstance(dbbproblem,DBBProblem):
			self.dbbproblem=dbbproblem
			self.menuMain.setEnabled(True)


	@Slot()
	def registerSelection(self, si):
		self.si=si
		if si.isEmpty():
			pass
		else:
			currDBB = self.si.getGeometry()
			print(self.dbbproblem)
			if isinstance(currDBB, DBB):
				pos: QPoint = self.mainwin.pos()
				pos.setX(pos.x() + self.mainwin.glWin.dragInfo.wStartPos.x() + 20)
				pos.setY(pos.y() + self.mainwin.glWin.size().height() - self.mainwin.glWin.dragInfo.wStartPos.y())
				msg = currDBB.get_info()
				QApplication.instance().clipboard().setText(str(msg))
				QToolTip.showText(pos, msg, msecShowTime=10)

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
	def __init__(self,func_set_problem):
		super().__init__()
		self.fsetproblem=func_set_problem #function for seting the problem of parent object

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