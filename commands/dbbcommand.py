from PySide2.QtWidgets import QApplication, QMenu
from PySide2.QtWidgets import QDialog, QPushButton,QGridLayout,QToolTip,QCheckBox,QComboBox
from PySide2.QtWidgets import QTreeView,QMainWindow,QVBoxLayout,QHBoxLayout,QSizePolicy
from PySide2.QtWidgets import QTreeWidget,QTreeWidgetItem,QDockWidget,QWidget,QGroupBox
from dbbdir.dbb import DBBProblem,DBB,DBBHullForm, DBBDeck
from dbbdir.dbb import DBBProblem_new,DBBBaseAll,DBBDeckHull,DBBDeckPlate,DBBDeckUnitBase,DBBSegment,DBBCompartment
import os
from PySide2.QtCore import Slot,Qt,QPoint
import dbbdir.dbbmenus as mm
#d3v imports
from commands import Command
from iohandlers import IOHandler
from signals import Signals
from typing import Dict,List, Tuple

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
		self.dbbproblem:DBBProblem_new=0
		self.si=0

		#tools = app.mainFrame.menuTools
		mb = app.mainFrame.menuBar()

		self.menuMain = QMenu("DBB")
		
		self.menuImportFromCsv = self.menuMain.addAction("Import From Csv")
		self.menuImportFromCsv.triggered.connect(self.onImportFromCsv)

		
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
		self._model_ctrl_dock = self._init_model_control()
		self._model_tree_dock,self._model_tree = self._init_treeveiew()


		Signals.get().geometryAdded.connect(self.registerDBB)
		Signals.get().selectionChanged.connect(self.registerSelection)
		self.dbb = 0

	def send_signals(self,geo_add_reb_del:Tuple[List[DBBBaseAll]]):
		(geo_add,geo_reb,geo_del)=geo_add_reb_del
		for geo in geo_del:
			Signals.get().geometryRemoved.emit(geo)
		for geo in geo_add:
			Signals.get().geometryAdded.emit(geo)
		for geo in geo_reb:
			Signals.get().geometryRebuild.emit(geo)

	def _init_treeveiew(self):
		tree_dock = QDockWidget('Model Tree View',self.mainwin)
		tree = QTreeWidget()
		tree_dock.setWidget(tree)
		tree_dock.setFloating(False)
		self.mainwin.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)
		tree.show()
		tree_dock.hide()
		return tree_dock,tree

	def _init_model_control(self):
		ctrl_dock = ModelControlDock('Model Control',self)
		ctrl_dock.setFloating(False)
		self.mainwin.addDockWidget(Qt.LeftDockWidgetArea, ctrl_dock)
		#ctrl_dock.hide()
		return ctrl_dock

	def setProblem(self,dbbproblem):
		self.dbbproblem=dbbproblem
		Signals.get().geometryImported.emit(self.dbbproblem.hull)
		self._model_ctrl_dock.set_dbb_problem(dbbproblem,self._model_tree)
		self._model_tree_dock.show()



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
		self.dbbproblem = DBBProblem_new(fileName)
		self.fsetproblem(self.dbbproblem)
		print(self.dbbproblem)


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

class ModelControlDock(QDockWidget):
	def __init__(self,name, parent:DBBCommand):
		super().__init__(name,parent.mainwin)
		self._dbbcommand:DBBCommand = parent
		self._mainwin = self._dbbcommand.mainwin
		self._tree = None
		self._dbb_problem:DBBProblem_new = None
		widget = QWidget()
		mainLayout = QVBoxLayout()
		current_design_layout = QHBoxLayout()
		side_layout = QHBoxLayout()
		deck_layout = QHBoxLayout()
		bb_layout = QHBoxLayout()
		dbbitemslayout=QGridLayout()

		gbx_current_design = QGroupBox('Current design')
		self._combo_curr_design = QComboBox()
		current_design_layout.addWidget(self._combo_curr_design)
		gbx_current_design.setLayout(current_design_layout)

		gbx_ps_ss = QGroupBox('Show ship side')
		gbx_ps_ss.setLayout(side_layout)



		self._cbx_view_portside = QCheckBox('Port')
		self._cbx_view_starbside = QCheckBox('Starboard')
		side_layout.addWidget(self._cbx_view_portside)
		side_layout.addWidget(self._cbx_view_starbside)

		gbx_components = QGroupBox('Show component')
		gbx_components.setLayout(dbbitemslayout)

		gbx_deck = QGroupBox('Deck items')
		gbx_deck.setLayout(deck_layout)

		gbx_bb = QGroupBox('Building block items')
		gbx_bb.setLayout(bb_layout)

		icmain =0
		ir=0
		self._cbx_view_decks = QCheckBox('Plates')
		deck_layout.addWidget(self._cbx_view_decks)
		self._cbx_view_hullbtwdecks = QCheckBox('Hull')
		deck_layout.addWidget(self._cbx_view_hullbtwdecks)
		dbbitemslayout.addWidget(gbx_deck, ir, icmain)
		ir += 1
		self._cbx_view_segments = QCheckBox('Segments')
		bb_layout.addWidget(self._cbx_view_segments)
		self._cbx_view_compartments = QCheckBox('Compartments')
		bb_layout.addWidget(self._cbx_view_compartments)
		dbbitemslayout.addWidget(gbx_bb, ir, icmain)
		ir += 1

		gbx_ps_ss.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
		gbx_components.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
		widget.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
		mainLayout.addWidget(gbx_current_design)
		mainLayout.addWidget(gbx_ps_ss)
		mainLayout.addWidget(gbx_components)
		widget.setLayout(mainLayout)
		self.setWidget(widget)
		#set events
		self._cbx_view_starbside.stateChanged.connect(self.on_change_cbx_port_starboard)
		self._cbx_view_portside.stateChanged.connect(self.on_change_cbx_port_starboard)
		#set defaults
		self._cbx_view_portside.setChecked(True)
		self._cbx_view_starbside.setChecked(True)
		self._cbx_view_decks.setChecked(True)
		self._cbx_view_hullbtwdecks.setChecked(True)
		self._cbx_view_segments.setChecked(False)
		self._cbx_view_compartments.setChecked(True)


	@property
	def dbb_cmnd(self):
	    return self._dbbcommand

	@property
	def dbb_prob(self):
		return self._dbb_problem
	@property
	def show_portside(self):
	    return self._cbx_view_portside.isChecked()

	@property
	def show_starbside(self):
		return self._cbx_view_starbside.isChecked()

	@property
	def show_deckplate(self):
		return self._cbx_view_decks.isChecked()

	@property
	def show_deckhull(self):
		return self._cbx_view_hullbtwdecks.isChecked()

	@property
	def show_dbbsegm(self):
		return self._cbx_view_segments.isChecked()

	@property
	def show_dbbcomp(self):
		return self._cbx_view_compartments.isChecked()

	def set_dbb_problem(self,dbb_problem,model_tree):
		self._dbb_problem = dbb_problem
		self._tree = model_tree
		self.populate_designs_combo()
		self.show()

	def populate_designs_combo(self):
		cb:QComboBox = self._combo_curr_design
		try:
			cb.currentIndexChanged.disconnect()
		except Exception:
			pass
		cb.clear()
		for key in self.dbb_prob.designs:
			cb.addItem(key)
		cb.setCurrentIndex(0)
		cb.currentIndexChanged.connect(self.on_change_design)
		self.on_change_design(0)


	def on_change_design(self,i):
		try:
			self._tree.itemChanged.disconnect()
		except Exception:
			pass
		id_design = self._combo_curr_design.currentText()
		geo_add_reb_del = self.dbb_prob.set_current_design(id_design)
		self._populate_treeveiew(self._tree, self.dbb_prob)
		self._tree.itemChanged.connect(self._handle_tree_ItemChanged)
		self.dbb_cmnd.send_signals(geo_add_reb_del)

	def on_change_cbx_port_starboard(self, state):
		if self.dbb_prob is not None:
			geo_add_reb_del = self._dbb_problem.set_portside_starboardside_visibility(
				self.show_portside,self.show_starbside)
			self.dbb_cmnd.send_signals(geo_add_reb_del)


	def on_change_cbx_decks(self,state):
		if self.dbb_prob is not None:
			if state == Qt.Checked:
				pass
			else:
				pass

	def _handle_tree_ItemChanged(self, item:QTreeWidgetItem, column):
		if isinstance(item,QTreeWidgetGeometryItem):
			do_show = (item.checkState(column) == Qt.Checked)
			geo:DBBBaseAll = item.geometry
			old_state = geo.show
			if isinstance(geo,DBBDeckUnitBase):
				if isinstance(geo,DBBDeckPlate):
					geo.set_show(do_show and self.show_deckplate)
				elif isinstance(geo,DBBDeckHull):
					geo.set_show(do_show and self.show_deckhull)
				elif isinstance(geo,DBBSegment):
					geo.set_show(do_show and self.show_dbbsegm)
				elif isinstance(geo,DBBCompartment):
					geo.set_show(do_show and self.show_dbbcomp)
				#send signals
				if do_show and (not old_state):
					Signals.get().geometryAdded.emit(geo)
				elif (not do_show) and old_state:
					Signals.get().geometryRemoved.emit(geo)
			else:
				if do_show:
					Signals.get().geometryAdded.emit(geo)
				else:
					Signals.get().geometryRemoved.emit(geo)

	def _populate_treeveiew(self,tree:QTreeWidget,dbbproblem:DBBProblem_new):
		tree.clear()
		#Hull Form
		child = QTreeWidgetGeometryItem(tree, dbbproblem.hull)
		child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
		child.setText(0, "Hull Form")
		child.setCheckState(0, Qt.Checked)

		for deck in dbbproblem.decks:
			#
			node_0 = QTreeWidgetItem(tree)
			node_0.setFlags(node_0.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
			node_0.setText(0, "Deck {}".format(deck.id))
			#
			node_1 = QTreeWidgetItem(node_0)
			node_1.setFlags(node_1.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
			node_1.setText(0, "Hull & Plate {}".format(deck.id))
			#
			#Plate
			child = QTreeWidgetGeometryItem(node_1,deck.plate)
			child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
			child.setText(0, "Plate{}".format(deck.id))
			child.setCheckState(0, Qt.Checked)
			#Hull
			child = QTreeWidgetGeometryItem(node_1, deck.hull)
			child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
			child.setText(0, "Hull{}".format(deck.id))
			child.setCheckState(0, Qt.Checked)
			# Segments
			node_1 = QTreeWidgetItem(node_0)
			node_1.setFlags(node_1.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
			node_1.setText(0, "Segments {}".format(deck.id))
			for segment in deck.segments:
				node_2 = QTreeWidgetGeometryItem(node_1, segment)
				node_2.setFlags(node_2.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
				node_2.setText(0, "Segment {}".format(segment.id))
				for comp in segment.compartments:
					child = QTreeWidgetGeometryItem(node_2, comp)
					child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
					child.setText(0, "Compartment {}".format(comp.id))
					child.setCheckState(0, Qt.Checked)


def createCommand():
	return DBBCommand()