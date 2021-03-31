# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AddBlock.ui'
##
## Created by: Qt User Interface Compiler version 5.14.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
	QRect, QSize, QUrl, Qt, SIGNAL)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QFont,
	QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
	QRadialGradient)
from PySide2.QtWidgets import *
import numpy as np
import sys
import os

class AddBlock_Dialog(object):
	def setupUi(self, Dialog):
		if Dialog.objectName():
			Dialog.setObjectName(u"Dialog")
		Dialog.resize(356, 126)
		self.buttonBox = QDialogButtonBox(Dialog)
		self.buttonBox.setObjectName(u"buttonBox")
		self.buttonBox.setGeometry(QRect(250, 60, 81, 61))
		self.buttonBox.setOrientation(Qt.Vertical)
		self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
		self.verticalLayoutWidget = QWidget(Dialog)
		self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
		self.verticalLayoutWidget.setGeometry(QRect(10, 40, 41, 71))
		self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
		self.verticalLayout.setObjectName(u"verticalLayout")
		self.verticalLayout.setContentsMargins(0, 0, 0, 0)
		self.XLabel = QLabel(self.verticalLayoutWidget)
		self.XLabel.setObjectName(u"XLabel")

		self.verticalLayout.addWidget(self.XLabel)

		self.YLabel = QLabel(self.verticalLayoutWidget)
		self.YLabel.setObjectName(u"YLabel")

		self.verticalLayout.addWidget(self.YLabel)

		self.ZLabel = QLabel(self.verticalLayoutWidget)
		self.ZLabel.setObjectName(u"ZLabel")

		self.verticalLayout.addWidget(self.ZLabel)

		self.verticalLayoutWidget_2 = QWidget(Dialog)
		self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
		self.verticalLayoutWidget_2.setGeometry(QRect(70, 40, 160, 71))
		self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
		self.verticalLayout_2.setObjectName(u"verticalLayout_2")
		self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
		self.XInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.XInputLine.setObjectName(u"XInputLine")

		self.verticalLayout_2.addWidget(self.XInputLine)

		self.YInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.YInputLine.setObjectName(u"YInputLine")

		self.verticalLayout_2.addWidget(self.YInputLine)

		self.ZInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.ZInputLine.setObjectName(u"ZInputLine")

		self.verticalLayout_2.addWidget(self.ZInputLine)

		self.DimensionNameLabel = QLabel(Dialog)
		self.DimensionNameLabel.setObjectName(u"DimensionNameLabel")
		self.DimensionNameLabel.setGeometry(QRect(10, 10, 89, 23))
		#buttonbox connections
		self.retranslateUi(Dialog)
		
		self.buttonBox.accepted.connect(Dialog.accept)		#accept and reject are slots
		self.buttonBox.rejected.connect(Dialog.reject)
		
		#Dialog.accept.connect(print("hahaah"))


		QMetaObject.connectSlotsByName(Dialog)
	# setupUi

	
	
	def retranslateUi(self, Dialog):	
		Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Add Block", None))
		self.XLabel.setText(QCoreApplication.translate("Dialog", u"X:", None))
		self.YLabel.setText(QCoreApplication.translate("Dialog", u"Y:", None))
		self.ZLabel.setText(QCoreApplication.translate("Dialog", u"Z:", None))
		self.XInputLine.setText(QCoreApplication.translate("Dialog", u"15", None))
		self.YInputLine.setText(QCoreApplication.translate("Dialog", u"15", None))
		self.ZInputLine.setText(QCoreApplication.translate("Dialog", u"15", None))
		self.DimensionNameLabel.setText(QCoreApplication.translate("Dialog", u"Block Dimensions:", None))
	# retranslateUi

	def getInput(self):
		return np.array([float(self.XInputLine.text()), float(self.YInputLine.text()), float(self.ZInputLine.text())])	#uzima text iz svakog linea i pretvara ih u float i stavlja u array
 
	
	def run(self):			#app = QApplication
		Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
		#self = AddBlock_Dialog()		#ui je sta se sve nalazi u menu
		self.setupUi(Form)		#setappa ui (ocito)
		Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
		if Form.result() == True:  #ako je pritisnut ok
			while True:
				try:
					block_dims = self.getInput()
				except:
					print("numbers only")
					Form.exec()
					if Form.result() == False:
						break
				else:
					print(block_dims)
					return block_dims
					break
	
class AddForm_Dialog(object):
	def setupUi(self, Dialog):
		if Dialog.objectName():
			Dialog.setObjectName(u"Dialog")
		Dialog.resize(240, 74)
		self.buttonBox = QDialogButtonBox(Dialog)
		self.buttonBox.setObjectName(u"buttonBox")
		self.buttonBox.setGeometry(QRect(150, 10, 81, 301))
		self.buttonBox.setOrientation(Qt.Vertical)
		self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
		self.ScaleLabel = QLabel(Dialog)
		self.ScaleLabel.setObjectName(u"ScaleLabel")
		self.ScaleLabel.setGeometry(QRect(10, 10, 61, 21))
		self.ScaleInputLine = QLineEdit(Dialog)
		self.ScaleInputLine.setObjectName(u"ScaleInputLine")
		self.ScaleInputLine.setGeometry(QRect(10, 40, 113, 20))

		self.retranslateUi(Dialog)
		self.buttonBox.accepted.connect(Dialog.accept)
		self.buttonBox.rejected.connect(Dialog.reject)

		QMetaObject.connectSlotsByName(Dialog)
	# setupUi

	def retranslateUi(self, Dialog):
		Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Add Form", None))
		self.ScaleLabel.setText(QCoreApplication.translate("Dialog", u"Form Scale:", None))
		self.ScaleInputLine.setText(QCoreApplication.translate("Dialog", u"5", None))
	# retranslateUi
	
	def getInput(self):
		return float(self.ScaleInputLine.text())
	
	def run(self):
		Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
		#self = AddForm_Dialog()		#ui je sta se sve nalazi u menu
		self.setupUi(Form)		#setappa ui (ocito)
		Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
		if Form.result() == True:  #ako je pritisnut ok
			while True:
				try:
					scale = self.getInput()
				except:
					print("numbers only")
					Form.exec()
					if Form.result() == False:
						break
				else:
					print(scale)
					return scale
					break

class Move_Dialog(object):
	def setupUi(self, Dialog):
		if Dialog.objectName():
			Dialog.setObjectName(u"Dialog")
		Dialog.resize(236, 140)
		self.buttonBox = QDialogButtonBox(Dialog)
		self.buttonBox.setObjectName(u"buttonBox")
		self.buttonBox.setGeometry(QRect(10, 100, 221, 41))
		self.buttonBox.setOrientation(Qt.Horizontal)
		self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
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
		return np.array([float(self.DeltaXInputLine.text()), float(self.DeltaYInputLine.text()), float(self.DeltaZInputLine.text())])
	
	def run(self):
		Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
		#self = AddForm_Dialog()		#ui je sta se sve nalazi u menu
		self.setupUi(Form)		#setappa ui (ocito)
		Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
		if Form.result() == True:  #ako je pritisnut ok
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

					
					
					
class SetPosition_Dialog(object):
	def setupUi(self, Dialog):
		if Dialog.objectName():
			Dialog.setObjectName(u"Dialog")
		Dialog.resize(248, 148)
		self.buttonBox = QDialogButtonBox(Dialog)
		self.buttonBox.setObjectName(u"buttonBox")
		self.buttonBox.setGeometry(QRect(50, 110, 171, 41))
		self.buttonBox.setOrientation(Qt.Horizontal)
		self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
		self.verticalLayoutWidget = QWidget(Dialog)
		self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
		self.verticalLayoutWidget.setGeometry(QRect(10, 20, 41, 80))
		self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
		self.verticalLayout.setObjectName(u"verticalLayout")
		self.verticalLayout.setContentsMargins(0, 0, 0, 0)
		self.SetPositionXLabel = QLabel(self.verticalLayoutWidget)
		self.SetPositionXLabel.setObjectName(u"SetPositionXLabel")

		self.verticalLayout.addWidget(self.SetPositionXLabel)

		self.SetPositionYLabel = QLabel(self.verticalLayoutWidget)
		self.SetPositionYLabel.setObjectName(u"SetPositionYLabel")

		self.verticalLayout.addWidget(self.SetPositionYLabel)

		self.SetPositionZLabel = QLabel(self.verticalLayoutWidget)
		self.SetPositionZLabel.setObjectName(u"SetPositionZLabel")

		self.verticalLayout.addWidget(self.SetPositionZLabel)

		self.verticalLayoutWidget_2 = QWidget(Dialog)
		self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
		self.verticalLayoutWidget_2.setGeometry(QRect(60, 20, 160, 80))
		self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
		self.verticalLayout_2.setObjectName(u"verticalLayout_2")
		self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
		self.SetPositionXInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.SetPositionXInputLine.setObjectName(u"SetPositionXInputLine")

		self.verticalLayout_2.addWidget(self.SetPositionXInputLine)

		self.SetPositionYInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.SetPositionYInputLine.setObjectName(u"SetPositionYInputLine")

		self.verticalLayout_2.addWidget(self.SetPositionYInputLine)

		self.SetPositionZInputLine = QLineEdit(self.verticalLayoutWidget_2)
		self.SetPositionZInputLine.setObjectName(u"SetPositionZInputLine")

		self.verticalLayout_2.addWidget(self.SetPositionZInputLine)


		self.retranslateUi(Dialog)
		self.buttonBox.accepted.connect(Dialog.accept)
		self.buttonBox.rejected.connect(Dialog.reject)

		QMetaObject.connectSlotsByName(Dialog)
	# setupUi

	def retranslateUi(self, Dialog):
		Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Set Position", None))
		self.SetPositionXLabel.setText(QCoreApplication.translate("Dialog", u"Set X:", None))
		self.SetPositionYLabel.setText(QCoreApplication.translate("Dialog", u"Set Y:", None))
		self.SetPositionZLabel.setText(QCoreApplication.translate("Dialog", u"Set Z:", None))
		self.SetPositionXInputLine.setText(QCoreApplication.translate("Dialog", u"0", None))
		self.SetPositionYInputLine.setText(QCoreApplication.translate("Dialog", u"0", None))
		self.SetPositionZInputLine.setText(QCoreApplication.translate("Dialog", u"0", None))
	# retranslateUi


	def getInput(self):
		return np.array([float(self.SetPositionXInputLine.text()), float(self.SetPositionYInputLine.text()), float(self.SetPositionZInputLine.text())])
	
	def run(self):
		Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
		#self = AddForm_Dialog()		#ui je sta se sve nalazi u menu
		self.setupUi(Form)		#setappa ui (ocito)
		Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
		if Form.result() == True:  #ako je pritisnut ok
			while True:
				try:
					new_position = self.getInput()
				except:
					print("numbers only")
					Form.exec()
					if Form.result() == False:
						break
				else:
					print(new_position)
					return new_position
					break



class ImportFromCsvMenu_Dialog(object):
	def setupUi(self, Dialog):
		if Dialog.objectName():
			Dialog.setObjectName(u"Dialog")
		Dialog.resize(240, 102)
		self.buttonBox = QDialogButtonBox(Dialog)
		self.buttonBox.setObjectName(u"buttonBox")
		self.buttonBox.setGeometry(QRect(20, 60, 221, 41))
		self.buttonBox.setOrientation(Qt.Horizontal)
		self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
		self.PathLabel = QLabel(Dialog)
		self.PathLabel.setObjectName(u"PathLabel")
		self.PathLabel.setGeometry(QRect(10, 10, 121, 16))
		self.PathLine = QLineEdit(Dialog)
		self.PathLine.setObjectName(u"PathLine")
		self.PathLine.setGeometry(QRect(10, 30, 200, 20))

		self.retranslateUi(Dialog)
		self.buttonBox.accepted.connect(Dialog.accept)
		self.buttonBox.rejected.connect(Dialog.reject)

		QMetaObject.connectSlotsByName(Dialog)
	# setupUi

	def retranslateUi(self, Dialog):
		Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Import from csv", None))
		self.PathLabel.setText(QCoreApplication.translate("Dialog", u"Path to Folder:", None))
		self.PathLine.setText("")
	# retranslateUi

	def getInput(self):
		return self.PathLine.text()
	
	def run(self):
		Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
		#self = AddForm_Dialog()		#ui je sta se sve nalazi u menu
		self.setupUi(Form)		#setappa ui (ocito)
		Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
		if Form.result() == True:  #ako je pritisnut ok
			while True:
				path = self.getInput()
				if os.path.isdir(path) == False:
					print("Path not valid.")
					Form.exec()
					if Form.result() == False:
						break
				else:
					print(path)
					return path
					break






	
#iz techinfo videa:
if __name__ == "__main__":
	app = QApplication(sys.argv) #app moramo importat
	a = SetPosition_Dialog()
	#a = Move_Dialog()
	a.run()		
	#pass
	#	Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
#	ui = AddBlock_Dialog()		#ui je sta se sve nalazi u menu
#	ui.setupUi(Form)		#setappa ui (ocito)
#	Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)
#	#app.run
#	#result = Form.result()
#	while Form.result() == True:
#		Form.exec()
#		print("array koji ide u dims")
		
		
#	Form = QDialog()		#Form je oblik ; Qwidget je emptybox a Qdialogue je menu sa ok i cancel
#	ui = AddBlock_Dialog()		#ui je sta se sve nalazi u menu
#	ui.setupUi(Form)		#setappa ui (ocito)
#	Form.exec()				#show je preview pogledaj modal dialogue u dokumentaciji (modalni blokiraju acsess ostatku aplikacije dok nije zavrsena)	#pokrece novi menu
#	if Form.result() == True:  #ako je pritisnut ok
#		while True:
#			try:
#				block_dims = ui.getInput()
#			except:
#				print("numbers only")
#				Form.exec()
#			else:
#				break
#		print(block_dims)
			
		
		
		
		
		
		
		
		