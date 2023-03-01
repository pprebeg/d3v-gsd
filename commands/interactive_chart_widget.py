import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QImage, QPainter, QColor,QColorConstants
from PySide6.QtCore import Qt

import sys

from PySide6.QtCore import Slot, QPointF, Qt
from PySide6.QtCharts import QChart, QChartView, QSplineSeries, QLineSeries, QScatterSeries
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QComboBox, QCheckBox, QLabel,\
     QSplitter, QVBoxLayout,QTableWidget,QTableWidgetItem
from typing import List,Dict,Set

#import rc_markers

def rectangle(point_type, image_size):
    image = QImage(image_size, image_size, QImage.Format_RGB32)
    painter = QPainter()
    painter.begin(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.fillRect(0, 0, image_size, image_size, point_type[2])
    painter.end()
    return image

def triangle(point_type, image_size):
    return QImage(point_type[3]).scaled(image_size, image_size)

def circle(point_type, image_size):
    image = QImage(image_size, image_size, QImage.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter()
    painter.begin(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(point_type[2])
    pen = painter.pen()
    pen.setWidth(0)
    painter.setPen(pen)
    painter.drawEllipse(0, 0, image_size * 0.9, image_size * 0.9)
    painter.end()
    return image

_point_types = [("RedRectangle", rectangle, QColorConstants.Red),
                ("RedCircle", circle, QColorConstants.Red),
                ("BlueRectangle", rectangle, QColorConstants.Blue),
                ("BlueCircle", circle, QColorConstants.Blue),
                ("GreenRectangle", rectangle, QColorConstants.Green),
                ("GreenCircle", circle, QColorConstants.Green),
                ("BlackRectangle", rectangle, QColorConstants.Black),
                ("BlackCircle", circle, QColorConstants.Black),
                ]
_selected_point_types = [("YellowCircle", circle, QColorConstants.Yellow),
                         ("YellowRectangle", rectangle, QColorConstants.Yellow),
                         ("MagentaCircle", circle, QColorConstants.Magenta),
                         ("MagentaRectangle", rectangle, QColorConstants.Magenta),
                         ]
_line_colors = [("Blue", QColorConstants.Blue), ("Black", QColorConstants.Black), ("Red",QColorConstants.Red)]

def point_type(index):
    return _point_types[index]

def selected_point_type(index):
    return _selected_point_types[index]

def line_color(index):
    return _line_colors[index]


def default_light_marker(image_size):
    return rectangle(_point_types[0], image_size)

def default_selected_light_marker(image_size):
    return circle(_selected_point_types[0], image_size)


def get_point_representation(point_type, image_size):
    return point_type[1](point_type, image_size)

def get_selected_point_representation(point_type, image_size):
    return point_type[1](point_type, image_size)

def make_line_color(line_color):
    return line_color[1]

class InteractiveChartWidget(QWidget):
    def __init__(self,parent=None,callback_selected_index=None):
        super(InteractiveChartWidget, self).__init__(parent)
        self.callback_selected_index=callback_selected_index
        self._selected = set()
        self._selected_color = selected_point_type(0)[2]
        self._marker_size = 10.
        self._chart = QChart()
        self._chart.legend().setVisible(False)
        self._chart_view = QChartView(self._chart)
        self._chart_view.setRenderHint(QPainter.Antialiasing)

        control_widget = QWidget()
        control_layout = QGridLayout(control_widget)
        self._char_point_combobox = QComboBox()
        self._char_point_selected_combobox = QComboBox()
        self._line_color_combobox = QComboBox()
        self._show_unselected_points_checkbox = QCheckBox()

        self._char_point = QLabel("Char point: ")
        for tp in _point_types:
            self._char_point_combobox.addItem(tp[0])
        self._char_point_combobox.currentIndexChanged.connect(self.set_light_marker)

        self._char_point_selected = QLabel("Char point selected: ")
        for tp in _selected_point_types:
            self._char_point_selected_combobox.addItem(tp[0])
        self._char_point_selected_combobox.currentIndexChanged.connect(self.set_selected_light_marker)

        self._line_color_label = QLabel("Line color: ")
        for tp in _line_colors:
            self._line_color_combobox.addItem(tp[0])
        self._line_color_combobox.currentIndexChanged.connect(self.set_line_color)


        self._show_unselected_points_label = QLabel("Display unselected points: ")
        self._show_unselected_points_checkbox.setChecked(True)
        self._show_unselected_points_checkbox.stateChanged.connect(self.display_unselected_points)

        self._control_label = QLabel("Marker and Selection Controls")
        self._control_label.setAlignment(Qt.AlignHCenter)
        control_label_font = self._control_label.font()
        control_label_font.setBold(True)
        self._control_label.setFont(control_label_font)
        control_layout.addWidget(self._control_label, 0, 0, 1, 2)
        control_layout.addWidget(self._char_point, 1, 0)
        control_layout.addWidget(self._char_point_combobox, 1, 1)

        control_layout.addWidget(self._char_point_selected, 2, 0)
        control_layout.addWidget(self._char_point_selected_combobox, 2, 1)

        control_layout.addWidget(self._line_color_label, 3, 0)
        control_layout.addWidget(self._line_color_combobox, 3, 1)

        control_layout.addWidget(self._show_unselected_points_label, 4, 0)
        control_layout.addWidget(self._show_unselected_points_checkbox, 4, 1, 1, 2)
        control_layout.setRowStretch(5, 1)

        splitter_chart_control = QSplitter(Qt.Horizontal,self)
        splitter_chart_control.addWidget(self._chart_view)
        splitter_chart_control.addWidget(control_widget)
        splitter_chart_control.setSizes([self.width(),0])

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(splitter_chart_control)
        self.setLayout(mainLayout)

    def set_series(self, series,x_name='',y_name=''):
        self._series = series
        self._series.setMarkerSize(self._marker_size)
        self._series.setLightMarker(default_light_marker(self._marker_size))
        self._series.setSelectedLightMarker(default_selected_light_marker(self._marker_size))

        self._series.clicked.connect(self.toggle_selection)
        self._chart.removeAllSeries()
        self._chart.addSeries(self._series)
        self._chart.createDefaultAxes()
        self._chart.axisX().setTitleText(x_name)
        self._chart.axisY().setTitleText(y_name)

    @Slot(QPointF)
    def toggle_selection(self,point):
        try:
            index = self._series.points().index(point)
            if index != -1:
                self._series.toggleSelection([index])
                if index in self._selected:
                    self._selected.remove(index)
                    if self.callback_selected_index is not None:
                        self.callback_selected_index(index,False)
                else:
                    self._selected.add(index)
                    if self.callback_selected_index is not None:
                        self.callback_selected_index(index, True)

        except ValueError:
            #print('line clicked')
            pass

    @Slot(int)
    def set_light_marker(self,index):
        if self._show_unselected_points_checkbox.isChecked():
            self._series.setLightMarker(get_point_representation(
                point_type(index), self._marker_size))

    @Slot(int)
    def set_selected_light_marker(self,index):
        point_type = selected_point_type(index)
        self._selected_color = point_type[2]
        self._series.setSelectedLightMarker(
            get_selected_point_representation(point_type, self._marker_size))

    @Slot(int)
    def set_line_color(self,index):
        self._series.setColor(make_line_color(line_color(index)))

    @Slot(int)
    def display_unselected_points(self,checkbox_state):
        if checkbox_state:
            self._series.setLightMarker(
                get_point_representation(point_type(self._char_point_combobox.currentIndex()), self._marker_size))
        else:
            self._series.setLightMarker(QImage())


class InteractiveTableChartWidget(QWidget):
    def __init__(self,parent=None,callback_selected_index=None):
        super(InteractiveTableChartWidget, self).__init__(parent)
        self._chart = InteractiveChartWidget(callback_selected_index= self.selected_index_changed)
        self._table = QTableWidget()
        tablechart_splitter = QSplitter(Qt.Horizontal)
        tablechart_splitter.addWidget(self._table)
        tablechart_splitter.addWidget(self._chart)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(tablechart_splitter)
        self.setLayout(mainLayout)
        self._data = None

    def set_data(self,n_cols:int,n_rows:int,data_names:List[str],data_formats:List[str],
                 data,chart_data_pairs:List[int],chart_type = 'line'):
        self._data = data
        self._table.clearContents()
        self._table.setColumnCount(n_cols)
        self._table.setHorizontalHeaderLabels(data_names)
        self._table.setRowCount(n_rows)
        for i in range(n_cols):
            for j in range(n_rows):
                item = QTableWidgetItem(data_formats[i].format(data[j][i]))
                self._table.setItem(j, i, item)
        self._table.show()

        self._chart._chart.removeAllSeries()
        ix = chart_data_pairs[0]
        iy = chart_data_pairs[1]
        if chart_type.lower() == 'spline':
            series = QSplineSeries()
        elif chart_type.lower() == 'scatter' or chart_type.lower() == 'scater':
            series = QScatterSeries()
        else:
            series = QLineSeries()
        for j in range(n_rows):
            series.append(QPointF(data[j][ix], data[j][iy]))
        self._chart.set_series(series,data_names[ix],data_names[iy])
    def selected_index_changed(self,index,do_select):
        n_cols = self._table.columnCount()
        for i in range(n_cols):
            item = self._table.item(index,i)
            item.setSelected(do_select)
        print('selected index = {0}, to select {1}'.format(index,do_select))


def test1(window):
    window.setWindowTitle("Light Markers and Points Selection")
    main_widget = InteractiveTableChartWidget(window)
    n_cols = 2
    data_names = ['x data', 'y data']
    data_formats = ['{:.3f}', '{:.3f}']
    chart_data_pairs = [0, 1]
    data = [[0, 0],
            [0.5, 2.27],
            [1.5, 2.2],
            [3.3, 1.7],
            [4.23, 3.1],
            [5.3, 2.3],
            [6.47, 4.1]]
    n_rows = len(data)
    main_widget.set_data(n_cols, n_rows, data_names, data_formats, data, chart_data_pairs, 'spline')
    window.setCentralWidget(main_widget)

def test2(window):
    window.setWindowTitle("Design Variant Selection")
    main_widget = InteractiveTableChartWidget(window)
    data = [[1,95,15.2,4.0,640.81,7.29,0.504],
            [2,95,16.0,4.2,984.15,9.26,0.532],
            [3,95,16.8,4.4,783.55,9.59,0.541],
            [4,100,15.2,4.0,619.69,6.80,0.504],
            [5,100,16.0,4.2,1164.8,9.39,0.532],
            [6,100,16.8,4.4,796.63,9.90,0.541],
            [7,105,15.2,4.0,612.7,6.66,0.504],
            [8,105,16.0,4.2,966.75,9.12,0.532],
            [9,105,16.8,4.4,824.72,10.80,0.541]]
    for row in data:
        row.append(3254.9/(row[1]*row[2]*row[3]))
    data_names = ['No.', 'L, m', 'B,m', 'T,m', 'R, kN', 'KM, m','Cp','CB']
    n_cols = len(data_names)
    data_formats = ['{:d}', '{:.1f}', '{:.1f}', '{:.1f}', '{:.2f}', '{:.2f}', '{:.3f}', '{:.3f}']
    chart_data_pairs = [4, 5]
    n_rows = len(data)
    main_widget = InteractiveTableChartWidget(window)
    main_widget.set_data(n_cols, n_rows, data_names, data_formats, data, chart_data_pairs, 'line')
    window.setCentralWidget(main_widget)
if __name__ == "__main__":

    a = QApplication(sys.argv)
    window = QMainWindow()
    #test1(window)
    test2(window)
    window.resize(1080, 720)
    window.show()
    sys.exit(a.exec())