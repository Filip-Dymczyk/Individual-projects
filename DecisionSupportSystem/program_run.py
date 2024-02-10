import numpy as np
import matplotlib
from read_data import read_excel_to_dataframe
from topsis_algorithm import Topsis
from spcs_algorithm import SPCS
from rsm_algorithm import RSM
from typing import Optional, Union, List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QLabel, QPushButton, QWidget, QHBoxLayout, \
    QDesktopWidget, QTableWidget, QTableWidgetItem, QScrollArea, QComboBox, QCheckBox, QMessageBox
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)

matplotlib.use('QtAgg')


# Class for 2D plots - SPCS method:
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=11, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


# Class for 3D plots - RSM method:
class MplCanvas3D(FigureCanvas):
    def __init__(self, parent=None, width=11, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super(MplCanvas3D, self).__init__(fig)


# Closing window:
class CloseWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Exit")
        self.setFixedSize(250, 100)
        center(self)

        # Layouts:
        final_layout = QVBoxLayout()
        label_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Label:
        label = QLabel("Do you want to leave?")
        label.setAlignment(Qt.AlignCenter)

        # Font:
        font = QFont()
        font.setFamily('Calibri')
        font.setPointSize(16)

        self.setFont(font)

        font.setPointSize(12)

        label.setFont(font)

        label_layout.addWidget(label)

        final_layout.addLayout(label_layout)
        final_layout.addLayout(button_layout)

        # Buttons:
        b_yes = QPushButton("Yes")
        b_no = QPushButton("No")

        font.setPointSize(10)

        b_yes.setFont(font)
        b_no.setFont(font)

        b_yes.clicked.connect(self.__close)
        b_no.clicked.connect(self.__back)

        button_layout.addWidget(b_yes)
        button_layout.addWidget(b_no)

        self.setLayout(final_layout)

    # Leaving the application - closing all windows:
    def __close(self) -> None:
        QApplication.closeAllWindows()

    # Returning to main window:
    def __back(self) -> None:
        self.close()


# Main window - algorithm selection:
class Window(QMainWindow):
    def __init__(self):
        # Initialize parent:
        super().__init__()

        # Setting title and size:
        self.setWindowTitle("Multi-Criteria Analysis")
        self.setFixedSize(475, 125)
        center(self)

        # Exit window initialization:
        self.c_w = None

        # Font
        font = QFont()
        font.setFamily('Calibri')
        font.setPointSize(16)

        # Final layout:
        final_layout = QVBoxLayout()

        # Labels
        label = QLabel("Choose decision support method:")
        label.setFont(font)

        final_layout.addWidget(label, alignment=Qt.AlignCenter)

        # Initializing button layout:
        button_layout = QHBoxLayout()

        # Buttons:
        b1 = QPushButton("TOPSIS")
        b2 = QPushButton("RSM")
        b3 = QPushButton("SPCS")
        b4 = QPushButton("Exit")

        buttons = [b1, b2, b3, b4]

        for i, button in enumerate(buttons):
            if i < len(buttons) - 1:
                button.clicked.connect(self.__data_acq)
                button.setFont(font)
                button_layout.addWidget(button)
            else:
                button.clicked.connect(self.__leave)
                font.setPointSize(12)
                button.setFont(font)
                final_layout.addLayout(button_layout)
                final_layout.addWidget(button, alignment=Qt.AlignRight)

        w = QWidget()
        w.setLayout(final_layout)
        self.setCentralWidget(w)

    # Move to data acquisition window:
    def __data_acq(self) -> None:
        button = self.sender()
        self.__data_acq_window = DataAcqWindow(button.text(), self)
        self.hide()
        self.__data_acq_window.show()

    # Show exit window:
    def __leave(self) -> None:
        if not self.c_w:
            self.c_w = CloseWindow()
        self.c_w.show()


# Window for data acquisition:
class DataAcqWindow(QWidget):
    def __init__(self, method: str, parent: QMainWindow) -> None:
        super().__init__()

        # Remembering main window (parent):
        self.__parent = parent

        # Title and size:
        self.setWindowTitle("Data acquisition")
        self.setFixedSize(275, 125)

        # Initializing exit window:
        self.c_w = None

        # Method selection:
        self.__method = method

        # Font
        font = QFont()
        font.setFamily('Calibri')
        font.setPointSize(14)

        # Widgets:
        label = QLabel("Enter your data set and confirm.")
        button = QPushButton("Confirm")

        # Connecting button to data display window:
        button.clicked.connect(self.__read_data_on_button_click)

        # Exit button:
        b_close = QPushButton("Exit")
        b_close.clicked.connect(self.__leave)

        # Font
        label.setFont(font)
        button.setFont(font)

        font.setPointSize(12)
        b_close.setFont(font)

        # Layouts:
        final_layout = QVBoxLayout()
        layout_main = QVBoxLayout()
        layout_main.addSpacing(10)
        layout_leave = QHBoxLayout()

        # Adding widgets:
        layout_main.addWidget(label, alignment=Qt.AlignCenter)
        layout_main.addWidget(button, alignment=Qt.AlignCenter)
        layout_leave.addWidget(b_close, alignment=Qt.AlignRight)

        # Adding to final layout:
        final_layout.addLayout(layout_main)
        final_layout.addLayout(layout_leave)

        # Window tailoring:
        center(self)
        self.setLayout(final_layout)

    # Move to data display window:
    def __read_data_on_button_click(self) -> None:

        # Reading from provided Excel file:
        self.evaluation_matrix = read_excel_to_dataframe()

        # Checking viability:
        if not self.__data_viable():

            # Setting up msg box:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Invalid data!")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        else:
            self.__data_show_window = DataShowWindow(self.evaluation_matrix, self.__method, self.__parent)
            self.close()
            self.__data_show_window.show()

    # Checking data viability:
    def __data_viable(self) -> bool:
        for vec in self.evaluation_matrix:
            if np.any(np.logical_not(np.vectorize(np.isreal)(vec))):
                return False
        return True

    # Show exit window:
    def __leave(self) -> None:
        if not self.c_w:
            self.c_w = CloseWindow()
        self.c_w.show()


# Window for data display:
class DataShowWindow(QWidget):
    def __init__(self, evalutaion_matrix: np.ndarray, method: str, parent: QMainWindow) -> None:
        super().__init__()

        # Remembering input arguments:
        self.evaluation_matrix = evalutaion_matrix
        self.__parent = parent
        self.__method = method

        shape = self.evaluation_matrix.shape

        # Title:
        self.setWindowTitle("Data display")

        # Layouts:
        label_layout = QVBoxLayout()
        final_layout = QVBoxLayout()

        checkbox_layout = QHBoxLayout()
        combobox_layout = QHBoxLayout()

        # Checkboxes and comboboxes (False = Min) statuses:
        self.checkbox_status = [False for _ in range(shape[1])]
        self.combobox_status = [False for _ in range(shape[1])]

        # Creating table with data:
        table = QTableWidget(self)
        table.setRowCount(shape[0])
        table.setColumnCount(shape[1])

        # Creating headers arrays:
        horizontal_headers = []
        vertical_headers = []

        # Entering data into created table:
        for row in range(table.rowCount()):
            vertical_headers.append(f"M{row + 1}")
            for col in range(table.columnCount()):
                item = QTableWidgetItem(f"{self.evaluation_matrix[row, col]}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)

                if row == 0:
                    horizontal_headers.append(f"Param {col + 1}")
                    checkbox = QCheckBox(f"Param {col + 1}")
                    checkbox.setObjectName(f"{col}")
                    checkbox.stateChanged.connect(self.__checkbox_state_changed)
                    checkbox_layout.addWidget(checkbox)

                    combobox = QComboBox()
                    combobox.addItems(["Min", "Max"])
                    combobox.setObjectName(f"{col}")
                    combobox.currentIndexChanged.connect(self.__combobox_index_changed)
                    combobox_layout.addWidget(combobox)

        # Acquiring approximate size of the table:
        window_width = sum(table.columnWidth(col) for col in range(table.columnCount()))
        window_height = sum(table.rowHeight(row) for row in range(5 if table.rowCount() > 5 else table.rowCount())) + 40

        window_width += int(0.75 * table.columnWidth(0))
        window_height *= 2
        if window_height < 100:
            window_height = 150
        if window_width < 100:
            window_width = 100

        # Setting window size:
        self.setFixedWidth(window_width)
        self.setFixedHeight(window_height)

        # Setting headers to table:
        table.setHorizontalHeaderLabels(horizontal_headers)
        table.setVerticalHeaderLabels(vertical_headers)

        # Assigning table to scroll area:
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(table)

        # Font
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)

        # Metrics label and combobox:
        label_metrics = QLabel("Select desired metric for calculations:")
        label_metrics.setFont(font)

        self.metrics_combobox = QComboBox()
        self.metrics_combobox.addItems(["Euclidean", "Chebyshev"])

        # Adding widgets and layouts:
        final_layout.addWidget(scroll_area)
        final_layout.addLayout(checkbox_layout)
        final_layout.addLayout(combobox_layout)
        final_layout.addWidget(label_metrics, alignment=Qt.AlignCenter)
        final_layout.addWidget(self.metrics_combobox, alignment=Qt.AlignCenter)

        # Connecting button to weights or results window (based on selected method):
        button = QPushButton("Continue")
        if self.__method == "TOPSIS":
            button.clicked.connect(self.__weights_window)
        elif self.__method in ["SPCS", "RSM"]:
            button.clicked.connect(self.__show_res)
        button.setFont(font)

        # Button layout:
        button_layout = QVBoxLayout()
        button_layout.addWidget(button, alignment=Qt.AlignRight)

        # Adding to final layout:
        final_layout.addLayout(label_layout)
        final_layout.addLayout(button_layout)

        # Window centering and final layout setting:
        center(self)
        self.setLayout(final_layout)

    # Detecting checking/unchecking of checkbox:
    def __checkbox_state_changed(self, state) -> None:
        checkbox = self.sender()
        checkbox_id, checkbox_state = int(checkbox.objectName()), checkbox.isChecked()
        self.checkbox_status[checkbox_id] = checkbox_state

    # Detecting combobox state change:
    def __combobox_index_changed(self, index) -> None:
        combobox = self.sender()
        combobox_id, min_or_max = int(combobox.objectName()), combobox.currentText()

        if min_or_max == "Min":
            self.combobox_status[combobox_id] = False
        elif min_or_max == "Max":
            self.combobox_status[combobox_id] = True

    # Moving to weights window - only for TOPSIS:
    def __weights_window(self) -> None:
        n_criteria = self.checkbox_status.count(True)

        # Check for at least 2 criteria selected:
        if n_criteria < 2:
            # Setting up msg box:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("You must select at least two criterias!")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            self.weights_window = WeightsWindow(self.__method, self.evaluation_matrix, self.checkbox_status,
                                                self.combobox_status, self.metrics_combobox.currentText(),
                                                self.__parent, n_criteria)
            self.close()
            self.weights_window.show()

    # Moving to results window immediately - SPCS and RSM:
    def __show_res(self) -> None:
        self.__res_window = None
        n_criteria = self.checkbox_status.count(True)

        if self.__method == "SPCS":
            if n_criteria == 2:
                self.__res_window = ResultsShowWindow(self.__method, self.evaluation_matrix,
                                                      self.checkbox_status, self.combobox_status, None,
                                                      self.metrics_combobox.currentText(), self.__parent)
            else:
                # Setting up msg box:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(f"Only two criteria at a time are are allowed for this method ({self.__method})")
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
        elif self.__method == "RSM":
            if n_criteria == 3:
                self.__res_window = ResultsShowWindow(self.__method, self.evaluation_matrix,
                                                      self.checkbox_status, self.combobox_status, None,
                                                      self.metrics_combobox.currentText(), self.__parent)
            else:
                # Setting up msg box:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(f"Only three criteria at a time are are allowed for this method ({self.__method})")
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
        if self.__res_window:
            self.close()
            self.__res_window.show()
        else:
            return


# Helper window to select weights for criteria - only in TOPSIS:
class WeightsWindow(QWidget):
    def __init__(self, method: str, evaluation_matrix: np.ndarray, selected_cols: List[bool],
                 min_max: List[bool], metric: str, parent: QMainWindow, n_criteria: int) -> None:
        super().__init__()

        # Remembering input arguments:
        self.__method = method
        self.evaluation_matrix = evaluation_matrix
        self.selected_cols = selected_cols
        self.min_max = min_max
        self.metric = metric
        self.__parent = parent

        self.setWindowTitle("Weights")

        final_layout = QVBoxLayout()

        # Creating appropriate table:
        self.table = QTableWidget()
        self.table.setRowCount(1)
        self.table.setColumnCount(n_criteria + 1)

        # Creating headers arrays:
        horizontal_headers = [f"w{col + 1}" for col, elem in enumerate(self.selected_cols) if elem]

        # Evenly distributed weights (will be changeable):
        value = 1 / n_criteria

        # Entering data into created table:
        for col in range(n_criteria):
            item = QTableWidgetItem(f"{round(value, 5)}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(0, col, item)

            if col == n_criteria - 1:
                horizontal_headers.append(f"Sum")
                item = QTableWidgetItem(f"{1}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(0, col + 1, item)

        self.table.setHorizontalHeaderLabels(horizontal_headers)
        self.table.itemChanged.connect(self.__item_changed)

        # Acquiring approximate size of the table:
        window_width = sum(self.table.columnWidth(col) for col in range(n_criteria + 1))

        window_width += int(0.75 * self.table.columnWidth(0))

        if window_width < 100:
            window_width = 100

        window_height = 185

        # Setting window size:
        self.setFixedWidth(window_width)
        self.setFixedHeight(window_height)

        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)

        label1 = QLabel("Adjust weights corresponding to criteria.")
        label2 = QLabel("Only weights between 0 and 1 are allowed.")
        label3 = QLabel("Sum has to be equal to 1 at all times.")
        label1.setFont(font)

        button = QPushButton()
        button.setText("Continue")
        button.clicked.connect(self.__results_window)

        font.setPointSize(12)
        button.setFont(font)
        label2.setFont(font)
        label3.setFont(font)

        final_layout.addWidget(label1, alignment=Qt.AlignCenter)
        final_layout.addWidget(self.table)
        final_layout.addWidget(label2, alignment=Qt.AlignCenter)
        final_layout.addWidget(label3, alignment=Qt.AlignCenter)
        final_layout.addWidget(button, alignment=Qt.AlignRight)

        center(self)
        self.setLayout(final_layout)

    # Handle item change in table - check viability and change sum:
    def __item_changed(self, item) -> None:

        # Disconnecting itemChanged signal to avoid infinite recursion:
        self.table.itemChanged.disconnect(self.__item_changed)

        col = item.column()

        # Criteria count:
        n_col = self.table.columnCount() - 1

        # Checking for entering something different from a number:
        try:
            item_value = float(item.text())
        except ValueError:
            self.__msg_box(1)
            return

        # Allowing values greater than 0 and lesser than 1:
        if item_value >= 1 or item_value <= 0:
            self.__msg_box(1)
            return

        # Calculating new sum:
        sum = 0
        for i in range(n_col):
            if i != col:
                sum += float(self.table.item(0, i).text())
        sum += item_value

        sum_item = QTableWidgetItem(f"{round(sum, 5)}")
        sum_item.setFlags(sum_item.flags() & ~Qt.ItemIsEditable)
        sum_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, n_col, sum_item)

        # Connecting the signal:
        self.table.itemChanged.connect(self.__item_changed)

    # Function managing msg box:
    def __msg_box(self, idx: int) -> None:

        # If we are coming from clicking continue button then the method is connected:
        if idx == 2:
            self.table.itemChanged.disconnect(self.__item_changed)

        # First situation - wrong value entered:
        if idx == 1:
            # Setting up msg box:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Only numbers from 0 to 1 interval are allowed!")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        # Second situation - sum >= 1:
        elif idx == 2:
            # Setting up msg box:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("The sum of all weights should be equal to 1!")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        # Criteria count:
        n_col = self.table.columnCount() - 1

        # Restoring the change:
        weight = 1 / n_col
        for i in range(n_col):
            item = QTableWidgetItem(f"{round(weight, 5)}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(0, i, item)

        sum_item = QTableWidgetItem(f"{1}")
        sum_item.setFlags(sum_item.flags() & ~Qt.ItemIsEditable)
        sum_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, n_col, sum_item)

        self.table.itemChanged.connect(self.__item_changed)

    # Moving to results window:
    def __results_window(self) -> None:
        weights = []

        for col in range(self.table.columnCount()):
            weights.append(float(self.table.item(0, col).text()))
        if weights[-1] != 1:
            self.__msg_box(2)
            return

        self.results_window = ResultsShowWindow(self.__method, self.evaluation_matrix,
                                                self.selected_cols, self.min_max, np.array(weights[:-1]),
                                                self.metric, self.__parent)

        self.close()
        self.results_window.show()


# Window for results display:
class ResultsShowWindow(QWidget):
    def __init__(self, method: str, evaluation_matrix: np.ndarray, selected_cols: List[int],
                 min_max: List[bool], weights_matrix: Optional[np.ndarray], metric: str, parent: QMainWindow) -> None:
        super().__init__()

        # Remembering parent method and selected method:
        self.__parent = parent
        self.__method = method

        self.setWindowTitle("Results")
        self.weights = weights_matrix
        self.metric = metric

        # Deleting not used criteria:
        to_del_columns = np.where(np.array(selected_cols) == False)[0]
        self.evaluation_matrix = np.delete(evaluation_matrix, to_del_columns, axis=1)
        self.min_max = np.delete(np.array(min_max), to_del_columns)

        # Initializing algorithm object object:
        self.alg_obj = None

        # Creating algorithm objects:
        if self.__method == "TOPSIS":
            self.alg_obj = Topsis(self.evaluation_matrix, self.weights, self.min_max, self.metric)
        elif self.__method == "SPCS":
            self.alg_obj = SPCS(self.evaluation_matrix, self.min_max, self.metric)
        elif self.__method == "RSM":
            self.alg_obj = RSM(self.evaluation_matrix, self.min_max, self.metric)

        # Final layout:
        final_layout = QVBoxLayout()

        # Scroll areas and labels lists:
        scroll_areas = []
        labels = []

        # Window sizes to calculate:
        window_height = 0
        window_width = 0

        # Font:
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(16)

        # Calculating the selected algorithm:
        self.alg_obj.calc()

        # TOPSIS:
        if self.__method == "TOPSIS":
            # Getting needed results:
            matrix_normalized_decision = self.alg_obj.normalized_decision
            matrix_normalized_weighted = self.alg_obj.weighted_normalized
            shape = matrix_normalized_decision.shape

            # Creating tables with data:
            table_normalized_decision = QTableWidget(self)
            table_normalized_weighted = QTableWidget(self)
            table_best_alternatives = QTableWidget(self)
            table_worst_alternatives = QTableWidget(self)
            table_best_distance = QTableWidget(self)
            table_worst_distance = QTableWidget(self)

            tables = [table_normalized_decision, table_normalized_weighted, table_best_alternatives,
                      table_worst_alternatives, table_best_distance, table_worst_distance]

            # Loop over tables to set their size:
            for i, table in enumerate(tables):
                table.setColumnCount(shape[1])
                if i < 2:
                    table.setRowCount(shape[0])
                else:
                    table.setRowCount(1)

            # Creating headers arrays:
            horizontal_headers = []
            vertical_headers = []

            # Creating texts and labels for tables:
            texts = ["Normalized decision matrix", "Normalized decision matrix with weights", "Best alternative",
                     "Worst alternative", "Best distance", "Worst distance"]

            for text in texts:
                label = QLabel(text)
                label.setFont(font)
                labels.append(label)

            # Entering data into created tables:
            for row in range(table_normalized_decision.rowCount()):
                vertical_headers.append(f"M{row + 1}")
                for col in range(table_normalized_decision.columnCount()):
                    item = QTableWidgetItem(f"{round(matrix_normalized_decision[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_normalized_decision.setItem(row, col, item)

                    item = QTableWidgetItem(f"{round(matrix_normalized_weighted[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_normalized_weighted.setItem(row, col, item)

                    if row == 0:
                        horizontal_headers.append(f"Param {col + 1}")

                        # Setting 1 row tables:
                        matrices = [self.alg_obj.best_alternatives, self.alg_obj.worst_alternatives,
                                    self.alg_obj.best_distance, self.alg_obj.best_distance]
                        tables_in = [table_best_alternatives, table_worst_alternatives,
                                     table_best_distance, table_worst_distance]

                        for matrix_alt, table in zip(matrices, tables_in):
                            item = QTableWidgetItem(f"{round(matrix_alt[col], 4)}")
                            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                            item.setTextAlignment(Qt.AlignCenter)
                            table.setItem(row, col, item)

            # Setting headers to tables:
            for i, table in enumerate(tables):
                table.setHorizontalHeaderLabels(horizontal_headers)
                if i < 2:
                    table.setVerticalHeaderLabels(vertical_headers)

            # Assigning tables to scroll area:
            for i, table in enumerate(tables):
                scroll_area = QScrollArea(self)
                scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(table)
                scroll_areas.append(scroll_area)

            # Acquiring approximate width of the window:
            window_width = sum(table_best_alternatives.columnWidth(col)
                               for col in range(table_best_alternatives.columnCount()))

            window_width += int(0.75 * table_best_alternatives.columnWidth(0))
            control_width = labels[1].fontMetrics().boundingRect(labels[1].text()).width() + 30
            if window_width < control_width:
                window_width = control_width

            # Acquiring approximate height of the window:
            for table, label in zip(tables, labels):
                window_height += sum(table.rowHeight(row)
                                     for row in range(2 if table.rowCount() > 2 else table.rowCount())) \
                                 + label.fontMetrics().boundingRect(label.text()).height() + 35
        # SPCS
        elif self.__method == "SPCS":
            # Getting needed results:
            matrix_alt = np.array(self.alg_obj.matrix)
            norm_matrix_alt = np.array(self.alg_obj.norm_matrix)
            model_nr = np.array([int(elem) for elem in matrix_alt[:, 0]])
            matrix_alt = matrix_alt[:, 1:]
            norm_matrix_alt = norm_matrix_alt[:, 1:]
            quo_point = np.array(self.alg_obj.quo_point)
            asp_point = np.array(self.alg_obj.asp_point)
            shape = matrix_alt.shape

            # Creating tables with data:
            table_not_dominated_alt = QTableWidget(self)
            table_not_dominated_alt_normalized = QTableWidget(self)
            table_quo_point = QTableWidget(self)
            table_asp_point = QTableWidget(self)

            tables = [table_not_dominated_alt,  table_not_dominated_alt_normalized, table_quo_point, table_asp_point]

            # Loop over tables to set their size:
            for i, table in enumerate(tables):
                table.setColumnCount(shape[1])
                if i < 2:
                    table.setRowCount(shape[0])
                else:
                    table.setRowCount(1)

            # Creating headers arrays:
            horizontal_headers = []
            vertical_headers = []

            # Creating texts and labels for tables:
            texts = ["Not dominated alternatives", "Normalized not dominated alternatives",
                     "Quo point (average criteria values)", "Aspiration point (best possible criteria values)"]
            labels = []

            for text in texts:
                label = QLabel(text)
                label.setFont(font)
                labels.append(label)

            # Entering data into created tables:
            for row in range(table_not_dominated_alt.rowCount()):
                vertical_headers.append(f"M{model_nr[row]}")
                for col in range(table_not_dominated_alt.columnCount()):
                    item = QTableWidgetItem(f"{round(matrix_alt[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_not_dominated_alt.setItem(row, col, item)

                    item = QTableWidgetItem(f"{round(norm_matrix_alt[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_not_dominated_alt_normalized.setItem(row, col, item)

                    if row == 0:
                        horizontal_headers.append(f"Param {col + 1}")

                        # Setting 1 row tables:
                        matrices = [quo_point, asp_point]
                        tables_in = [table_quo_point, table_asp_point]

                        for matrix, table in zip(matrices, tables_in):
                            item = QTableWidgetItem(f"{round(matrix[col], 4)}")
                            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                            item.setTextAlignment(Qt.AlignCenter)
                            table.setItem(row, col, item)

            # Setting headers to tables:
            for i, table in enumerate(tables):
                table.setHorizontalHeaderLabels(horizontal_headers)
                if i < 2:
                    table.setVerticalHeaderLabels(vertical_headers)

            # Assigning tables to scroll area:
            for i, table in enumerate(tables):
                scroll_area = QScrollArea(self)
                scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(table)
                scroll_areas.append(scroll_area)

            # Acquiring approximate width of the window:
            window_width = sum(table_quo_point.columnWidth(col)
                               for col in range(table_quo_point.columnCount()))

            window_width += int(0.75 * table_quo_point.columnWidth(0))
            control_width = labels[1].fontMetrics().boundingRect(labels[3].text()).width() + 30

            if window_width < control_width:
                window_width = control_width

            # Acquiring approximate height of the window:
            window_height = 0
            for table, label in zip(tables, labels):
                window_height += sum(table.rowHeight(row)
                                     for row in range(2 if table.rowCount() > 2 else table.rowCount())) \
                                 + label.fontMetrics().boundingRect(label.text()).height() + 35
        # RSM:
        elif self.__method == "RSM":
            # Getting needed results:
            pareto_alt_matrix = np.array(self.alg_obj.pareto_alt_matrix)
            model_nr = np.array([int(elem) for elem in pareto_alt_matrix[:, 0]])
            pareto_alt_matrix = pareto_alt_matrix[:, 1:]
            quo_points = np.array(self.alg_obj.quo_points)
            asp_points = np.array(self.alg_obj.asp_points)
            anti_asp_points = np.array(self.alg_obj.anti_asp_points)
            optimum_lim_points = np.array(self.alg_obj.optimum_lim_opt_points)
            shape = pareto_alt_matrix.shape

            # Creating tables with data:
            table_pareto_alt_matrix = QTableWidget(self)
            table_quo_points = QTableWidget(self)
            table_asp_points = QTableWidget(self)
            table_anti_asp_points = QTableWidget(self)
            table_optimum_lim_points = QTableWidget(self)

            tables = [table_pareto_alt_matrix, table_quo_points, table_asp_points, table_anti_asp_points, table_optimum_lim_points]

            # Loop over tables to set their size:
            for i, table in enumerate(tables):
                table.setColumnCount(shape[1])
                if i == 0:
                    table.setRowCount(shape[0])
                elif i == 4:
                    table.setRowCount(2)
                else:
                    table.setRowCount(3)

            # Creating headers arrays:
            horizontal_headers = []
            vertical_headers = []

            # Creating texts and labels for tables:
            texts = ["Pareto set of alternatives (not dominated)", "Quo points (average criteria values with shifts)",
                     "Aspiration points (best possible criteria values with shifts)",
                     "Anti-aspiration points (worst possible criteria values with shifts)",
                     "Optimum lim points (boundary points with shifts)"]
            labels = []

            font.setPointSize(14)

            for text in texts:
                label = QLabel(text)
                label.setFont(font)
                labels.append(label)

            # Entering data into created tables:
            for row in range(table_pareto_alt_matrix.rowCount()):
                vertical_headers.append(f"M{model_nr[row]}")
                for col in range(table_pareto_alt_matrix.columnCount()):
                    item = QTableWidgetItem(f"{round(pareto_alt_matrix[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table_pareto_alt_matrix.setItem(row, col, item)

                    if row == 0:
                        horizontal_headers.append(f"Param {col + 1}")

            # Setting points tables:
            matrices = [quo_points, asp_points, anti_asp_points, optimum_lim_points]
            tables_in = [table_quo_points, table_asp_points,
                         table_anti_asp_points, table_optimum_lim_points]
            for matrix, table in zip(matrices, tables_in):
                for row in range(matrix.shape[0]):
                    for col in range(matrix.shape[1]):
                        item = QTableWidgetItem(f"{round(matrix[row, col], 4)}")
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(row, col, item)

            # Setting headers to tables:
            for i, table in enumerate(tables):
                table.setHorizontalHeaderLabels(horizontal_headers)
                if i == 0:
                    table.setVerticalHeaderLabels(vertical_headers)

            # Assigning tables to scroll area:
            for i, table in enumerate(tables):
                scroll_area = QScrollArea(self)
                scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(table)
                scroll_areas.append(scroll_area)

            # Acquiring approximate width of the window:
            window_width = sum(table_quo_points.columnWidth(col)
                               for col in range(table_quo_points.columnCount()))

            window_width += int(0.75 * table_quo_points.columnWidth(0))
            control_width = labels[3].fontMetrics().boundingRect(labels[3].text()).width() + 25

            if window_width < control_width:
                window_width = control_width

            # Acquiring approximate height of the window:
            window_height = 0
            for table, label in zip(tables, labels):
                window_height += sum(table.rowHeight(row)
                                     for row in range(3 if table.rowCount() > 2 else table.rowCount())) \
                                 + label.fontMetrics().boundingRect(label.text()).height() + 35

        # Setting window size:
        self.setFixedWidth(window_width)
        self.setFixedHeight(window_height)

        for label, scroll_area in zip(labels, scroll_areas):
            final_layout.addWidget(label, alignment=Qt.AlignCenter)
            if self.__method == "TOPSIS" or self.__method == "RSM":
                final_layout.addWidget(scroll_area)
            else:
                final_layout.addWidget(scroll_area, alignment=Qt.AlignCenter)

        # Changing font size:
        font.setPointSize(12)

        # Continue button:
        button_continue = QPushButton("Continue")
        if self.__method == "RSM":
            button_continue.clicked.connect(self.__backup_calc_window)
        else:
            button_continue.clicked.connect(self.__ranking_show)

        button_continue.setFont(font)
        final_layout.addWidget(button_continue, alignment=Qt.AlignRight)

        # Window centering and final layout setting:
        self.setLayout(final_layout)
        center(self)

    # Moving to result back-up window:
    def __backup_calc_window(self) -> None:
        self.backup_calc_window = BackupCalcWindow(self.__method, self.alg_obj, self.__parent)
        self.destroy()
        self.backup_calc_window.show()

    # Results display:
    def __ranking_show(self) -> None:
        self.__ranking_show_window = RankingShow(self.__method, self.alg_obj, self.__parent)
        self.destroy()
        self.__ranking_show_window.show()


# Backup window for results display - only for RSM:
class BackupCalcWindow(QWidget):
    def __init__(self, method: str, alg_obj: RSM, parent: QMainWindow) -> None:
        super().__init__()

        self.__method = method
        self.alg_obj = alg_obj
        self.__parent = parent

        self.setWindowTitle("Results")

        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(13)

        final_layout = QVBoxLayout()

        # Getting needed results:
        pareto_alt_matrix_norm = np.array(self.alg_obj.pareto_norm_matrix)
        model_nr = np.array([int(elem) for elem in pareto_alt_matrix_norm[:, 0]])
        pareto_alt_matrix_norm = pareto_alt_matrix_norm[:, 1:]
        norm_quo_points = np.array(self.alg_obj.norm_quo_points)
        norm_asp_points = np.array(self.alg_obj.norm_asp_points)
        norm_anti_asp_points = np.array(self.alg_obj.norm_anti_asp_points)
        norm_optimum_lim_points = np.array(self.alg_obj.norm_optimum_lim_opt_points)
        shape = pareto_alt_matrix_norm.shape

        # Creating tables with data:
        table_pareto_alt_matrix_norm = QTableWidget(self)
        table_norm_quo_points = QTableWidget(self)
        table_norm_asp_points = QTableWidget(self)
        table_norm_anti_asp_points = QTableWidget(self)
        table_norm_optimum_lim_points = QTableWidget(self)

        tables = [table_pareto_alt_matrix_norm, table_norm_quo_points, table_norm_asp_points,
                  table_norm_anti_asp_points,table_norm_optimum_lim_points]

        # Loop over tables to set their size:
        for i, table in enumerate(tables):
            table.setColumnCount(shape[1])
            if i == 0:
                table.setRowCount(shape[0])
            elif i == 4:
                table.setRowCount(2)
            else:
                table.setRowCount(3)

        # Creating headers arrays:
        horizontal_headers = []
        vertical_headers = []

        # Creating texts and labels for tables:
        texts = ["Normalized Pareto set of alternatives (not dominated)",
                 "Normalized quo points (average criteria values with shifts)",
                 "Normalized aspiration points (best possible criteria values with shifts)",
                 "Normalized anti-aspiration points (worst possible criteria values with shifts)",
                 "Normalized optimum lim points (boundary points with shifts)"]
        labels = []

        font.setPointSize(14)

        for text in texts:
            label = QLabel(text)
            label.setFont(font)
            labels.append(label)

        # Entering data into created tables:
        for row in range(table_pareto_alt_matrix_norm.rowCount()):
            vertical_headers.append(f"M{model_nr[row]}")
            for col in range(table_pareto_alt_matrix_norm.columnCount()):
                item = QTableWidgetItem(f"{round(pareto_alt_matrix_norm[row, col], 4)}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                table_pareto_alt_matrix_norm.setItem(row, col, item)

                if row == 0:
                    horizontal_headers.append(f"Param {col + 1}")

        # Setting points tables:
        matrices = [norm_quo_points, norm_asp_points, norm_anti_asp_points, norm_optimum_lim_points]
        tables_in = [table_norm_quo_points, table_norm_asp_points,
                     table_norm_anti_asp_points, table_norm_optimum_lim_points]

        for matrix, table in zip(matrices, tables_in):
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    item = QTableWidgetItem(f"{round(matrix[row, col], 4)}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row, col, item)

        # Setting headers to tables:
        for i, table in enumerate(tables):
            table.setHorizontalHeaderLabels(horizontal_headers)
            if i == 0:
                table.setVerticalHeaderLabels(vertical_headers)

        scroll_areas = []

        # Assigning tables to scroll area:
        for i, table in enumerate(tables):
            scroll_area = QScrollArea(self)
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(table)
            scroll_areas.append(scroll_area)

        # Acquiring approximate width of the window:
        window_width = sum(table_norm_quo_points.columnWidth(col)
                           for col in range(table_norm_quo_points.columnCount()))

        window_width += int(0.75 * table_norm_quo_points.columnWidth(0))
        control_width = labels[3].fontMetrics().boundingRect(labels[3].text()).width() + 25

        if window_width < control_width:
            window_width = control_width

        # Acquiring approximate height of the window:
        window_height = 0
        for table, label in zip(tables, labels):
            window_height += sum(table.rowHeight(row)
                                 for row in range(3 if table.rowCount() > 2 else table.rowCount())) \
                             + label.fontMetrics().boundingRect(label.text()).height() + 35

        # Setting window size:

        self.setFixedWidth(window_width)
        self.setFixedHeight(window_height)

        for label, scroll_area in zip(labels, scroll_areas):
            final_layout.addWidget(label, alignment=Qt.AlignCenter)
            final_layout.addWidget(scroll_area)

        # Changing font size:
        font.setPointSize(12)

        # Continue button:
        button_continue = QPushButton("Continue")
        button_continue.clicked.connect(self.__ranking_show)

        button_continue.setFont(font)
        final_layout.addWidget(button_continue, alignment=Qt.AlignRight)

        # Window centering and final layout setting:
        self.setLayout(final_layout)
        center(self)

    # Moving to ranking window:
    def __ranking_show(self) -> None:
        self.__ranking_show_window = RankingShow(self.__method, self.alg_obj, self.__parent)
        self.destroy()
        self.__ranking_show_window.show()


# Ranking display:
class RankingShow(QWidget):
    def __init__(self, method: str, alg_obj: Union[Topsis, SPCS, RSM], parent: QMainWindow) -> None:
        super().__init__()

        # Remembering parent and method:
        self.__parent = parent
        self.__method = method

        self.setWindowTitle("Ranking")

        # Final layout:
        final_layout = QVBoxLayout()

        # Initializing canvas for SPCS and RSM method:
        canvas = None

        # Initializing algorithm object:
        self.alg_obj = alg_obj

        # TOPSIS:
        if self.__method == "TOPSIS":
            # Results:
            ranking = self.alg_obj.rank_to_best_similarity()
            scoring = self.alg_obj.worst_similarity
            scoring = np.sort(scoring)[::-1]

            # Setting up results table:
            self.table = QTableWidget(self)
            self.table.setRowCount(len(ranking))
            self.table.setColumnCount(3)

            # Setting up headers:
            horizontal_headers = ["Place", "Model", "Scoring"]

            # Entering data into created table:
            for row in range(self.table.rowCount()):
                for col in range(3):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 0:
                        item.setText(f"{row + 1}")
                    elif col == 1:
                        item.setText(f"M{ranking[row]}")
                    elif col == 2:
                        item.setText(f"{round(scoring[row], 5)}")

                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, col, item)

            self.table.setHorizontalHeaderLabels(horizontal_headers)
            self.table.setVerticalHeaderLabels([])
            final_layout.addWidget(self.table)

            font = QFont()
            font.setFamily("Calibri")
            font.setPointSize(12)
            label = QLabel("The higher the scoring, the better the alternative.")
            label.setFont(font)
            final_layout.addWidget(label)

        # SPCS:
        elif self.__method == "SPCS":
            # Results:
            model_nr = []
            scoring = []
            for tup in self.alg_obj.sc_val:
                model_nr.append(tup[0])
                scoring.append(tup[1])

            # Setting up results table:
            self.table = QTableWidget(self)
            self.table.setRowCount(len(scoring))
            self.table.setColumnCount(3)

            # Setting up headers:
            horizontal_headers = ["Place", "Model", "Scoring"]

            # Entering data into created table:
            for row in range(self.table.rowCount()):
                for col in range(3):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 0:
                        item.setText(f"{row + 1}")
                    elif col == 1:
                        item.setText(f"M{model_nr[row]}")
                    elif col == 2:
                        item.setText(f"{round(scoring[row], 5)}")

                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, col, item)

            self.table.setHorizontalHeaderLabels(horizontal_headers)
            self.table.setVerticalHeaderLabels([])
            final_layout.addWidget(self.table)

            font = QFont()
            font.setFamily("Calibri")
            font.setPointSize(12)
            label = QLabel("The lower the scoring, the better the alternative.")
            label.setFont(font)
            final_layout.addWidget(label)

            # Calculations necessary to prepare result plot:
            fx = lambda x:  self.alg_obj.a * x + self.alg_obj.b

            if self.alg_obj.asp_point[0] < self.alg_obj.quo_point[0]:
                x = np.arange(self.alg_obj.asp_point[0], self.alg_obj.quo_point[0], 0.001)
            else:
                x = np.arange(self.alg_obj.quo_point[0], self.alg_obj.asp_point[0], 0.001)

            # Calculating linear function values (interpolation):
            y = fx(x)

            # Creating canvas:
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            canvas.axes.grid(True)

            # Legend variable:
            l = []

            # Adding plots to canvas and labels to legend variable:
            canvas.axes.plot(x, y)
            l.append("Asp-Quo section")

            canvas.axes.scatter( self.alg_obj.asp_point[0],  self.alg_obj.asp_point[1])
            l.append("Aspiration point")

            canvas.axes.scatter( self.alg_obj.quo_point[0],  self.alg_obj.quo_point[1])
            l.append("Quo point")

            for point, values in self.alg_obj.alt.items():
                canvas.axes.scatter(values[0], values[1])
                l.append(f"M{point}")

            # Finalizing canvas creation:
            canvas.axes.grid(True)
            canvas.axes.set_title("Solution")
            canvas.axes.legend(l, fontsize="small")
            canvas.draw()

        # RSM:
        elif self.__method == "RSM":
            # Results:
            model_nr = []
            scoring = []
            for tup in self.alg_obj.sc_val:
                model_nr.append(tup[0])
                scoring.append(tup[1])

            # Setting up results table:
            self.table = QTableWidget(self)
            self.table.setRowCount(len(scoring))
            self.table.setColumnCount(3)

            # Setting up headers:
            horizontal_headers = ["Place", "Model", "Scoring"]

            # Entering data into created table:
            for row in range(self.table.rowCount()):
                for col in range(3):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 0:
                        item.setText(f"{row + 1}")
                    elif col == 1:
                        item.setText(f"M{model_nr[row]}")
                    elif col == 2:
                        item.setText(f"{round(scoring[row], 5)}")

                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, col, item)

            self.table.setHorizontalHeaderLabels(horizontal_headers)
            self.table.setVerticalHeaderLabels([])
            final_layout.addWidget(self.table)

            font = QFont()
            font.setFamily("Calibri")
            font.setPointSize(12)
            label = QLabel("The lower the scoring, the better the alternative.")
            label.setFont(font)
            final_layout.addWidget(label)

            # Acquiring necessary results:
            asp_points = self.alg_obj.norm_asp_points
            quo_points = self.alg_obj.norm_quo_points
            anti_asp_points = self.alg_obj.norm_anti_asp_points
            optimum_lim_min_points = self.alg_obj.norm_optimum_lim_opt_points

            # 3D plots:
            canvas = MplCanvas3D(self, width=5, height=4, dpi=100)
            canvas.axes.grid(True)

            # Drawing the solution:
            x, y, z = np.array(self.alg_obj.pareto_norm_alt[self.alg_obj.sc_val[0][0]]).T
            canvas.axes.scatter(x, y, z, color="black", marker="*", label="Best")

            # Drawing set of alternatives:
            _, x, y, z = np.array(self.alg_obj.alt_norm_matrix).T
            canvas.axes.scatter(x, y, z, color="orange", marker="*", label="U")
            l = ["U - initial set"]

            # Drawing Pareto set of alternatives:
            _, x, y, z = np.array(self.alg_obj.pareto_norm_matrix).T
            canvas.axes.scatter(x, y, z, color="magenta", marker="*", label="PU")
            l.append("PU - pareto-optimal set")

            # Drawing A0 set - lower optimum lim points:
            x, y, z = np.array(optimum_lim_min_points).T
            canvas.axes.plot(x, y, z, color="blue", marker="*", label="A0")

            # Drawing A1 set - aspiration points:
            x, y, z = np.array(asp_points).T
            canvas.axes.plot(x, y, z, color="green", marker="*", label="A1")

            # Drawing A2 set - quo points:
            x, y, z = np.array(quo_points).T
            canvas.axes.plot(x, y, z, color="cyan", marker="*", label="A2")

            # Drawing A3 set - anti-aspiration spoints:
            x, y, z = np.array(anti_asp_points).T
            canvas.axes.plot(x, y, z, color="red", marker="*", label="A3")

            # Axes limits:
            canvas.axes.set_xlim(0, 1)
            canvas.axes.set_ylim(0, 1)
            canvas.axes.set_zlim(0, 1)

            # Grid on:
            canvas.axes.grid(True)

            # Title:
            canvas.axes.set_title("Solution")

            # Adding legend:
            canvas.axes.legend(bbox_to_anchor=(0.1, 1.05))

            # Draw the canvas:
            canvas.draw()

        # Acquiring approximate size of the table:
        window_width = sum(self.table.columnWidth(col) for col in range(self.table.columnCount()))
        window_width += int(0.8 * self.table.columnWidth(0))

        # Acquiring approximate height of the window:
        window_height = 0
        window_height += sum(self.table.rowHeight(row)
                             for row in range(5 if self.table.rowCount() > 4 else self.table.rowCount())) + 50

        # If there is a plot:
        if canvas:
            final_layout.addWidget(canvas)
            window_height += 400
            window_width += 50

        # Font:
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)

        # Return to main menu button:
        button_return = QPushButton("Return to main menu")
        button_return.clicked.connect(self.__return_main_window)
        button_return.setFont(font)
        final_layout.addWidget(button_return, alignment=Qt.AlignRight)

        final_layout.setSpacing(5)

        # Setting window size:
        self.setFixedWidth(window_width)
        self.setFixedHeight(window_height)

        # Window centering and final layout setting:
        self.setLayout(final_layout)
        center(self)

    # Function handling return to main window:
    def __return_main_window(self) -> None:
        self.close()
        self.__parent.show()


# Window centering:
def center(window: QWidget) -> None:
    qr = window.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    window.move(qr.topLeft())


# Running the app:
def run() -> None:
    application = QApplication([])
    w = Window()
    w.show()
    application.exec()
