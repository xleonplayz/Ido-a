# -*- coding: utf-8 -*-
"""
GUI module for importing and analyzing pulsed measurement data with NV state analysis.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import pyqtgraph as pg
from functools import partial

from PySide2 import QtCore, QtWidgets

from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.module import GuiBase
from qudi.util import uic
from qudi.util.colordefs import QudiPalettePale as palette


class PulsedDataAnalysisMainWindow(QtWidgets.QMainWindow):
    """Main window for the PulsedDataAnalysis GUI"""
    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_pulsed_data_analysis.ui')
        
        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)


class PulsedDataAnalysisSettingsDialog(QtWidgets.QDialog):
    """Dialog for extraction/analysis settings"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Settings')
        self.setMinimumWidth(400)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        
        # Settings widgets will be added during runtime


class PulsedDataAnalysisGui(GuiBase):
    """
    GUI for importing and analyzing pulsed measurement data with a focus on NV center state analysis.
    
    Example config:
    
    pulsed_data_analysis_gui:
        module.Class: 'pulsed.pulsed_data_analysis_gui.PulsedDataAnalysisGui'
        connect:
            analysis_logic: 'pulsed_data_analysis_logic'
    """
    
    # Connectors
    _analysis_logic = Connector(interface='PulsedDataAnalysisLogic')
    
    # Status variables
    _default_save_path = StatusVar(default=None)
    _recent_files = StatusVar(default=[])
    _data_selection_range = StatusVar(default=(0, 0))
    
    # Signals
    sigOpenPulsedFile = QtCore.Signal(str)  # Changed to be specific to pulsed data
    sigOpenRawFile = QtCore.Signal(str)  # New signal for raw data file
    sigOpenLaserFile = QtCore.Signal(str)  # New signal for laser data file
    sigExtractLaserPulses = QtCore.Signal()
    sigAnalyzePulses = QtCore.Signal()
    sigSetThreshold = QtCore.Signal(float)
    sigSetReferenceFromSelection = QtCore.Signal(int, int)
    sigToggleMsStateDisplay = QtCore.Signal(bool)
    sigSaveAnalysisResults = QtCore.Signal(str, str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._extraction_settings_dialog = None
        self._analysis_settings_dialog = None
        self._nv_settings_dialog = None
        
        # Plot items
        self._data_plot_curves = []
        self._laser_plot_curves = []
        self._nv_plot_curves = []
        self._nv_state_region = None
        self._histogram_plot_items = []
        
        # Data for plots
        self._signal_data = None
        self._laser_data = None
        self._nv_state_data = None
        self._state_histogram = None
        
        # Selection for reference level
        self._data_selection_active = False
        self._data_selection_region = None
        
    def on_activate(self):
        """Activate the module and initialize the GUI"""
        self._mw = PulsedDataAnalysisMainWindow()
        
        # Create settings dialogs
        self._extraction_settings_dialog = PulsedDataAnalysisSettingsDialog()
        self._analysis_settings_dialog = PulsedDataAnalysisSettingsDialog()
        self._nv_settings_dialog = PulsedDataAnalysisSettingsDialog()
        
        # Initialize plots
        self._initialize_plots()
        
        # Add the file selection buttons to the UI
        self._add_file_selection_buttons()
        
        # Create signal-slot connections
        self._create_ui_connections()
        self._connect_signals_to_logic()
        
        # Set up the Recent Files menu
        self._update_recent_files_menu()
        
        # Apply status variables
        if self._data_selection_range != (0, 0):
            # Create selection region if there was one saved
            self._create_data_selection_region(*self._data_selection_range)
        
        # Show window
        self.show()
    
    def _add_file_selection_buttons(self):
        """Add the three file selection buttons to the UI"""
        # Create a new widget to hold the file selection buttons
        file_selection_widget = QtWidgets.QWidget()
        file_selection_layout = QtWidgets.QHBoxLayout()
        file_selection_widget.setLayout(file_selection_layout)
        
        # Create title label
        title_label = QtWidgets.QLabel("Select Files Individually:")
        file_selection_layout.addWidget(title_label)
        
        # Create the three buttons
        self._pulsed_file_button = QtWidgets.QPushButton("Open Pulsed Measurement")
        self._pulsed_file_button.setToolTip("Open a pulsed measurement file (_pulsed_measurement)")
        file_selection_layout.addWidget(self._pulsed_file_button)
        
        self._raw_file_button = QtWidgets.QPushButton("Open Raw Data")
        self._raw_file_button.setToolTip("Open a raw timetrace file (_raw_timetrace)")
        file_selection_layout.addWidget(self._raw_file_button)
        
        self._laser_file_button = QtWidgets.QPushButton("Open Laser Pulses")
        self._laser_file_button.setToolTip("Open a laser pulses file (_laser_pulses)")
        file_selection_layout.addWidget(self._laser_file_button)
        
        # Add spacer to right-align the buttons
        file_selection_layout.addItem(QtWidgets.QSpacerItem(40, 20, 
                                       QtWidgets.QSizePolicy.Expanding, 
                                       QtWidgets.QSizePolicy.Minimum))
        
        # Add the file selection widget to the main window
        # Insert it after the data tab's vertical layout
        self._mw.data_tab.layout().insertWidget(0, file_selection_widget)
    
    def on_deactivate(self):
        """Deactivate the module and clean up"""
        # Disconnect signals
        self._disconnect_signals_from_logic()
        
        # Close window
        self._mw.close()
        self._mw = None
        
        self._extraction_settings_dialog.close()
        self._extraction_settings_dialog = None
        
        self._analysis_settings_dialog.close()
        self._analysis_settings_dialog = None
        
        self._nv_settings_dialog.close()
        self._nv_settings_dialog = None
    
    def show(self):
        """Show the main window"""
        self._mw.show()
        
    def _initialize_plots(self):
        """Initialize all plot widgets and items"""
        # Configure data plot
        self._mw.data_plot_widget.setLabel('left', 'Signal', units='a.u.')
        self._mw.data_plot_widget.setLabel('bottom', 'Time', units='s')
        self._mw.data_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self._mw.data_plot_widget.setMouseEnabled(x=True, y=True)
        
        # Configure laser plot
        self._mw.laser_plot_widget.setLabel('left', 'Counts', units='')
        self._mw.laser_plot_widget.setLabel('bottom', 'Time', units='s')
        self._mw.laser_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self._mw.laser_plot_widget.setMouseEnabled(x=True, y=True)
        
        # Configure NV state plot
        self._mw.nv_plot_widget.setLabel('left', 'Signal', units='a.u.')
        self._mw.nv_plot_widget.setLabel('bottom', 'Time', units='s')
        self._mw.nv_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self._mw.nv_plot_widget.setMouseEnabled(x=True, y=True)
        
        # Configure histogram plot
        self._mw.histogram_plot_widget.setLabel('left', 'Count', units='')
        self._mw.histogram_plot_widget.setLabel('bottom', 'State', units='')
        self._mw.histogram_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self._mw.histogram_plot_widget.setMouseEnabled(x=True, y=True)
    
    def _create_ui_connections(self):
        """Create connections between UI elements and methods"""
        # Menu connections
        self._mw.action_open.triggered.connect(self.open_file_dialog)
        self._mw.action_save_analysis.triggered.connect(self.save_analysis)
        self._mw.action_save_analysis_as.triggered.connect(self.save_analysis_as)
        self._mw.actionClear_Recent_Files.triggered.connect(self.clear_recent_files)
        
        # Settings menu connections
        self._mw.action_extraction_settings.triggered.connect(self.show_extraction_settings)
        self._mw.action_analysis_settings.triggered.connect(self.show_analysis_settings)
        self._mw.action_nv_settings.triggered.connect(self.show_nv_settings)
        
        # Button connections
        self._mw.extract_laser_button.clicked.connect(self.extract_laser_pulses)
        self._mw.analyze_button.clicked.connect(self.analyze_pulses)
        self._mw.extraction_settings_button.clicked.connect(self.show_extraction_settings)
        self._mw.analysis_settings_button.clicked.connect(self.show_analysis_settings)
        
        # File selection buttons connections
        self._pulsed_file_button.clicked.connect(self.open_pulsed_file_dialog)
        self._raw_file_button.clicked.connect(self.open_raw_file_dialog)
        self._laser_file_button.clicked.connect(self.open_laser_file_dialog)
        
        # NV state settings connections
        self._mw.ms_minus1_radiobutton.toggled.connect(self.toggle_ms_state_display)
        self._mw.threshold_spinbox.valueChanged.connect(self.update_threshold_from_spinbox)
        self._mw.threshold_slider.valueChanged.connect(self.update_threshold_from_slider)
        self._mw.set_reference_button.clicked.connect(self.set_reference_from_selection)
        self._mw.save_analysis_button.clicked.connect(self.save_analysis)
        
        # Additional toolbar connections
        self._mw.action_extract_laser.triggered.connect(self.extract_laser_pulses)
        self._mw.action_analyze_pulses.triggered.connect(self.analyze_pulses)
    
    def _connect_signals_to_logic(self):
        """Connect GUI signals to logic module methods"""
        # Connect GUI signals to logic signals
        self.sigOpenPulsedFile.connect(self._analysis_logic().load_pulsed_file)
        self.sigOpenRawFile.connect(self._analysis_logic().load_raw_file)
        self.sigOpenLaserFile.connect(self._analysis_logic().load_laser_file)
        self.sigExtractLaserPulses.connect(self._analysis_logic().extract_laser_pulses)
        self.sigAnalyzePulses.connect(self._analysis_logic().analyze_laser_pulses)
        self.sigSetThreshold.connect(self._analysis_logic().set_nv_threshold)
        self.sigSetReferenceFromSelection.connect(self._analysis_logic().set_reference_from_selection)
        self.sigToggleMsStateDisplay.connect(self._analysis_logic().toggle_ms_state_display)
        
        # Connect logic signals to GUI methods
        self._analysis_logic().sigDataLoaded.connect(self.update_data_display)
        self._analysis_logic().sigAnalysisComplete.connect(self.update_analysis_display)
        self._analysis_logic().sigNvStateHistogramUpdated.connect(self.update_nv_state_display)
        self._analysis_logic().sigThresholdChanged.connect(self.update_threshold_display)
        self._analysis_logic().sigReferenceChanged.connect(self.update_reference_display)
    
    def _disconnect_signals_from_logic(self):
        """Disconnect all signal-slot connections to the logic module"""
        # Disconnect GUI signals from logic signals
        self.sigOpenPulsedFile.disconnect()
        self.sigOpenRawFile.disconnect()
        self.sigOpenLaserFile.disconnect()
        self.sigExtractLaserPulses.disconnect()
        self.sigAnalyzePulses.disconnect()
        self.sigSetThreshold.disconnect()
        self.sigSetReferenceFromSelection.disconnect()
        self.sigToggleMsStateDisplay.disconnect()
        
        # Disconnect logic signals from GUI methods
        self._analysis_logic().sigDataLoaded.disconnect()
        self._analysis_logic().sigAnalysisComplete.disconnect()
        self._analysis_logic().sigNvStateHistogramUpdated.disconnect()
        self._analysis_logic().sigThresholdChanged.disconnect()
        self._analysis_logic().sigReferenceChanged.disconnect()
    
    def _update_recent_files_menu(self):
        """Update the Recent Files menu with the current list of recent files"""
        # Clear all existing items except the first two (Clear and separator)
        menu = self._mw.menuRecent_Files
        actions = menu.actions()
        for action in actions[2:]:
            menu.removeAction(action)
        
        # Add recent files
        for file_path in self._analysis_logic()._recent_files:
            action = QtWidgets.QAction(os.path.basename(file_path), self._mw)
            action.setData(file_path)
            action.triggered.connect(partial(self.open_recent_file, file_path))
            menu.addAction(action)
    
    def open_file_dialog(self):
        """Open a file dialog to select a file to load (legacy method)"""
        # Show message box suggesting the use of specific file buttons
        response = QtWidgets.QMessageBox.question(
            self._mw,
            "Use Specific File Buttons",
            "It's recommended to use the specific file buttons to load each file type.\n\n"
            "Do you still want to open a generic file?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if response == QtWidgets.QMessageBox.No:
            return
            
        # Get the initial directory
        initial_dir = self._default_save_path
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._mw,
            "Open Data File",
            initial_dir,
            "Data Files (*.dat *.csv *.npy);;All Files (*)"
        )
        
        if file_path:
            # Update default save path
            self._default_save_path = os.path.dirname(file_path)
            
            # Determine file type and emit the appropriate signal
            basename = os.path.basename(file_path).lower()
            if "_pulsed_measurement" in basename:
                self.sigOpenPulsedFile.emit(file_path)
            elif "_raw_timetrace" in basename:
                self.sigOpenRawFile.emit(file_path)
            elif "_laser_pulses" in basename:
                self.sigOpenLaserFile.emit(file_path)
            else:
                # If file type can't be determined from name, show dialog to select type
                file_type, ok = QtWidgets.QInputDialog.getItem(
                    self._mw,
                    "Select File Type",
                    "What type of data is this file?",
                    ["Pulsed Measurement", "Raw Timetrace", "Laser Pulses"],
                    0,
                    False
                )
                
                if ok:
                    if file_type == "Pulsed Measurement":
                        self.sigOpenPulsedFile.emit(file_path)
                    elif file_type == "Raw Timetrace":
                        self.sigOpenRawFile.emit(file_path)
                    elif file_type == "Laser Pulses":
                        self.sigOpenLaserFile.emit(file_path)
            
            # Update recent files menu
            self._update_recent_files_menu()
    
    def open_pulsed_file_dialog(self):
        """Open a file dialog specifically for pulsed measurement data"""
        # Get the initial directory
        initial_dir = self._default_save_path
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._mw,
            "Open Pulsed Measurement File",
            initial_dir,
            "Data Files (*.dat *.csv *.npy);;All Files (*)"
        )
        
        if file_path:
            # Update default save path
            self._default_save_path = os.path.dirname(file_path)
            
            # Emit signal to load pulsed file
            self.sigOpenPulsedFile.emit(file_path)
            
            # Update status
            self._mw.statusbar.showMessage(f"Loaded pulsed measurement file: {os.path.basename(file_path)}")
            
            # Update recent files menu
            self._update_recent_files_menu()
    
    def open_raw_file_dialog(self):
        """Open a file dialog specifically for raw timetrace data"""
        # Get the initial directory
        initial_dir = self._default_save_path
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._mw,
            "Open Raw Timetrace File",
            initial_dir,
            "Data Files (*.dat *.csv *.npy);;All Files (*)"
        )
        
        if file_path:
            # Update default save path
            self._default_save_path = os.path.dirname(file_path)
            
            # Emit signal to load raw file
            self.sigOpenRawFile.emit(file_path)
            
            # Update status
            self._mw.statusbar.showMessage(f"Loaded raw timetrace file: {os.path.basename(file_path)}")
            
            # Update recent files menu
            self._update_recent_files_menu()
    
    def open_laser_file_dialog(self):
        """Open a file dialog specifically for laser pulses data"""
        # Get the initial directory
        initial_dir = self._default_save_path
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._mw,
            "Open Laser Pulses File",
            initial_dir,
            "Data Files (*.dat *.csv *.npy);;All Files (*)"
        )
        
        if file_path:
            # Update default save path
            self._default_save_path = os.path.dirname(file_path)
            
            # Emit signal to load laser file
            self.sigOpenLaserFile.emit(file_path)
            
            # Update status
            self._mw.statusbar.showMessage(f"Loaded laser pulses file: {os.path.basename(file_path)}")
            
            # Update recent files menu
            self._update_recent_files_menu()
    
    def open_recent_file(self, file_path):
        """Open a file from the recent files list"""
        if os.path.isfile(file_path):
            # Determine file type and emit the appropriate signal
            basename = os.path.basename(file_path).lower()
            if "_pulsed_measurement" in basename:
                self.sigOpenPulsedFile.emit(file_path)
            elif "_raw_timetrace" in basename:
                self.sigOpenRawFile.emit(file_path)
            elif "_laser_pulses" in basename:
                self.sigOpenLaserFile.emit(file_path)
            else:
                # If file type can't be determined from name, show dialog to select type
                file_type, ok = QtWidgets.QInputDialog.getItem(
                    self._mw,
                    "Select File Type",
                    "What type of data is this file?",
                    ["Pulsed Measurement", "Raw Timetrace", "Laser Pulses"],
                    0,
                    False
                )
                
                if ok:
                    if file_type == "Pulsed Measurement":
                        self.sigOpenPulsedFile.emit(file_path)
                    elif file_type == "Raw Timetrace":
                        self.sigOpenRawFile.emit(file_path)
                    elif file_type == "Laser Pulses":
                        self.sigOpenLaserFile.emit(file_path)
        else:
            # File doesn't exist anymore
            QtWidgets.QMessageBox.warning(
                self._mw,
                "File Not Found",
                f"The file {file_path} could not be found."
            )
            # Remove from recent files
            if file_path in self._analysis_logic()._recent_files:
                self._analysis_logic()._recent_files.remove(file_path)
            self._update_recent_files_menu()
    
    def clear_recent_files(self):
        """Clear the list of recent files"""
        self._analysis_logic()._recent_files = []
        self._update_recent_files_menu()
    
    def extract_laser_pulses(self):
        """Extract laser pulses from raw data"""
        self.sigExtractLaserPulses.emit()
    
    def analyze_pulses(self):
        """Analyze the laser pulses and perform NV state analysis"""
        self.sigAnalyzePulses.emit()
    
    def update_threshold_from_spinbox(self, value):
        """Update threshold value from the spinbox"""
        # Update slider without triggering another signal
        self._mw.threshold_slider.blockSignals(True)
        self._mw.threshold_slider.setValue(int(value * 100))
        self._mw.threshold_slider.blockSignals(False)
        
        # Send signal to logic
        self.sigSetThreshold.emit(value)
    
    def update_threshold_from_slider(self, value):
        """Update threshold value from the slider"""
        threshold = value / 100.0
        
        # Update spinbox without triggering another signal
        self._mw.threshold_spinbox.blockSignals(True)
        self._mw.threshold_spinbox.setValue(threshold)
        self._mw.threshold_spinbox.blockSignals(False)
        
        # Send signal to logic
        self.sigSetThreshold.emit(threshold)
    
    def toggle_ms_state_display(self, checked):
        """Toggle between displaying ms=-1 or ms=+1 states"""
        display_ms_minus1 = self._mw.ms_minus1_radiobutton.isChecked()
        
        # Update label texts
        if display_ms_minus1:
            self._mw.ms1_label.setText("ms=-1 states:")
            self._mw.ms1_percentage_label.setText("ms=-1 percentage:")
        else:
            self._mw.ms1_label.setText("ms=+1 states:")
            self._mw.ms1_percentage_label.setText("ms=+1 percentage:")
        
        # Send signal to logic
        self.sigToggleMsStateDisplay.emit(display_ms_minus1)
        
        # Update state display if available
        if self._nv_state_data is not None:
            self.update_nv_plot()
    
    def show_extraction_settings(self):
        """Show the extraction settings dialog"""
        # Get current settings from logic
        settings = self._analysis_logic().extraction_settings
        
        # Clear existing widgets
        for i in reversed(range(self._extraction_settings_dialog.layout.count())):
            widget = self._extraction_settings_dialog.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Add settings widgets
        for key, value in settings.items():
            layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(key)
            
            if isinstance(value, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, (int, float)):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setDecimals(5 if isinstance(value, float) else 0)
                widget.setRange(-1e9, 1e9)
                widget.setValue(value)
            else:
                widget = QtWidgets.QLineEdit(str(value))
            
            layout.addWidget(label)
            layout.addWidget(widget)
            
            # Store the key and widget in widget properties for later retrieval
            widget.setProperty("key", key)
            widget.setProperty("type", type(value).__name__)
            
            self._extraction_settings_dialog.layout.addLayout(layout)
        
        # Add OK and Cancel buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._apply_extraction_settings)
        buttons.rejected.connect(self._extraction_settings_dialog.reject)
        
        self._extraction_settings_dialog.layout.addWidget(buttons)
        
        # Show dialog
        self._extraction_settings_dialog.setWindowTitle("Extraction Settings")
        self._extraction_settings_dialog.exec_()
    
    def _apply_extraction_settings(self):
        """Apply the settings from the extraction settings dialog"""
        settings = {}
        
        # Get all settings from the dialog
        for i in range(self._extraction_settings_dialog.layout.count()):
            item = self._extraction_settings_dialog.layout.itemAt(i)
            if isinstance(item, QtWidgets.QHBoxLayout):
                widget = item.itemAt(1).widget()
                key = widget.property("key")
                type_name = widget.property("type")
                
                if isinstance(widget, QtWidgets.QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    value = widget.value()
                    if type_name == "int":
                        value = int(value)
                else:
                    value = widget.text()
                    if type_name == "int":
                        value = int(value)
                    elif type_name == "float":
                        value = float(value)
                
                settings[key] = value
        
        # Apply settings to logic
        self._analysis_logic().extraction_settings = settings
        
        # Close dialog
        self._extraction_settings_dialog.accept()
    
    def show_analysis_settings(self):
        """Show the analysis settings dialog"""
        # Get current settings from logic
        settings = self._analysis_logic().analysis_settings
        
        # Clear existing widgets
        for i in reversed(range(self._analysis_settings_dialog.layout.count())):
            widget = self._analysis_settings_dialog.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Add settings widgets
        for key, value in settings.items():
            layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(key)
            
            if isinstance(value, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, (int, float)):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setDecimals(5 if isinstance(value, float) else 0)
                widget.setRange(-1e9, 1e9)
                widget.setValue(value)
            else:
                widget = QtWidgets.QLineEdit(str(value))
            
            layout.addWidget(label)
            layout.addWidget(widget)
            
            # Store the key and widget in widget properties for later retrieval
            widget.setProperty("key", key)
            widget.setProperty("type", type(value).__name__)
            
            self._analysis_settings_dialog.layout.addLayout(layout)
        
        # Add OK and Cancel buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._apply_analysis_settings)
        buttons.rejected.connect(self._analysis_settings_dialog.reject)
        
        self._analysis_settings_dialog.layout.addWidget(buttons)
        
        # Show dialog
        self._analysis_settings_dialog.setWindowTitle("Analysis Settings")
        self._analysis_settings_dialog.exec_()
    
    def _apply_analysis_settings(self):
        """Apply the settings from the analysis settings dialog"""
        settings = {}
        
        # Get all settings from the dialog
        for i in range(self._analysis_settings_dialog.layout.count()):
            item = self._analysis_settings_dialog.layout.itemAt(i)
            if isinstance(item, QtWidgets.QHBoxLayout):
                widget = item.itemAt(1).widget()
                key = widget.property("key")
                type_name = widget.property("type")
                
                if isinstance(widget, QtWidgets.QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    value = widget.value()
                    if type_name == "int":
                        value = int(value)
                else:
                    value = widget.text()
                    if type_name == "int":
                        value = int(value)
                    elif type_name == "float":
                        value = float(value)
                
                settings[key] = value
        
        # Apply settings to logic
        self._analysis_logic().analysis_settings = settings
        
        # Close dialog
        self._analysis_settings_dialog.accept()
    
    def show_nv_settings(self):
        """Show the NV state settings dialog"""
        # Create a simple dialog with threshold and reference settings
        self._nv_settings_dialog.setWindowTitle("NV State Settings")
        
        # Clear existing widgets
        for i in reversed(range(self._nv_settings_dialog.layout.count())):
            widget = self._nv_settings_dialog.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Create threshold setting
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_label = QtWidgets.QLabel("Threshold:")
        threshold_spinbox = QtWidgets.QDoubleSpinBox()
        threshold_spinbox.setRange(0.1, 1.0)
        threshold_spinbox.setSingleStep(0.05)
        threshold_spinbox.setDecimals(2)
        threshold_spinbox.setValue(self._analysis_logic()._nv_threshold)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_spinbox)
        
        # Create reference setting
        reference_layout = QtWidgets.QHBoxLayout()
        reference_label = QtWidgets.QLabel("Reference Level:")
        reference_value = QtWidgets.QLineEdit()
        reference_value.setReadOnly(True)
        if self._analysis_logic()._nv_reference_level is not None:
            reference_value.setText(str(self._analysis_logic()._nv_reference_level))
        else:
            reference_value.setText("Not set")
        
        reference_layout.addWidget(reference_label)
        reference_layout.addWidget(reference_value)
        
        # Create radio buttons for ms state display
        ms_state_layout = QtWidgets.QHBoxLayout()
        ms_state_label = QtWidgets.QLabel("Display state:")
        ms_minus1_radio = QtWidgets.QRadioButton("ms = -1")
        ms_plus1_radio = QtWidgets.QRadioButton("ms = +1")
        
        ms_minus1_radio.setChecked(self._analysis_logic()._display_ms_minus1)
        ms_plus1_radio.setChecked(not self._analysis_logic()._display_ms_minus1)
        
        ms_state_layout.addWidget(ms_state_label)
        ms_state_layout.addWidget(ms_minus1_radio)
        ms_state_layout.addWidget(ms_plus1_radio)
        
        # Add all layouts to the dialog
        self._nv_settings_dialog.layout.addLayout(threshold_layout)
        self._nv_settings_dialog.layout.addLayout(reference_layout)
        self._nv_settings_dialog.layout.addLayout(ms_state_layout)
        
        # Add OK and Cancel buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        
        # Connect OK button to apply settings
        buttons.accepted.connect(lambda: self._apply_nv_settings(
            threshold_spinbox.value(),
            ms_minus1_radio.isChecked()
        ))
        buttons.rejected.connect(self._nv_settings_dialog.reject)
        
        self._nv_settings_dialog.layout.addWidget(buttons)
        
        # Show dialog
        self._nv_settings_dialog.exec_()
    
    def _apply_nv_settings(self, threshold, display_ms_minus1):
        """Apply the settings from the NV settings dialog"""
        # Apply threshold
        self.sigSetThreshold.emit(threshold)
        
        # Apply ms state display
        self.sigToggleMsStateDisplay.emit(display_ms_minus1)
        
        # Update GUI elements
        self._mw.threshold_spinbox.setValue(threshold)
        self._mw.ms_minus1_radiobutton.setChecked(display_ms_minus1)
        self._mw.ms_plus1_radiobutton.setChecked(not display_ms_minus1)
        
        # Close dialog
        self._nv_settings_dialog.accept()
    
    def update_data_display(self, data_info):
        """Update the data display with the loaded data"""
        # Clear any existing plots
        self.clear_plots()
        
        # Show status in statusbar
        self._mw.statusbar.showMessage(f"Loaded {os.path.basename(data_info['file_path'])}")
        
        # Get data from logic
        if data_info['has_signal_data']:
            self._signal_data = self._analysis_logic().signal_data
            self.update_data_plot()
            
            # Create selection region for reference level
            if self._data_selection_region is None:
                self._create_data_selection_region(0, min(10, len(self._signal_data[0])-1))
        
        if data_info['has_laser_data']:
            self._laser_data = self._analysis_logic().laser_data
            self.update_laser_plot()
        
        # Enable/disable buttons based on available data
        self._mw.extract_laser_button.setEnabled(data_info['has_raw_data'])
        self._mw.analyze_button.setEnabled(data_info['has_laser_data'] or data_info['has_signal_data'])
        self._mw.action_extract_laser.setEnabled(data_info['has_raw_data'])
        self._mw.action_analyze_pulses.setEnabled(data_info['has_laser_data'] or data_info['has_signal_data'])
        
        # Update recent files menu
        self._update_recent_files_menu()
    
    def update_analysis_display(self, results):
        """Update the display with analysis results"""
        # Store the results
        self._signal_data = results['signal_data']
        self._nv_state_data = results['nv_state_data']
        
        # Update plots
        self.update_data_plot()
        self.update_nv_plot()
        
        # Show success message
        self._mw.statusbar.showMessage("Analysis complete")
        
        # Enable saving
        self._mw.save_analysis_button.setEnabled(True)
        self._mw.action_save_analysis.setEnabled(True)
        self._mw.action_save_analysis_as.setEnabled(True)
    
    def update_nv_state_display(self, histogram, statistics):
        """Update the NV state histogram and statistics display"""
        self._state_histogram = histogram
        self._state_statistics = statistics
        
        # Update histogram plot
        self.update_histogram_plot()
        
        # Update statistics display
        self._mw.ms0_count_lcd.display(statistics['ms0_count'])
        self._mw.ms1_count_lcd.display(statistics['ms1_count'])
        self._mw.ms0_percentage_lcd.display(f"{statistics['ms0_percentage']:.1f}")
        self._mw.ms1_percentage_lcd.display(f"{statistics['ms1_percentage']:.1f}")
        
        # Update threshold and reference displays
        self.update_threshold_display(statistics['threshold'] / statistics['reference_level'])
        self.update_reference_display(statistics['reference_level'])
    
    def update_threshold_display(self, threshold):
        """Update the threshold display with the current value"""
        # Update spinbox and slider
        self._mw.threshold_spinbox.blockSignals(True)
        self._mw.threshold_slider.blockSignals(True)
        
        self._mw.threshold_spinbox.setValue(threshold)
        self._mw.threshold_slider.setValue(int(threshold * 100))
        
        self._mw.threshold_spinbox.blockSignals(False)
        self._mw.threshold_slider.blockSignals(False)
        
        # Update threshold value display
        if self._analysis_logic()._nv_reference_level is not None:
            threshold_value = threshold * self._analysis_logic()._nv_reference_level
            self._mw.threshold_value_display.setText(f"{threshold_value:.2f}")
    
    def update_reference_display(self, reference):
        """Update the reference level display"""
        self._mw.reference_level_display.setText(f"{reference:.2f}")
        
        # Also update threshold value display
        threshold_value = self._analysis_logic()._nv_threshold * reference
        self._mw.threshold_value_display.setText(f"{threshold_value:.2f}")
    
    def clear_plots(self):
        """Clear all plots and plot items"""
        # Clear data plot
        for curve in self._data_plot_curves:
            self._mw.data_plot_widget.removeItem(curve)
        self._data_plot_curves = []
        
        # Clear laser plot
        for curve in self._laser_plot_curves:
            self._mw.laser_plot_widget.removeItem(curve)
        self._laser_plot_curves = []
        
        # Clear NV state plot
        for curve in self._nv_plot_curves:
            self._mw.nv_plot_widget.removeItem(curve)
        self._nv_plot_curves = []
        
        # Clear histogram plot
        for item in self._histogram_plot_items:
            self._mw.histogram_plot_widget.removeItem(item)
        self._histogram_plot_items = []
        
        # Remove selection region
        if self._data_selection_region is not None:
            self._mw.data_plot_widget.removeItem(self._data_selection_region)
            self._data_selection_region = None
    
    def update_data_plot(self):
        """Update the data plot with current signal data"""
        if self._signal_data is None:
            return
        
        # Clear existing curves
        for curve in self._data_plot_curves:
            self._mw.data_plot_widget.removeItem(curve)
        self._data_plot_curves = []
        
        # Plot main signal data
        curve = pg.PlotDataItem(
            x=self._signal_data[0],
            y=self._signal_data[1],
            pen=pg.mkPen(color=palette.c1, width=2),
            symbol='o',
            symbolSize=5,
            symbolBrush=palette.c1,
            symbolPen=None
        )
        self._mw.data_plot_widget.addItem(curve)
        self._data_plot_curves.append(curve)
        
        # If alternating data, plot second curve
        if self._signal_data.shape[0] > 2:
            curve = pg.PlotDataItem(
                x=self._signal_data[0],
                y=self._signal_data[2],
                pen=pg.mkPen(color=palette.c2, width=2),
                symbol='o',
                symbolSize=5,
                symbolBrush=palette.c2,
                symbolPen=None
            )
            self._mw.data_plot_widget.addItem(curve)
            self._data_plot_curves.append(curve)
        
        # Add threshold line if reference level is set
        if self._analysis_logic()._nv_reference_level is not None:
            threshold = self._analysis_logic()._nv_reference_level * self._analysis_logic()._nv_threshold
            threshold_line = pg.InfiniteLine(
                pos=threshold,
                angle=0,
                pen=pg.mkPen(color=palette.c3, width=2, style=QtCore.Qt.DashLine),
                label=f"Threshold: {threshold:.2f}",
                labelOpts={'position': 0.1, 'color': palette.c3, 'movable': True}
            )
            self._mw.data_plot_widget.addItem(threshold_line)
            self._data_plot_curves.append(threshold_line)
            
            # Also add reference level line
            reference_line = pg.InfiniteLine(
                pos=self._analysis_logic()._nv_reference_level,
                angle=0,
                pen=pg.mkPen(color=palette.c4, width=2, style=QtCore.Qt.DashLine),
                label=f"Reference: {self._analysis_logic()._nv_reference_level:.2f}",
                labelOpts={'position': 0.9, 'color': palette.c4, 'movable': True}
            )
            self._mw.data_plot_widget.addItem(reference_line)
            self._data_plot_curves.append(reference_line)
        
        # Auto range the view
        self._mw.data_plot_widget.autoRange()
    
    def update_laser_plot(self):
        """Update the laser plot with current laser data"""
        if self._laser_data is None:
            return
        
        # Clear existing curves
        for curve in self._laser_plot_curves:
            self._mw.laser_plot_widget.removeItem(curve)
        self._laser_plot_curves = []
        
        # Plot the first few laser pulses
        max_pulses = min(5, self._laser_data.shape[0])
        bin_width = self._analysis_logic().metadata.get('bin width (s)', 1e-9)
        
        for i in range(max_pulses):
            x = np.arange(len(self._laser_data[i])) * bin_width
            curve = pg.PlotDataItem(
                x=x,
                y=self._laser_data[i],
                pen=pg.mkPen(color=palette.get_palette_color(i), width=1)
            )
            self._mw.laser_plot_widget.addItem(curve)
            self._laser_plot_curves.append(curve)
        
        # Auto range the view
        self._mw.laser_plot_widget.autoRange()
    
    def update_nv_plot(self):
        """Update the NV state plot with state classifications"""
        if self._signal_data is None or self._nv_state_data is None:
            return
        
        # Clear existing curves
        for curve in self._nv_plot_curves:
            self._mw.nv_plot_widget.removeItem(curve)
        self._nv_plot_curves = []
        
        # Plot original signal data
        curve = pg.PlotDataItem(
            x=self._signal_data[0],
            y=self._signal_data[1],
            pen=pg.mkPen(color=palette.c1, width=2, alpha=128),
            symbol=None
        )
        self._mw.nv_plot_widget.addItem(curve)
        self._nv_plot_curves.append(curve)
        
        # Plot state classification using different symbols for each state
        # State 0 = ms=0 (bright state)
        mask_ms0 = self._nv_state_data == 0
        if np.any(mask_ms0):
            curve = pg.ScatterPlotItem(
                x=self._signal_data[0][mask_ms0],
                y=self._signal_data[1][mask_ms0],
                symbol='o',
                size=8,
                brush=pg.mkBrush(color=palette.c3),
                pen=None
            )
            self._mw.nv_plot_widget.addItem(curve)
            self._nv_plot_curves.append(curve)
        
        # State 1 = ms=-1/+1 (dark state)
        mask_ms1 = self._nv_state_data == 1
        if np.any(mask_ms1):
            curve = pg.ScatterPlotItem(
                x=self._signal_data[0][mask_ms1],
                y=self._signal_data[1][mask_ms1],
                symbol='s',
                size=8,
                brush=pg.mkBrush(color=palette.c5),
                pen=None
            )
            self._mw.nv_plot_widget.addItem(curve)
            self._nv_plot_curves.append(curve)
        
        # Add threshold line
        if self._analysis_logic()._nv_reference_level is not None:
            threshold = self._analysis_logic()._nv_reference_level * self._analysis_logic()._nv_threshold
            threshold_line = pg.InfiniteLine(
                pos=threshold,
                angle=0,
                pen=pg.mkPen(color=palette.c3, width=2, style=QtCore.Qt.DashLine),
                label=f"Threshold: {threshold:.2f}",
                labelOpts={'position': 0.1, 'color': palette.c3, 'movable': True}
            )
            self._mw.nv_plot_widget.addItem(threshold_line)
            self._nv_plot_curves.append(threshold_line)
        
        # Auto range the view
        self._mw.nv_plot_widget.autoRange()
    
    def update_histogram_plot(self):
        """Update the histogram plot with state statistics"""
        if self._state_histogram is None:
            return
        
        # Clear existing items
        for item in self._histogram_plot_items:
            self._mw.histogram_plot_widget.removeItem(item)
        self._histogram_plot_items = []
        
        # Get the histogram data
        unique, counts = self._state_histogram
        
        # Create the bar graph
        bargraph = pg.BarGraphItem(
            x=unique,
            height=counts,
            width=0.7,
            brush=pg.mkBrush(color=palette.c1)
        )
        self._mw.histogram_plot_widget.addItem(bargraph)
        self._histogram_plot_items.append(bargraph)
        
        # Add labels
        for i, (x, y) in enumerate(zip(unique, counts)):
            label = pg.TextItem(
                text=str(y),
                color=palette.c6,
                anchor=(0.5, 0)
            )
            label.setPos(x, y + 1)
            self._mw.histogram_plot_widget.addItem(label)
            self._histogram_plot_items.append(label)
        
        # Set axis range and labels
        self._mw.histogram_plot_widget.setXRange(-0.5, 1.5)
        self._mw.histogram_plot_widget.getAxis('bottom').setTicks([
            [(0, "ms=0"), (1, "ms=-1" if self._analysis_logic()._display_ms_minus1 else "ms=+1")]
        ])
    
    def _create_data_selection_region(self, start_idx, end_idx):
        """Create a linear region item for data selection on the data plot"""
        # Remove existing region if any
        if self._data_selection_region is not None:
            self._mw.data_plot_widget.removeItem(self._data_selection_region)
        
        # Get x coordinates for indices
        if self._signal_data is not None and len(self._signal_data[0]) > 0:
            x_start = self._signal_data[0][start_idx]
            x_end = self._signal_data[0][end_idx]
        else:
            # Default x values if no data
            x_start = start_idx
            x_end = end_idx
        
        # Create linear region item
        self._data_selection_region = pg.LinearRegionItem(
            values=[x_start, x_end],
            brush=pg.mkBrush(color=palette.c2, alpha=50),
            pen=pg.mkPen(color=palette.c2, width=1),
            movable=True
        )
        self._data_selection_region.sigRegionChanged.connect(self._update_selection_range)
        
        # Add to plot
        self._mw.data_plot_widget.addItem(self._data_selection_region)
        
        # Store the current selection
        self._data_selection_range = (start_idx, end_idx)
    
    def _update_selection_range(self):
        """Update the selection range when the region is changed by the user"""
        if self._signal_data is None or self._data_selection_region is None:
            return
        
        # Get the region boundaries
        region_min, region_max = self._data_selection_region.getRegion()
        
        # Find the closest indices in the x data
        x_data = self._signal_data[0]
        start_idx = np.abs(x_data - region_min).argmin()
        end_idx = np.abs(x_data - region_max).argmin()
        
        # Store the current selection
        self._data_selection_range = (start_idx, end_idx)
    
    def set_reference_from_selection(self):
        """Set the reference level based on the current selection"""
        if self._signal_data is None or self._data_selection_region is None:
            return
        
        # Get the current selection range
        start_idx, end_idx = self._data_selection_range
        
        # Send to logic
        self.sigSetReferenceFromSelection.emit(start_idx, end_idx)
    
    def save_analysis(self):
        """Save the analysis results using the last used path"""
        # Check if analysis results are available
        if self._nv_state_data is None or self._signal_data is None:
            QtWidgets.QMessageBox.warning(
                self._mw,
                "No Results",
                "No analysis results to save."
            )
            return
        
        # Save to default location (let logic handle it)
        self._analysis_logic().save_analysis_results()
        
        # Show success message
        self._mw.statusbar.showMessage("Analysis results saved")
    
    def save_analysis_as(self):
        """Save the analysis results to a specified path"""
        # Check if analysis results are available
        if self._nv_state_data is None or self._signal_data is None:
            QtWidgets.QMessageBox.warning(
                self._mw,
                "No Results",
                "No analysis results to save."
            )
            return
        
        # Determine initial directory
        initial_dir = self._default_save_path
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # Generate suggested filename based on original file
        filename = "nv_state_analysis"
        if self._analysis_logic().current_file_path:
            base_name = os.path.splitext(os.path.basename(self._analysis_logic().current_file_path))[0]
            if "_pulsed_measurement" in base_name:
                base_name = base_name.replace("_pulsed_measurement", "")
            elif "_raw_timetrace" in base_name:
                base_name = base_name.replace("_raw_timetrace", "")
            elif "_laser_pulses" in base_name:
                base_name = base_name.replace("_laser_pulses", "")
            filename = f"{base_name}_nv_analysis"
        
        # Get file format based on the current storage type
        file_format = "Data Files (*.dat)"
        storage_type = self._analysis_logic()._default_data_storage_cls.__name__
        if storage_type == "CsvDataStorage":
            file_format = "CSV Files (*.csv)"
        elif storage_type == "NpyDataStorage":
            file_format = "NumPy Files (*.npy)"
        
        # Open save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._mw,
            "Save Analysis Results",
            os.path.join(initial_dir, filename),
            f"{file_format};;All Files (*)"
        )
        
        if file_path:
            # Update default save path
            self._default_save_path = os.path.dirname(file_path)
            
            # Save to specified path
            self._analysis_logic().save_analysis_results(file_path)
            
            # Show success message
            self._mw.statusbar.showMessage(f"Analysis results saved to {os.path.basename(file_path)}")