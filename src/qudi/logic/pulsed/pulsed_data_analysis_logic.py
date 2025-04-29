# -*- coding: utf-8 -*-
"""
Logic module for analyzing imported pulsed measurement data with a focus on NV center state analysis.

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
import datetime
import numpy as np
from PySide2 import QtCore

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.core.module import LogicBase
from qudi.util.mutex import Mutex
from qudi.util.datastorage import TextDataStorage, CsvDataStorage, NpyDataStorage
from qudi.util.colordefs import QudiMatplotlibStyle
from qudi.logic.pulsed.pulse_analyzer import PulseAnalyzer
from qudi.logic.pulsed.pulse_extractor import PulseExtractor


def _data_storage_from_cfg_option(cfg_str):
    """Helper method to get the appropriate data storage class"""
    cfg_str = cfg_str.lower()
    if cfg_str == 'text':
        return TextDataStorage
    if cfg_str == 'csv':
        return CsvDataStorage
    if cfg_str == 'npy':
        return NpyDataStorage
    raise ValueError('Invalid ConfigOption value to specify data storage type.')


class PulsedDataAnalysisLogic(LogicBase):
    """
    Logic module for analyzing imported pulsed measurement data with a focus on NV center state analysis.
    
    Example config:
    pulsed_data_analysis_logic:
        module.Class: 'pulsed.pulsed_data_analysis_logic.PulsedDataAnalysisLogic'
        options:
            default_data_storage_type: 'text'
        connect:
            pulseanalyzer: 'pulseanalyzer'
    """
    
    # No connectors needed as we instantiate helpers directly
    
    # Config options
    _default_data_storage_cls = ConfigOption(name='default_data_storage_type',
                                           default='text',
                                           constructor=_data_storage_from_cfg_option)
    
    # Status variables
    _nv_threshold = StatusVar(default=0.7)  # Threshold for NV state identification
    _nv_reference_level = StatusVar(default=None)  # Reference level for ms=0
    _display_ms_minus1 = StatusVar(default=True)  # Whether to display ms=-1 state (True) or ms=+1 (False)
    _recent_files = StatusVar(default=[])  # List of recently loaded files
    
    # Signals for GUI communication
    sigDataLoaded = QtCore.Signal(dict)
    sigAnalysisComplete = QtCore.Signal(dict)
    sigNvStateHistogramUpdated = QtCore.Signal(object, object)
    sigThresholdChanged = QtCore.Signal(float)
    sigReferenceChanged = QtCore.Signal(float)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._threadlock = Mutex()
        
        # Data containers
        self.raw_data = None
        self.laser_data = None
        self.signal_data = None
        self.metadata = None
        
        # Analysis results
        self.analysis_results = None
        self.nv_state_data = None
        self.state_histogram = None
        self.state_statistics = None
        
        # File information
        self.current_file_path = None
        self.current_file_type = None
        
        # Create pulse extractor and analyzer instances
        self._pulseextractor = None
        self._pulseanalyzer = None
        
    def on_activate(self):
        """Initializes the module when activated"""
        # Directly create helper instances
        self._pulseextractor = PulseExtractor(pulsedmeasurementlogic=self)
        self._pulseanalyzer = PulseAnalyzer(pulsedmeasurementlogic=self)
        
        # Initialize data containers
        self._initialize_data_containers()
    
    def on_deactivate(self):
        """Cleanup when module is deactivated"""
        pass
    
    def _initialize_data_containers(self):
        """Initialize all data containers with empty values"""
        self.raw_data = None
        self.laser_data = None
        self.signal_data = None
        self.metadata = {}
        self.analysis_results = {}
        self.nv_state_data = None
        self.state_histogram = None
        self.state_statistics = {}
    
    def load_data(self, file_path):
        """
        Load data from a saved pulsed measurement file
        
        @param str file_path: path to the file to load
        @return dict: metadata of the loaded file
        """
        if not os.path.isfile(file_path):
            self.log.error(f"File does not exist: {file_path}")
            return
        
        self._initialize_data_containers()
        self.current_file_path = file_path
        
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.dat':
            storage = TextDataStorage()
            self.current_file_type = 'text'
        elif file_extension == '.csv':
            storage = CsvDataStorage()
            self.current_file_type = 'csv'
        elif file_extension == '.npy':
            storage = NpyDataStorage()
            self.current_file_type = 'npy'
        else:
            self.log.error(f"Unsupported file type: {file_extension}")
            return
        
        try:
            # First try to load pulsed measurement data
            if "_pulsed_measurement" in file_path:
                data, metadata = storage.load_data(file_path)
                self.signal_data = data.T  # Transpose to match expected format
                self.metadata = metadata
                
                # Try to find and load the corresponding raw data file
                raw_file_path = file_path.replace("_pulsed_measurement", "_raw_timetrace")
                if os.path.isfile(raw_file_path):
                    raw_data, raw_metadata = storage.load_data(raw_file_path)
                    self.raw_data = raw_data.squeeze()
                    self.metadata.update(raw_metadata)
                
                # Try to find and load the corresponding laser pulses file
                laser_file_path = file_path.replace("_pulsed_measurement", "_laser_pulses")
                if os.path.isfile(laser_file_path):
                    laser_data, laser_metadata = storage.load_data(laser_file_path)
                    self.laser_data = laser_data
                    self.metadata.update(laser_metadata)
            
            # If not a pulsed measurement file, try to load raw data
            elif "_raw_timetrace" in file_path:
                raw_data, raw_metadata = storage.load_data(file_path)
                self.raw_data = raw_data.squeeze()
                self.metadata = raw_metadata
                
                # Try to find and load the corresponding files
                pulsed_file_path = file_path.replace("_raw_timetrace", "_pulsed_measurement")
                if os.path.isfile(pulsed_file_path):
                    data, metadata = storage.load_data(pulsed_file_path)
                    self.signal_data = data.T
                    self.metadata.update(metadata)
                
                laser_file_path = file_path.replace("_raw_timetrace", "_laser_pulses")
                if os.path.isfile(laser_file_path):
                    laser_data, laser_metadata = storage.load_data(laser_file_path)
                    self.laser_data = laser_data
                    self.metadata.update(laser_metadata)
            
            # If laser pulses file, load it
            elif "_laser_pulses" in file_path:
                laser_data, laser_metadata = storage.load_data(file_path)
                self.laser_data = laser_data
                self.metadata = laser_metadata
                
                # Try to find and load the corresponding files
                raw_file_path = file_path.replace("_laser_pulses", "_raw_timetrace")
                if os.path.isfile(raw_file_path):
                    raw_data, raw_metadata = storage.load_data(raw_file_path)
                    self.raw_data = raw_data.squeeze()
                    self.metadata.update(raw_metadata)
                
                pulsed_file_path = file_path.replace("_laser_pulses", "_pulsed_measurement")
                if os.path.isfile(pulsed_file_path):
                    data, metadata = storage.load_data(pulsed_file_path)
                    self.signal_data = data.T
                    self.metadata.update(metadata)
            
            else:
                # Generic file, try to load as raw data
                data, metadata = storage.load_data(file_path)
                self.raw_data = data.squeeze()
                self.metadata = metadata
            
            # Add to recent files list
            if file_path in self._recent_files:
                self._recent_files.remove(file_path)
            self._recent_files.insert(0, file_path)
            # Limit to 10 recent files
            self._recent_files = self._recent_files[:10]
            
            # Emit signal with loaded data info
            data_info = {
                'has_raw_data': self.raw_data is not None,
                'has_laser_data': self.laser_data is not None,
                'has_signal_data': self.signal_data is not None,
                'file_path': file_path,
                'metadata': self.metadata
            }
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error loading data from {file_path}: {str(e)}")
            return
    
    def extract_laser_pulses(self):
        """
        Extract laser pulses from raw data if available
        
        @return dict: result from pulse extraction
        """
        if self.raw_data is None:
            self.log.error("No raw data available for pulse extraction")
            return
        
        try:
            return_dict = self._pulseextractor.extract_laser_pulses(self.raw_data)
            self.laser_data = return_dict['laser_counts_arr']
            return return_dict
        except Exception as e:
            self.log.error(f"Error extracting laser pulses: {str(e)}")
            return None
    
    def analyze_laser_pulses(self):
        """
        Analyze extracted laser pulses
        
        @return tuple: (signal_data, error_data)
        """
        if self.laser_data is None:
            self.log.error("No laser data available for analysis")
            return None, None
        
        try:
            signal, error = self._pulseanalyzer.analyse_laser_pulses(self.laser_data)
            
            # Create signal data array if not already loaded
            if self.signal_data is None:
                # Check if controlled_variable is in metadata
                if 'Controlled variable' in self.metadata:
                    x_data = np.array(self.metadata['Controlled variable'])
                elif 'controlled_variable' in self.metadata:
                    x_data = np.array(self.metadata['controlled_variable'])
                else:
                    # Create default x data
                    x_data = np.arange(len(signal))
                
                # Determine signal dimension based on alternating flag
                alternating = self.metadata.get('alternating', False)
                signal_dim = 3 if alternating else 2
                
                self.signal_data = np.zeros((signal_dim, len(x_data)), dtype=float)
                self.signal_data[0] = x_data
                
                # Populate signal data
                if alternating:
                    # Exclude laser pulses to ignore if present
                    laser_ignore_list = self.metadata.get('Laser ignore indices', [])
                    laser_ignore_list = self.metadata.get('laser_ignore_list', laser_ignore_list)
                    
                    if len(laser_ignore_list) > 0:
                        signal = np.delete(signal, laser_ignore_list)
                        error = np.delete(error, laser_ignore_list)
                    
                    # Split alternating data
                    self.signal_data[1] = signal[::2]
                    self.signal_data[2] = signal[1::2]
                else:
                    self.signal_data[1] = signal
            
            self.analysis_results = {
                'signal': signal,
                'error': error
            }
            
            # Perform NV state analysis
            self.analyze_nv_states()
            
            # Emit results
            self.sigAnalysisComplete.emit({
                'signal_data': self.signal_data,
                'analysis_results': self.analysis_results,
                'nv_state_data': self.nv_state_data,
                'state_statistics': self.state_statistics
            })
            
            return signal, error
            
        except Exception as e:
            self.log.error(f"Error analyzing laser pulses: {str(e)}")
            return None, None
    
    def analyze_nv_states(self):
        """
        Analyze NV states based on count data using threshold
        
        @return ndarray: Array with state assignments (0: ms=0, 1: ms=-1/+1)
        """
        if self.signal_data is None or self.signal_data.shape[0] < 2:
            self.log.error("No signal data available for NV state analysis")
            return None
        
        try:
            # Get reference level
            reference = self._nv_reference_level
            if reference is None:
                # If no reference set, use the mean as reference
                reference = np.mean(self.signal_data[1])
                self._nv_reference_level = reference
            
            # Determine threshold value
            threshold = reference * self._nv_threshold
            
            # Assign states based on threshold
            # If counts > threshold: ms=0 state (state=0)
            # If counts <= threshold: ms=-1 or ms=+1 state (state=1)
            states = np.zeros_like(self.signal_data[1])
            states[self.signal_data[1] <= threshold] = 1
            
            self.nv_state_data = states
            
            # Calculate histogram and statistics
            unique, counts = np.unique(states, return_counts=True)
            self.state_histogram = (unique, counts)
            
            total_counts = len(states)
            ms0_count = np.sum(states == 0)
            ms1_count = np.sum(states == 1)
            
            self.state_statistics = {
                'total_counts': total_counts,
                'ms0_count': ms0_count,
                'ms1_count': ms1_count,
                'ms0_percentage': (ms0_count / total_counts) * 100 if total_counts > 0 else 0,
                'ms1_percentage': (ms1_count / total_counts) * 100 if total_counts > 0 else 0,
                'threshold': threshold,
                'reference_level': reference
            }
            
            # Emit histogram and statistics
            self.sigNvStateHistogramUpdated.emit(self.state_histogram, self.state_statistics)
            
            return states
            
        except Exception as e:
            self.log.error(f"Error analyzing NV states: {str(e)}")
            return None
    
    def set_nv_threshold(self, threshold):
        """
        Set threshold for NV state identification
        
        @param float threshold: Threshold value (0-1)
        """
        if 0 <= threshold <= 1:
            self._nv_threshold = threshold
            self.sigThresholdChanged.emit(threshold)
            # Re-analyze states if data is available
            if self.signal_data is not None:
                self.analyze_nv_states()
    
    def set_reference_from_selection(self, start_idx, end_idx):
        """
        Set reference level based on user selection
        
        @param int start_idx: Start index of selection
        @param int end_idx: End index of selection
        """
        if self.signal_data is None or self.signal_data.shape[0] < 2:
            return
        
        if start_idx < 0 or end_idx >= len(self.signal_data[1]) or start_idx > end_idx:
            self.log.error("Invalid selection range for reference level")
            return
        
        # Calculate mean of selected range
        selected_data = self.signal_data[1][start_idx:end_idx+1]
        reference = np.mean(selected_data)
        
        self._nv_reference_level = reference
        self.sigReferenceChanged.emit(reference)
        
        # Re-analyze states
        self.analyze_nv_states()
    
    def toggle_ms_state_display(self, display_ms_minus1):
        """
        Toggle between displaying ms=-1 or ms=+1 states
        
        @param bool display_ms_minus1: If True, display ms=-1, else ms=+1
        """
        self._display_ms_minus1 = display_ms_minus1
    
    def save_analysis_results(self, file_path=None, tag=None):
        """
        Save analysis results to file
        
        @param str file_path: Path to save file, if None use default path
        @param str tag: Optional tag to add to filename
        @return str: Path to saved file
        """
        if self.nv_state_data is None or self.signal_data is None:
            self.log.error("No analysis results to save")
            return
        
        # Use default data storage type if not specified
        storage_cls = self._default_data_storage_cls
        
        # Determine file path
        if file_path is None:
            # Use same directory as current file
            if self.current_file_path:
                data_dir = os.path.dirname(self.current_file_path)
                filename = None
            else:
                data_dir = self.module_default_data_dir
                filename = None
        else:
            data_dir, filename = os.path.split(file_path)
        
        # Create timestamp
        timestamp = datetime.datetime.now()
        
        # Initialize data storage object
        data_storage = storage_cls(root_dir=data_dir)
        
        # Prepare data to save
        save_data = np.vstack((
            self.signal_data[0],  # x data
            self.signal_data[1],  # signal
            self.nv_state_data    # state classification
        )).T
        
        # Prepare metadata
        metadata = {
            'original_file': self.current_file_path,
            'nv_threshold': self._nv_threshold,
            'reference_level': self._nv_reference_level,
            'ms_state_label': 'ms=-1' if self._display_ms_minus1 else 'ms=+1',
            'state_statistics': self.state_statistics
        }
        
        # Add original metadata
        if self.metadata:
            metadata['original_metadata'] = self.metadata
        
        # Get correct filename
        if filename is None:
            base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
            if tag:
                nametag = f"{base_name}_{tag}_nv_analysis"
            else:
                nametag = f"{base_name}_nv_analysis"
        else:
            nametag = None
        
        # Save data
        save_path, _, _ = data_storage.save_data(
            save_data,
            metadata=metadata,
            nametag=nametag,
            filename=filename,
            timestamp=timestamp,
            column_headers=['x_data', 'counts', 'nv_state']
        )
        
        return save_path
    
    @property
    def extraction_settings(self):
        """Return current extraction settings"""
        return self._pulseextractor.extraction_settings
    
    @extraction_settings.setter
    def extraction_settings(self, settings_dict):
        """Set extraction settings"""
        if isinstance(settings_dict, dict):
            self._pulseextractor.extraction_settings = settings_dict
    
    @property
    def analysis_settings(self):
        """Return current analysis settings"""
        return self._pulseanalyzer.analysis_settings
    
    @analysis_settings.setter
    def analysis_settings(self, settings_dict):
        """Set analysis settings"""
        if isinstance(settings_dict, dict):
            self._pulseanalyzer.analysis_settings = settings_dict