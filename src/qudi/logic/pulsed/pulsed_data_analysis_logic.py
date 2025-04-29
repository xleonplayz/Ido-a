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
    # Optional additional paths to import from
    extraction_import_path = ConfigOption(name='additional_extraction_path', default=None)
    analysis_import_path = ConfigOption(name='additional_analysis_path', default=None)
    
    _default_data_storage_cls = ConfigOption(name='default_data_storage_type',
                                           default='text',
                                           constructor=_data_storage_from_cfg_option)
    
    # Status variables
    # Fast counter settings (needed by PulseExtractor)
    __fast_counter_record_length = StatusVar(default=3.0e-6)
    __fast_counter_binwidth = StatusVar(default=1.0e-9)
    __fast_counter_gates = StatusVar(default=0)
    
    # PulseExtractor and PulseAnalyzer settings
    extraction_parameters = StatusVar(default=None)
    analysis_parameters = StatusVar(default=None)
    
    # NV state analysis settings
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
        
    @extraction_parameters.representer
    def __repr_extraction_parameters(self, value):
        return self._pulseextractor.full_settings_dict if hasattr(self, '_pulseextractor') else None
    
    @analysis_parameters.representer
    def __repr_analysis_parameters(self, value):
        return self._pulseanalyzer.full_settings_dict if hasattr(self, '_pulseanalyzer') else None
    
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
        
        self.log.info(f"Attempting to load data from: {file_path}")
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
            # First try to load the selected file, regardless of name pattern
            try:
                data, metadata = storage.load_data(file_path)
                self.log.info(f"Successfully loaded data from {file_path}")
                
                # Determine data type based on shape and content
                if data.ndim == 2 and data.shape[1] >= 2:
                    # Likely pulsed measurement data (x and y columns)
                    self.log.info(f"File appears to be pulsed measurement data with shape {data.shape}")
                    self.signal_data = data.T  # Transpose to match expected format
                    self.metadata = metadata
                elif data.ndim == 2 and data.shape[0] > 1:
                    # Likely laser pulse data (multiple rows/pulses)
                    self.log.info(f"File appears to be laser pulse data with shape {data.shape}")
                    self.laser_data = data
                    self.metadata = metadata
                else:
                    # Likely raw data or generic data
                    self.log.info(f"File appears to be raw data with shape {data.shape}")
                    self.raw_data = data.squeeze()
                    self.metadata = metadata
            except Exception as e:
                self.log.error(f"Error loading the main file {file_path}: {str(e)}")
                # Continue to try loading related files
            
            # Now try to identify file type based on name and load related files
            basename = os.path.basename(file_path)
            dirname = os.path.dirname(file_path)
            
            # Try to get base name without any of the special suffixes
            base_name = basename
            for suffix in ["_pulsed_measurement", "_raw_timetrace", "_laser_pulses"]:
                if suffix in base_name:
                    base_name = base_name.replace(suffix, "")
            
            # File extensions to try
            extensions = [file_extension]  # Start with the same extension as the loaded file
            if file_extension != '.dat':
                extensions.append('.dat')  # .dat is the most common
            if file_extension != '.csv':
                extensions.append('.csv')
            if file_extension != '.npy':
                extensions.append('.npy')
            
            # Try to find related files with all possible extensions
            self.log.info(f"Looking for related files with base name: {base_name}")
            
            # If we don't have pulsed measurement data yet, try to find it
            if self.signal_data is None:
                for ext in extensions:
                    pulsed_file_path = os.path.join(dirname, base_name + "_pulsed_measurement" + ext)
                    self.log.info(f"Checking for pulsed measurement file: {pulsed_file_path}")
                    if os.path.isfile(pulsed_file_path) and pulsed_file_path != file_path:
                        try:
                            data, metadata = storage.load_data(pulsed_file_path)
                            self.signal_data = data.T
                            if not self.metadata:
                                self.metadata = metadata
                            else:
                                self.metadata.update(metadata)
                            self.log.info(f"Loaded pulsed measurement data from {pulsed_file_path}")
                            break
                        except Exception as e:
                            self.log.error(f"Error loading pulsed file {pulsed_file_path}: {str(e)}")
            
            # If we don't have raw data yet, try to find it
            if self.raw_data is None:
                for ext in extensions:
                    raw_file_path = os.path.join(dirname, base_name + "_raw_timetrace" + ext)
                    self.log.info(f"Checking for raw data file: {raw_file_path}")
                    if os.path.isfile(raw_file_path) and raw_file_path != file_path:
                        try:
                            raw_data, raw_metadata = storage.load_data(raw_file_path)
                            self.raw_data = raw_data.squeeze()
                            if not self.metadata:
                                self.metadata = raw_metadata
                            else:
                                self.metadata.update(raw_metadata)
                            self.log.info(f"Loaded raw data from {raw_file_path}")
                            break
                        except Exception as e:
                            self.log.error(f"Error loading raw file {raw_file_path}: {str(e)}")
            
            # If we don't have laser data yet, try to find it
            if self.laser_data is None:
                for ext in extensions:
                    laser_file_path = os.path.join(dirname, base_name + "_laser_pulses" + ext)
                    self.log.info(f"Checking for laser pulses file: {laser_file_path}")
                    if os.path.isfile(laser_file_path) and laser_file_path != file_path:
                        try:
                            laser_data, laser_metadata = storage.load_data(laser_file_path)
                            self.laser_data = laser_data
                            if not self.metadata:
                                self.metadata = laser_metadata
                            else:
                                self.metadata.update(laser_metadata)
                            self.log.info(f"Loaded laser data from {laser_file_path}")
                            break
                        except Exception as e:
                            self.log.error(f"Error loading laser file {laser_file_path}: {str(e)}")
            
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
            
            self.log.info(f"Data loading complete. Raw data: {data_info['has_raw_data']}, "
                         f"Laser data: {data_info['has_laser_data']}, "
                         f"Signal data: {data_info['has_signal_data']}")
            
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error during data loading process for {file_path}: {str(e)}")
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
        # Check if we have laser data to analyze
        if self.laser_data is None:
            self.log.error("No laser data available for analysis")
            
            # If signal data is already loaded (e.g., from a _pulsed_measurement file),
            # we can still perform NV state analysis on that
            if self.signal_data is not None:
                self.log.info("Using already loaded signal data for NV state analysis")
                self.analyze_nv_states()
                
                # Emit results with what we have
                self.sigAnalysisComplete.emit({
                    'signal_data': self.signal_data,
                    'analysis_results': {},
                    'nv_state_data': self.nv_state_data,
                    'state_statistics': self.state_statistics
                })
                return None, None
            else:
                return None, None
        
        try:
            self.log.info(f"Analyzing laser data with shape {self.laser_data.shape}")
            signal, error = self._pulseanalyzer.analyse_laser_pulses(self.laser_data)
            self.log.info(f"Laser pulse analysis complete, got signal with length {len(signal)}")
            
            # Create signal data array if not already loaded
            if self.signal_data is None:
                self.log.info("Creating new signal data array from analyzed laser pulses")
                # Check if controlled_variable is in metadata
                if 'Controlled variable' in self.metadata:
                    x_data = np.array(self.metadata['Controlled variable'])
                    self.log.info(f"Using 'Controlled variable' from metadata with length {len(x_data)}")
                elif 'controlled_variable' in self.metadata:
                    x_data = np.array(self.metadata['controlled_variable'])
                    self.log.info(f"Using 'controlled_variable' from metadata with length {len(x_data)}")
                else:
                    # Create default x data
                    x_data = np.arange(len(signal))
                    self.log.info(f"No controlled variable in metadata, creating default x data with length {len(x_data)}")
                
                # Determine signal dimension based on alternating flag
                alternating = self.metadata.get('alternating', False)
                signal_dim = 3 if alternating else 2
                
                # Check if dimensions match
                if alternating and len(x_data) * 2 != len(signal):
                    self.log.warning(f"Mismatch in data dimensions: x_data length ({len(x_data)}) * 2 != signal length ({len(signal)})")
                    self.log.info("Adjusting x_data to match signal data")
                    x_data = np.arange(len(signal) // 2) if alternating else np.arange(len(signal))
                elif not alternating and len(x_data) != len(signal):
                    self.log.warning(f"Mismatch in data dimensions: x_data length ({len(x_data)}) != signal length ({len(signal)})")
                    self.log.info("Adjusting x_data to match signal data")
                    x_data = np.arange(len(signal))
                
                # Create signal data array
                self.signal_data = np.zeros((signal_dim, len(x_data)), dtype=float)
                self.signal_data[0] = x_data
                
                # Populate signal data
                if alternating:
                    self.log.info(f"Processing alternating data, signal length: {len(signal)}")
                    # Exclude laser pulses to ignore if present
                    laser_ignore_list = self.metadata.get('Laser ignore indices', [])
                    laser_ignore_list = self.metadata.get('laser_ignore_list', laser_ignore_list)
                    
                    if len(laser_ignore_list) > 0:
                        self.log.info(f"Excluding {len(laser_ignore_list)} laser pulses from analysis")
                        signal = np.delete(signal, laser_ignore_list)
                        error = np.delete(error, laser_ignore_list)
                    
                    # Make sure the signal length is even for alternating data
                    if len(signal) % 2 != 0:
                        self.log.warning(f"Odd number of signal points ({len(signal)}) for alternating data, trimming last point")
                        signal = signal[:-1]
                        error = error[:-1]
                    
                    # Split alternating data
                    self.signal_data[1] = signal[::2]
                    self.signal_data[2] = signal[1::2]
                    self.log.info(f"Alternating data processed, signal_data shape: {self.signal_data.shape}")
                else:
                    self.log.info(f"Processing non-alternating data, signal length: {len(signal)}")
                    # Make sure lengths match
                    if len(signal) != len(x_data):
                        self.log.warning(f"Signal length ({len(signal)}) doesn't match x_data length ({len(x_data)}), trimming to shorter")
                        min_len = min(len(signal), len(x_data))
                        self.signal_data[0] = x_data[:min_len]
                        self.signal_data[1] = signal[:min_len]
                    else:
                        self.signal_data[1] = signal
                    
                    self.log.info(f"Non-alternating data processed, signal_data shape: {self.signal_data.shape}")
            else:
                self.log.info(f"Using existing signal_data with shape {self.signal_data.shape}")
            
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
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
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
            self.log.info(f"Analyzing NV states from signal data with shape {self.signal_data.shape}")
            
            # Get reference level
            reference = self._nv_reference_level
            if reference is None:
                # If no reference set, use the mean as reference
                reference = np.mean(self.signal_data[1])
                self._nv_reference_level = reference
                self.log.info(f"No reference level set, using mean value: {reference}")
            else:
                self.log.info(f"Using existing reference level: {reference}")
            
            # Determine threshold value
            threshold = reference * self._nv_threshold
            self.log.info(f"Using threshold value: {threshold} (reference: {reference} * threshold factor: {self._nv_threshold})")
            
            # Assign states based on threshold
            # If counts > threshold: ms=0 state (state=0)
            # If counts <= threshold: ms=-1 or ms=+1 state (state=1)
            states = np.zeros_like(self.signal_data[1])
            states[self.signal_data[1] <= threshold] = 1
            
            # Log how many points are assigned to each state
            ms0_count = np.sum(states == 0)
            ms1_count = np.sum(states == 1)
            self.log.info(f"Assigned states: ms=0: {ms0_count}, ms=-1/+1: {ms1_count}")
            
            self.nv_state_data = states
            
            # Calculate histogram and statistics
            unique, counts = np.unique(states, return_counts=True)
            self.state_histogram = (unique, counts)
            
            total_counts = len(states)
            
            self.state_statistics = {
                'total_counts': total_counts,
                'ms0_count': ms0_count,
                'ms1_count': ms1_count,
                'ms0_percentage': (ms0_count / total_counts) * 100 if total_counts > 0 else 0,
                'ms1_percentage': (ms1_count / total_counts) * 100 if total_counts > 0 else 0,
                'threshold': threshold,
                'reference_level': reference
            }
            
            self.log.info(f"NV state statistics: {self.state_statistics}")
            
            # Emit histogram and statistics
            self.sigNvStateHistogramUpdated.emit(self.state_histogram, self.state_statistics)
            
            return states
            
        except Exception as e:
            self.log.error(f"Error analyzing NV states: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
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
    def fast_counter_settings(self):
        """Return fast counter settings (required by PulseExtractor)"""
        settings_dict = dict()
        settings_dict['bin_width'] = float(self.__fast_counter_binwidth)
        settings_dict['record_length'] = float(self.__fast_counter_record_length)
        settings_dict['number_of_gates'] = int(self.__fast_counter_gates)
        settings_dict['is_gated'] = False  # We don't have a real fast counter
        return settings_dict
    
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