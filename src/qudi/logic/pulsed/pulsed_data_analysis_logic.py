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
    
    This module supports direct loading of individual file types through these methods:
    - load_pulsed_file: Load a pulsed measurement file (signal data)
    - load_raw_file: Load a raw timetrace file (raw data)
    - load_laser_file: Load a laser pulses file (laser data)
    
    This simplifies the workflow by allowing users to load files individually without
    relying on automatic detection of related files.
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
        
        # Validation status for files
        self._pulsed_file_valid = False
        self._raw_file_valid = False
        self._laser_file_valid = False
        
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
        
        # Reset validation status
        self._pulsed_file_valid = False
        self._raw_file_valid = False
        self._laser_file_valid = False
    
    def validate_pulsed_measurement_data(self, data):
        """
        Validate if the loaded data has the correct format for pulsed measurement data
        
        @param ndarray data: The data to validate
        @return tuple: (is_valid, error_message)
        """
        # Check if data is None
        if data is None:
            return False, "Data is None"
            
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            return False, f"Data is not a numpy array but {type(data)}"
        
        # Basic dimensionality check
        if data.ndim < 2:
            return False, f"Data must be at least 2-dimensional, but has {data.ndim} dimensions"
            
        # For pulsed measurement, we expect data with at least 2 columns (x and y values)
        if data.shape[1] < 2:  # Note: we check before transpose
            return False, f"Pulsed measurement data must have at least 2 columns, but has {data.shape[1]}"
            
        # Check data types
        if not np.issubdtype(data.dtype, np.number):
            return False, f"Data must contain numeric values, but has dtype {data.dtype}"
            
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Data contains NaN or infinite values"
            
        # Additional checks specific to your application could be added here
        
        return True, "Data is valid for pulsed measurement"
    
    def load_pulsed_file(self, file_path):
        """
        Load data specifically from a pulsed measurement file
        
        @param str file_path: path to the pulsed measurement file to load
        @return dict: metadata of the loaded file
        """
        if not os.path.isfile(file_path):
            self.log.error(f"Pulsed measurement file does not exist: {file_path}")
            return
            
        self.log.info(f"Loading pulsed measurement file: {file_path}")
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
            # Load the pulsed measurement file
            data, metadata = storage.load_data(file_path)
            self.log.info(f"Successfully loaded pulsed measurement data from {file_path}")
            
            # Extra debug information
            self.log.info(f"Loaded data shape: {data.shape}, type: {data.dtype}")
            self.log.info(f"Data sample (first 5 values): {data.flatten()[:5]}")
            
            # Validate the data
            is_valid, validation_message = self.validate_pulsed_measurement_data(data)
            if not is_valid:
                self.log.error(f"Validation failed for pulsed measurement data: {validation_message}")
                self.log.error("Please check if this is the correct file type.")
                self._pulsed_file_valid = False
                return None
            
            self._pulsed_file_valid = True
            self.log.info(f"Data validation passed: {validation_message}")
            
            # Set as signal data (pulsed measurement data)
            self.signal_data = data.T  # Transpose to match expected format
            self.metadata = metadata
            
            # Verify data was properly set
            if self.signal_data is None:
                self.log.error("Failed to set signal_data variable after loading")
            else:
                self.log.info(f"Signal data set successfully with shape: {self.signal_data.shape}")
            
            # Add to recent files list
            if file_path in self._recent_files:
                self._recent_files.remove(file_path)
            self._recent_files.insert(0, file_path)
            self._recent_files = self._recent_files[:10]  # Limit to 10
            
            # Emit signal with loaded data info
            data_info = {
                'has_raw_data': self.raw_data is not None,
                'has_laser_data': self.laser_data is not None,
                'has_signal_data': self.signal_data is not None,
                'file_path': file_path,
                'metadata': self.metadata
            }
            
            self.log.info(f"Pulsed measurement file loaded successfully.")
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error loading pulsed measurement file {file_path}: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def validate_raw_timetrace_data(self, data):
        """
        Validate if the loaded data has the correct format for raw timetrace data
        
        @param ndarray data: The data to validate
        @return tuple: (is_valid, error_message)
        """
        # Check if data is None
        if data is None:
            return False, "Data is None"
            
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            return False, f"Data is not a numpy array but {type(data)}"
        
        # For raw time trace, we expect a 1D array or a 2D array that can be squeezed to 1D
        if data.ndim > 2:
            return False, f"Raw timetrace data must be 1D or 2D, but has {data.ndim} dimensions"
            
        # Check data size - raw data should have significant number of points
        if data.size < 100:  # Arbitrary threshold, adjust as needed
            return False, f"Raw timetrace data has only {data.size} points, which seems too few"
            
        # Check data types
        if not np.issubdtype(data.dtype, np.number):
            return False, f"Data must contain numeric values, but has dtype {data.dtype}"
            
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Data contains NaN or infinite values"
            
        # Check for all zeros
        if np.all(data == 0):
            return False, "Raw data contains only zeros"
            
        # For time trace data, we might expect some variation in the signal
        if np.std(data) < 1e-10:  # Very small standard deviation
            return False, "Raw data has almost no variation, which is unusual for time trace data"
        
        return True, "Data is valid for raw timetrace"
    
    def load_raw_file(self, file_path):
        """
        Load data specifically from a raw timetrace file
        
        @param str file_path: path to the raw timetrace file to load
        @return dict: metadata of the loaded file
        """
        if not os.path.isfile(file_path):
            self.log.error(f"Raw timetrace file does not exist: {file_path}")
            return
            
        self.log.info(f"Loading raw timetrace file: {file_path}")
        # Don't initialize containers, we want to keep other data if already loaded
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
            # Load the raw timetrace file
            data, metadata = storage.load_data(file_path)
            self.log.info(f"Successfully loaded raw timetrace data from {file_path}")
            
            # Extra debug information
            self.log.info(f"Loaded data shape: {data.shape}, type: {data.dtype}")
            self.log.info(f"Data sample (first 5 values): {data.flatten()[:5]}")
            
            # Validate the data
            is_valid, validation_message = self.validate_raw_timetrace_data(data)
            if not is_valid:
                self.log.error(f"Validation failed for raw timetrace data: {validation_message}")
                self.log.error("Please check if this is the correct file type.")
                self._raw_file_valid = False
                return None
            
            self._raw_file_valid = True
            self.log.info(f"Data validation passed: {validation_message}")
            
            # Set as raw data
            self.raw_data = data.squeeze()
            
            # Verify data was properly set
            if self.raw_data is None:
                self.log.error("Failed to set raw_data variable after loading")
            else:
                self.log.info(f"Raw data set successfully with shape: {self.raw_data.shape}")
            
            # Update metadata
            if not self.metadata:
                self.metadata = metadata
            else:
                self.metadata.update(metadata)
            
            # Add to recent files list
            if file_path in self._recent_files:
                self._recent_files.remove(file_path)
            self._recent_files.insert(0, file_path)
            self._recent_files = self._recent_files[:10]  # Limit to 10
            
            # Emit signal with loaded data info
            data_info = {
                'has_raw_data': self.raw_data is not None,
                'has_laser_data': self.laser_data is not None,
                'has_signal_data': self.signal_data is not None,
                'file_path': file_path,
                'metadata': self.metadata
            }
            
            self.log.info(f"Raw timetrace file loaded successfully.")
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error loading raw timetrace file {file_path}: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def validate_laser_pulses_data(self, data):
        """
        Validate if the loaded data has the correct format for laser pulses data
        
        @param ndarray data: The data to validate
        @return tuple: (is_valid, error_message)
        """
        # Check if data is None
        if data is None:
            return False, "Data is None"
            
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            return False, f"Data is not a numpy array but {type(data)}"
        
        # For laser pulses, we expect a 2D array (multiple pulses, each with multiple time points)
        if data.ndim != 2:
            return False, f"Laser pulses data must be 2D, but has {data.ndim} dimensions"
            
        # Check that we have a reasonable number of pulses and time points
        if data.shape[0] < 1:
            return False, "No laser pulses found in data"
            
        if data.shape[1] < 10:  # Expecting at least some time points per pulse
            return False, f"Laser pulses have only {data.shape[1]} time points, which seems too few"
            
        # Check data types
        if not np.issubdtype(data.dtype, np.number):
            return False, f"Data must contain numeric values, but has dtype {data.dtype}"
            
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Data contains NaN or infinite values"
            
        # Check for all zeros
        if np.all(data == 0):
            return False, "Laser pulses data contains only zeros"
            
        # For laser pulses, we expect non-negative counts
        if np.any(data < 0):
            return False, "Laser pulses data contains negative values, which is invalid for count data"
            
        # For laser pulses, we expect some pulses to have non-zero values
        # Count pulses with significant counts (sum > 0)
        pulses_with_counts = np.sum(np.sum(data, axis=1) > 0)
        if pulses_with_counts == 0:
            return False, "No laser pulses have any counts"
            
        # Additional check: For most laser pulse data, the pulses should have similar shapes
        # Calculate variation in pulse height (max value per pulse)
        pulse_heights = np.max(data, axis=1)
        if pulse_heights.size > 1:  # Only if we have more than one pulse
            height_variation = np.std(pulse_heights) / np.mean(pulse_heights)
            if height_variation > 5.0:  # Very high variation
                self.log.warning(f"Unusual variation in laser pulse heights: {height_variation:.2f}")
                # Don't fail validation, just warn
        
        return True, "Data is valid for laser pulses"
    
    def load_laser_file(self, file_path):
        """
        Load data specifically from a laser pulses file
        
        @param str file_path: path to the laser pulses file to load
        @return dict: metadata of the loaded file
        """
        if not os.path.isfile(file_path):
            self.log.error(f"Laser pulses file does not exist: {file_path}")
            return
            
        self.log.info(f"Loading laser pulses file: {file_path}")
        # Don't initialize containers, we want to keep other data if already loaded
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
            # Load the laser pulses file
            data, metadata = storage.load_data(file_path)
            self.log.info(f"Successfully loaded laser pulses data from {file_path}")
            
            # Extra debug information
            self.log.info(f"Loaded data shape: {data.shape}, type: {data.dtype}")
            self.log.info(f"Data sample (first 5 values): {data.flatten()[:5]}")
            
            # Validate the data
            is_valid, validation_message = self.validate_laser_pulses_data(data)
            if not is_valid:
                self.log.error(f"Validation failed for laser pulses data: {validation_message}")
                self.log.error("Please check if this is the correct file type.")
                self._laser_file_valid = False
                return None
            
            self._laser_file_valid = True
            self.log.info(f"Data validation passed: {validation_message}")
            
            # Set as laser data
            self.laser_data = data
            
            # Verify data was properly set
            if self.laser_data is None:
                self.log.error("Failed to set laser_data variable after loading")
            else:
                self.log.info(f"Laser data set successfully with shape: {self.laser_data.shape}")
            
            # Update metadata
            if not self.metadata:
                self.metadata = metadata
            else:
                self.metadata.update(metadata)
            
            # Add to recent files list
            if file_path in self._recent_files:
                self._recent_files.remove(file_path)
            self._recent_files.insert(0, file_path)
            self._recent_files = self._recent_files[:10]  # Limit to 10
            
            # Emit signal with loaded data info
            data_info = {
                'has_raw_data': self.raw_data is not None,
                'has_laser_data': self.laser_data is not None,
                'has_signal_data': self.signal_data is not None,
                'file_path': file_path,
                'metadata': self.metadata
            }
            
            self.log.info(f"Laser pulses file loaded successfully.")
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error loading laser pulses file {file_path}: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def load_data(self, file_path):
        """
        Legacy method to load data from a saved pulsed measurement file
        and attempt to find related files automatically.
        
        @param str file_path: path to the file to load
        @return dict: metadata of the loaded file
        """
        if not os.path.isfile(file_path):
            self.log.error(f"File does not exist: {file_path}")
            return
        
        self.log.info(f"Attempting to load data from: {file_path}")
        self.log.debug(f"Current working directory: {os.getcwd()}")
        self._initialize_data_containers()
        self.current_file_path = file_path
        
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        self.log.debug(f"File extension: {file_extension}")
        
        if file_extension == '.dat':
            storage = TextDataStorage()
            self.current_file_type = 'text'
            self.log.debug("Using TextDataStorage for file loading")
        elif file_extension == '.csv':
            storage = CsvDataStorage()
            self.current_file_type = 'csv'
            self.log.debug("Using CsvDataStorage for file loading")
        elif file_extension == '.npy':
            storage = NpyDataStorage()
            self.current_file_type = 'npy'
            self.log.debug("Using NpyDataStorage for file loading")
        else:
            self.log.error(f"Unsupported file type: {file_extension}")
            return
        
        try:
            # First try to load the selected file, regardless of name pattern
            initial_load_successful = False
            try:
                data, metadata = storage.load_data(file_path)
                initial_load_successful = True
                self.log.info(f"Successfully loaded data from {file_path}")
                self.log.debug(f"Data shape: {data.shape}, Data type: {data.dtype}")
                self.log.debug(f"Metadata keys: {list(metadata.keys())}")
                
                # Determine data type based on shape and content
                if data.ndim == 2 and data.shape[1] >= 2:
                    # Likely pulsed measurement data (x and y columns)
                    self.log.info(f"File appears to be pulsed measurement data with shape {data.shape}")
                    self.signal_data = data.T  # Transpose to match expected format
                    self.metadata = metadata
                    self.log.debug(f"Signal data shape after transpose: {self.signal_data.shape}")
                elif data.ndim == 2 and data.shape[0] > 1:
                    # Likely laser pulse data (multiple rows/pulses)
                    self.log.info(f"File appears to be laser pulse data with shape {data.shape}")
                    self.laser_data = data
                    self.metadata = metadata
                    self.log.debug(f"Laser data rows: {self.laser_data.shape[0]}, columns: {self.laser_data.shape[1]}")
                else:
                    # Likely raw data or generic data
                    self.log.info(f"File appears to be raw data with shape {data.shape}")
                    self.raw_data = data.squeeze()
                    self.metadata = metadata
                    self.log.debug(f"Raw data shape after squeeze: {self.raw_data.shape}")
            except Exception as e:
                self.log.error(f"Error loading the main file {file_path}: {str(e)}")
                import traceback
                self.log.debug(f"Traceback for main file loading error: {traceback.format_exc()}")
                # Continue to try loading related files
            
            # Now try to identify file type based on name and load related files
            basename = os.path.basename(file_path)
            dirname = os.path.dirname(file_path)
            
            # Check filenames for potential typos or encoding issues
            self.log.debug(f"Raw basename: {basename}")
            self.log.debug(f"Directory path: {dirname}")
            
            # Print the hexdump of the filename for debugging
            hex_dump = ' '.join([f"{ord(c):02x}" for c in basename])
            self.log.debug(f"Filename hex: {hex_dump}")
            
            # Check for spaces in filenames - this can cause issues
            if ' ' in basename:
                self.log.warning(f"Filename contains spaces which may cause issues: '{basename}'")
                # Note the positions of spaces for debugging
                space_positions = [i for i, char in enumerate(basename) if char == ' ']
                self.log.debug(f"Spaces found at positions: {space_positions}")
                
                # Try to normalize spaces around suffixes
                basename_normalized = basename
                for suffix in ["_pulsed_measurement", "_raw_timetrace", "_laser_pulses"]:
                    # Fix case with space before the suffix
                    if f" {suffix}" in basename_normalized:
                        basename_normalized = basename_normalized.replace(f" {suffix}", suffix)
                        self.log.debug(f"Normalized space before suffix: {suffix}")
                    
                    # Fix case with space after the suffix
                    if f"{suffix} " in basename_normalized:
                        basename_normalized = basename_normalized.replace(f"{suffix} ", suffix)
                        self.log.debug(f"Normalized space after suffix: {suffix}")
                
                if basename_normalized != basename:
                    self.log.info(f"Normalized filename for processing: '{basename_normalized}'")
                    basename = basename_normalized
            
            # Try to get base name without any of the special suffixes
            base_name = basename
            special_suffixes = ["_pulsed_measurement", "_raw_timetrace", "_laser_pulses"]
            found_suffixes = []
            for suffix in special_suffixes:
                if suffix in base_name:
                    found_suffixes.append(suffix)
                    base_name = base_name.replace(suffix, "")
            
            if found_suffixes:
                self.log.debug(f"Found suffixes in filename: {found_suffixes}")
            else:
                self.log.debug(f"No standard suffixes found in filename. Will try to detect related files anyway.")
                
                # Check for variations with spaces or special characters
                basename_lower = basename.lower()
                for suffix in special_suffixes:
                    suffix_no_underscore = suffix[1:]  # Remove the leading underscore
                    
                    # Check for suffix without underscore
                    if suffix_no_underscore in basename_lower:
                        self.log.warning(f"Found suffix without underscore: '{suffix_no_underscore}' instead of '{suffix}'")
                        # Try to fix the base name by removing this variation
                        idx = basename_lower.find(suffix_no_underscore)
                        if idx > 0:
                            potential_base = basename[:idx]
                            self.log.info(f"Extracted potential base name: '{potential_base}'")
                            base_name = potential_base
                            break
            
            # Special handling for filenames with unusual characters or encoding issues
            self.log.debug(f"Cleaned base name: {base_name}")
            
            # Check for special characters that might cause issues
            special_chars = []
            for i, char in enumerate(base_name):
                if not char.isalnum() and char not in "-_. ":
                    special_chars.append((i, char, ord(char)))
                    self.log.warning(f"Special character at position {i} in filename: '{char}' (code {ord(char)})")
                elif ord(char) > 127:
                    special_chars.append((i, char, ord(char)))
                    self.log.warning(f"Non-ASCII character at position {i} in filename: '{char}' (code {ord(char)})")
            
            if special_chars:
                self.log.debug(f"Special characters found: {special_chars}")
                
                # Try to create a clean base name
                clean_base = ''.join(c if (c.isalnum() or c in "-_. ") and ord(c) < 128 else '_' for c in base_name)
                if clean_base != base_name:
                    self.log.info(f"Created clean base name: '{clean_base}' (original: '{base_name}')")
                    alt_base_name = clean_base
                else:
                    alt_base_name = base_name
            else:
                alt_base_name = base_name
            
            # Create a list of potential base names to try
            potential_base_names = [base_name]
            if alt_base_name != base_name:
                potential_base_names.append(alt_base_name)
            
            # Strip trailing/leading spaces from base names
            potential_base_names = [bn.strip() for bn in potential_base_names]
            
            # Check if the base name contains other possible variations of the suffixes
            possible_suffix_variations = {
                "_pulsed_measurement": ["_pulsed_measurement", "_pulsedmeasurement", "_pulsed measurement", "pulsed_measurement", "_pulsed", "_measurement"],
                "_raw_timetrace": ["_raw_timetrace", "_rawtimetrace", "_raw timetrace", "raw_timetrace", "_raw", "_timetrace"],
                "_laser_pulses": ["_laser_pulses", "_laserpulses", "_laser pulses", "laser_pulses", "_laser"]
            }
            
            variation_found = False
            for standard, variations in possible_suffix_variations.items():
                for var in variations:
                    if var != standard and var in basename:
                        self.log.warning(f"Found possible suffix variation: '{var}' instead of '{standard}'")
                        variation_found = True
                        
                        # Try to create another potential base name by removing this variation
                        potential_base = basename.replace(var, "")
                        self.log.info(f"Created potential base name by removing variation: '{potential_base}'")
                        potential_base_names.append(potential_base.strip())
            
            # File extensions to try
            extensions = [file_extension]  # Start with the same extension as the loaded file
            if file_extension != '.dat':
                extensions.append('.dat')  # .dat is the most common
            if file_extension != '.csv':
                extensions.append('.csv')
            if file_extension != '.npy':
                extensions.append('.npy')
                
            # Log all the potential base names we'll try
            self.log.debug(f"Will try the following base names: {potential_base_names}")
            self.log.debug(f"Will try the following extensions: {extensions}")
            
            self.log.debug(f"Will try the following extensions: {extensions}")
            
            # Try to find related files with all possible extensions
            self.log.info(f"Looking for related files with multiple base name options")
            
            # Define a helper function to try loading a file with different storage types
            def try_load_file(file_path, file_type):
                if not os.path.isfile(file_path) or file_path == self.current_file_path:
                    return None, None
                
                self.log.info(f"Found matching file, attempting to load: {file_path}")
                
                # Try to determine the correct storage type based on extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.dat' and self.current_file_type != 'text':
                    temp_storage = TextDataStorage()
                    self.log.debug("Using TextDataStorage for this file")
                elif ext == '.csv' and self.current_file_type != 'csv':
                    temp_storage = CsvDataStorage()
                    self.log.debug("Using CsvDataStorage for this file")
                elif ext == '.npy' and self.current_file_type != 'npy':
                    temp_storage = NpyDataStorage()
                    self.log.debug("Using NpyDataStorage for this file")
                else:
                    temp_storage = storage
                
                try:
                    data, metadata = temp_storage.load_data(file_path)
                    self.log.debug(f"Successfully loaded {file_type} data with shape: {data.shape}")
                    return data, metadata
                except Exception as e:
                    self.log.error(f"Error loading {file_type} file {file_path}: {str(e)}")
                    import traceback
                    self.log.debug(f"Traceback for file loading error: {traceback.format_exc()}")
                    return None, None
            
            # Helper function to check for any matching files with the given pattern
            def find_files_matching_pattern(dirname, pattern):
                matching_files = []
                import glob
                # Use glob to find all matching files
                for matching_file in glob.glob(os.path.join(dirname, pattern)):
                    matching_files.append(matching_file)
                return matching_files
            
            # If we don't have pulsed measurement data yet, try to find it
            if self.signal_data is None:
                self.log.debug("No signal data loaded yet, looking for pulsed measurement files")
                signal_data_loaded = False
                
                # First try the standard format with our potential base names
                for base in potential_base_names:
                    if signal_data_loaded:
                        break
                        
                    for ext in extensions:
                        if signal_data_loaded:
                            break
                            
                        # Try standard naming format
                        pulsed_file_path = os.path.join(dirname, base + "_pulsed_measurement" + ext)
                        data, metadata = try_load_file(pulsed_file_path, "pulsed measurement")
                        
                        if data is not None:
                            self.signal_data = data.T
                            if not self.metadata:
                                self.metadata = metadata
                            else:
                                self.metadata.update(metadata)
                            self.log.info(f"Successfully loaded pulsed measurement data from {pulsed_file_path}")
                            signal_data_loaded = True
                            break
                
                # If still not found, try alternative naming formats
                if not signal_data_loaded:
                    # Try with variations in naming convention
                    for base in potential_base_names:
                        if signal_data_loaded:
                            break
                            
                        # Try variations of the suffix
                        for suffix_var in ["_pulsed_measurement", "_pulsedmeasurement", "_pulsed measurement", 
                                          "pulsed_measurement", " _pulsed_measurement", "_pulsed_measurement "]:
                            if signal_data_loaded:
                                break
                                
                            for ext in extensions:
                                pulsed_file_path = os.path.join(dirname, base + suffix_var + ext)
                                data, metadata = try_load_file(pulsed_file_path, "pulsed measurement")
                                
                                if data is not None:
                                    self.signal_data = data.T
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded pulsed measurement data from {pulsed_file_path}")
                                    signal_data_loaded = True
                                    break
                
                # If still not found, try using glob to find any files that might match
                if not signal_data_loaded:
                    self.log.debug("Trying to find pulsed measurement files using pattern matching")
                    for base in potential_base_names:
                        if signal_data_loaded:
                            break
                            
                        # Look for any file containing both the base name and pulsed_measurement
                        matching_files = find_files_matching_pattern(dirname, f"*{base}*pulsed*measurement*.*")
                        matching_files.extend(find_files_matching_pattern(dirname, f"*pulsed*measurement*{base}*.*"))
                        
                        for match_file in matching_files:
                            if match_file != file_path:  # Don't try to load the current file again
                                data, metadata = try_load_file(match_file, "pulsed measurement")
                                
                                if data is not None:
                                    self.signal_data = data.T
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded pulsed measurement data from {match_file}")
                                    signal_data_loaded = True
                                    break
                
                if not signal_data_loaded:
                    self.log.warning("Failed to find or load any pulsed measurement data")
            
            # If we don't have raw data yet, try to find it
            if self.raw_data is None:
                self.log.debug("No raw data loaded yet, looking for raw timetrace files")
                raw_data_loaded = False
                
                # First try the standard format with our potential base names
                for base in potential_base_names:
                    if raw_data_loaded:
                        break
                        
                    for ext in extensions:
                        if raw_data_loaded:
                            break
                            
                        # Try standard naming format
                        raw_file_path = os.path.join(dirname, base + "_raw_timetrace" + ext)
                        data, metadata = try_load_file(raw_file_path, "raw timetrace")
                        
                        if data is not None:
                            self.raw_data = data.squeeze()
                            if not self.metadata:
                                self.metadata = metadata
                            else:
                                self.metadata.update(metadata)
                            self.log.info(f"Successfully loaded raw data from {raw_file_path}")
                            raw_data_loaded = True
                            break
                
                # If still not found, try alternative naming formats
                if not raw_data_loaded:
                    # Try with variations in naming convention
                    for base in potential_base_names:
                        if raw_data_loaded:
                            break
                            
                        # Try variations of the suffix
                        for suffix_var in ["_raw_timetrace", "_rawtimetrace", "_raw timetrace", 
                                          "raw_timetrace", " _raw_timetrace", "_raw_timetrace "]:
                            if raw_data_loaded:
                                break
                                
                            for ext in extensions:
                                raw_file_path = os.path.join(dirname, base + suffix_var + ext)
                                data, metadata = try_load_file(raw_file_path, "raw timetrace")
                                
                                if data is not None:
                                    self.raw_data = data.squeeze()
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded raw data from {raw_file_path}")
                                    raw_data_loaded = True
                                    break
                
                # If still not found, try using glob to find any files that might match
                if not raw_data_loaded:
                    self.log.debug("Trying to find raw timetrace files using pattern matching")
                    for base in potential_base_names:
                        if raw_data_loaded:
                            break
                            
                        # Look for any file containing both the base name and raw_timetrace
                        matching_files = find_files_matching_pattern(dirname, f"*{base}*raw*timetrace*.*")
                        matching_files.extend(find_files_matching_pattern(dirname, f"*raw*timetrace*{base}*.*"))
                        
                        for match_file in matching_files:
                            if match_file != file_path:  # Don't try to load the current file again
                                data, metadata = try_load_file(match_file, "raw timetrace")
                                
                                if data is not None:
                                    self.raw_data = data.squeeze()
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded raw data from {match_file}")
                                    raw_data_loaded = True
                                    break
                
                if not raw_data_loaded:
                    self.log.warning("Failed to find or load any raw timetrace data")
            
            # If we don't have laser data yet, try to find it
            if self.laser_data is None:
                self.log.debug("No laser data loaded yet, looking for laser pulses files")
                laser_data_loaded = False
                
                # First try the standard format with our potential base names
                for base in potential_base_names:
                    if laser_data_loaded:
                        break
                        
                    for ext in extensions:
                        if laser_data_loaded:
                            break
                            
                        # Try standard naming format
                        laser_file_path = os.path.join(dirname, base + "_laser_pulses" + ext)
                        data, metadata = try_load_file(laser_file_path, "laser pulses")
                        
                        if data is not None:
                            self.laser_data = data
                            if not self.metadata:
                                self.metadata = metadata
                            else:
                                self.metadata.update(metadata)
                            self.log.info(f"Successfully loaded laser data from {laser_file_path}")
                            laser_data_loaded = True
                            break
                
                # If still not found, try alternative naming formats
                if not laser_data_loaded:
                    # Try with variations in naming convention
                    for base in potential_base_names:
                        if laser_data_loaded:
                            break
                            
                        # Try variations of the suffix
                        for suffix_var in ["_laser_pulses", "_laserpulses", "_laser pulses", 
                                          "laser_pulses", " _laser_pulses", "_laser_pulses "]:
                            if laser_data_loaded:
                                break
                                
                            for ext in extensions:
                                laser_file_path = os.path.join(dirname, base + suffix_var + ext)
                                data, metadata = try_load_file(laser_file_path, "laser pulses")
                                
                                if data is not None:
                                    self.laser_data = data
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded laser data from {laser_file_path}")
                                    laser_data_loaded = True
                                    break
                
                # If still not found, try using glob to find any files that might match
                if not laser_data_loaded:
                    self.log.debug("Trying to find laser pulses files using pattern matching")
                    for base in potential_base_names:
                        if laser_data_loaded:
                            break
                            
                        # Look for any file containing both the base name and laser_pulses
                        matching_files = find_files_matching_pattern(dirname, f"*{base}*laser*pulses*.*")
                        matching_files.extend(find_files_matching_pattern(dirname, f"*laser*pulses*{base}*.*"))
                        
                        for match_file in matching_files:
                            if match_file != file_path:  # Don't try to load the current file again
                                data, metadata = try_load_file(match_file, "laser pulses")
                                
                                if data is not None:
                                    self.laser_data = data
                                    if not self.metadata:
                                        self.metadata = metadata
                                    else:
                                        self.metadata.update(metadata)
                                    self.log.info(f"Successfully loaded laser data from {match_file}")
                                    laser_data_loaded = True
                                    break
                
                if not laser_data_loaded:
                    self.log.warning("Failed to find or load any laser pulses data")
            
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
            
            # Log detailed status information for diagnosing issues
            if not initial_load_successful and not any([data_info['has_raw_data'], 
                                                       data_info['has_laser_data'], 
                                                       data_info['has_signal_data']]):
                self.log.error(f"Failed to load any data from {file_path} or related files!")
                # Add specific suggestions based on common patterns in error messages
                if "_pulsed_measurement" in file_path and not os.path.isfile(file_path.replace("_pulsed_measurement", "_laser_pulses")):
                    self.log.error("Related laser_pulses file does not exist.")
                if "_raw_timetrace" in file_path and not os.path.isfile(file_path.replace("_raw_timetrace", "_laser_pulses")):
                    self.log.error("Related laser_pulses file does not exist.")
            elif data_info['has_signal_data'] and not data_info['has_laser_data']:
                self.log.warning("Signal data loaded, but no laser data available. NV state analysis might still work.")
            elif data_info['has_laser_data'] and not data_info['has_raw_data']:
                self.log.warning("Laser data loaded, but no raw data available. This is normal if only analyzing pre-extracted laser pulses.")
            
            self.sigDataLoaded.emit(data_info)
            return self.metadata
            
        except Exception as e:
            self.log.error(f"Error during data loading process for {file_path}: {str(e)}")
            import traceback
            self.log.error(f"Detailed error traceback: {traceback.format_exc()}")
            
            # Try to provide more helpful error messages
            if "encoding" in str(e).lower() or "decode" in str(e).lower():
                self.log.error("This appears to be a file encoding issue. Check for unusual characters in the filename.")
            if "permission" in str(e).lower():
                self.log.error("This appears to be a file permission issue. Check if the file is accessible.")
            if "disk" in str(e).lower() or "space" in str(e).lower():
                self.log.error("This might be a disk space or I/O issue.")
                
            return
    
    def extract_laser_pulses(self):
        """
        Extract laser pulses from raw data if available
        
        @return dict: result from pulse extraction
        """
        if self.raw_data is None:
            self.log.error("No raw data available for pulse extraction")
            self.log.debug("Raw data is None. This can happen if the raw_timetrace file wasn't found or couldn't be loaded.")
            self.log.debug("Make sure the raw_timetrace file exists and has the correct format. Check for typos in the filename.")
            return None
        
        try:
            self.log.info(f"Starting laser pulse extraction from raw data with shape {self.raw_data.shape}")
            self.log.debug(f"Raw data statistics: min={np.min(self.raw_data)}, max={np.max(self.raw_data)}, mean={np.mean(self.raw_data)}")
            self.log.debug(f"Extraction settings: {self._pulseextractor.extraction_settings}")
            
            # Time the extraction process
            start_time = datetime.datetime.now()
            self.log.debug(f"Starting pulse extraction at {start_time}")
            
            return_dict = self._pulseextractor.extract_laser_pulses(self.raw_data)
            
            end_time = datetime.datetime.now()
            extraction_duration = (end_time - start_time).total_seconds()
            self.log.debug(f"Pulse extraction took {extraction_duration:.2f} seconds")
            
            if 'laser_counts_arr' not in return_dict or return_dict['laser_counts_arr'] is None:
                self.log.error("Pulse extractor did not return valid laser_counts_arr")
                self.log.debug(f"Return dictionary keys: {list(return_dict.keys())}")
                return None
            
            self.laser_data = return_dict['laser_counts_arr']
            
            # Log information about the extracted pulses
            num_pulses = self.laser_data.shape[0] if self.laser_data is not None else 0
            self.log.info(f"Successfully extracted {num_pulses} laser pulses")
            
            if num_pulses > 0:
                pulse_length = self.laser_data.shape[1]
                self.log.debug(f"Pulse length: {pulse_length} bins")
                self.log.debug(f"Laser data min counts: {np.min(self.laser_data)}, max: {np.max(self.laser_data)}")
                
                # Calculate average pulse shape for visualization/debugging
                avg_pulse = np.mean(self.laser_data, axis=0)
                binwidth = self.fast_counter_settings['bin_width']
                self.log.debug(f"Average pulse shape (first 5 bins): {avg_pulse[:5]}")
                
                # Check for potential issues with the extracted pulses
                zero_pulses = np.sum(np.all(self.laser_data == 0, axis=1))
                if zero_pulses > 0:
                    self.log.warning(f"Detected {zero_pulses} pulses with all zero counts")
                
                very_low_pulses = np.sum(np.max(self.laser_data, axis=1) < 5)
                if very_low_pulses > 0:
                    self.log.warning(f"Detected {very_low_pulses} pulses with very low counts (max < 5)")
                
                # Log some additional info about the return dictionary
                for key in return_dict:
                    if key != 'laser_counts_arr':
                        value = return_dict[key]
                        if isinstance(value, np.ndarray):
                            self.log.debug(f"Returned '{key}' with shape {value.shape}")
                        else:
                            self.log.debug(f"Returned '{key}': {value}")
            else:
                self.log.error("No laser pulses were extracted from the raw data")
                self.log.debug("This could indicate an issue with the extraction settings or raw data format")
            
            return return_dict
            
        except Exception as e:
            self.log.error(f"Error extracting laser pulses: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            
            # Try to provide more specific error messages
            error_msg = str(e).lower()
            if "shape" in error_msg or "dimension" in error_msg:
                self.log.error("This appears to be a data shape/dimension mismatch. Check if the raw data format is correct.")
            elif "memory" in error_msg:
                self.log.error("This appears to be a memory error. The raw data might be too large to process.")
            elif "attribute" in error_msg:
                self.log.error("This appears to be a missing attribute error. Check the pulse extractor settings.")
            elif "zero" in error_msg or "divide" in error_msg:
                self.log.error("This appears to be a division by zero error, possibly due to invalid extraction parameters.")
                
            return None
    
    def analyze_laser_pulses(self):
        """
        Analyze extracted laser pulses
        
        @return tuple: (signal_data, error_data)
        """
        # Check if we have laser data to analyze
        if self.laser_data is None:
            self.log.error("No laser data available for analysis")
            self.log.debug("Laser data is None. This can happen if the laser_pulses file wasn't found or couldn't be loaded.")
            
            # If signal data is already loaded (e.g., from a _pulsed_measurement file),
            # we can still perform NV state analysis on that
            if self.signal_data is not None:
                self.log.info("Using already loaded signal data for NV state analysis")
                self.log.debug(f"Signal data shape: {self.signal_data.shape}")
                
                try:
                    self.analyze_nv_states()
                    self.log.info("NV state analysis completed on existing signal data")
                except Exception as e:
                    self.log.error(f"Failed to analyze NV states on existing signal data: {str(e)}")
                    import traceback
                    self.log.error(f"Traceback: {traceback.format_exc()}")
                
                # Emit results with what we have
                self.sigAnalysisComplete.emit({
                    'signal_data': self.signal_data,
                    'analysis_results': {},
                    'nv_state_data': self.nv_state_data,
                    'state_statistics': self.state_statistics
                })
                return None, None
            else:
                self.log.error("No signal data available either. Cannot perform NV state analysis.")
                self.log.debug("Both laser_data and signal_data are None. Try loading a different file or check file integrity.")
                return None, None
        
        try:
            self.log.info(f"Analyzing laser data with shape {self.laser_data.shape}")
            self.log.debug(f"Laser data type: {self.laser_data.dtype}, min: {np.min(self.laser_data)}, max: {np.max(self.laser_data)}")
            self.log.debug(f"Analysis settings: {self._pulseanalyzer.analysis_settings}")
            
            # Call the pulse analyzer
            start_time = datetime.datetime.now()
            self.log.debug(f"Starting laser pulse analysis at {start_time}")
            signal, error = self._pulseanalyzer.analyse_laser_pulses(self.laser_data)
            end_time = datetime.datetime.now()
            analysis_duration = (end_time - start_time).total_seconds()
            self.log.debug(f"Pulse analysis took {analysis_duration:.2f} seconds")
            
            if signal is None or len(signal) == 0:
                self.log.error("Pulse analyzer returned empty signal data")
                self.log.debug("This could indicate an issue with the laser pulse data or analysis settings")
                return None, None
                
            self.log.info(f"Laser pulse analysis complete, got signal with length {len(signal)}")
            self.log.debug(f"Signal min: {np.min(signal)}, max: {np.max(signal)}, mean: {np.mean(signal)}")
            
            # Create signal data array if not already loaded
            if self.signal_data is None:
                self.log.info("Creating new signal data array from analyzed laser pulses")
                
                # Check for controlled variable in metadata (with different possible keys)
                x_data = None
                controlled_var_keys = ['Controlled variable', 'controlled_variable', 'x_data', 'x-data', 'x axis']
                found_key = None
                
                for key in controlled_var_keys:
                    if key in self.metadata:
                        x_data = np.array(self.metadata[key])
                        found_key = key
                        self.log.info(f"Using '{key}' from metadata with length {len(x_data)}")
                        break
                
                if x_data is None:
                    # Create default x data
                    x_data = np.arange(len(signal))
                    self.log.info(f"No controlled variable found in metadata, creating default x data with length {len(x_data)}")
                    self.log.debug(f"Metadata keys: {list(self.metadata.keys())}")
                else:
                    self.log.debug(f"X data min: {np.min(x_data)}, max: {np.max(x_data)}, type: {x_data.dtype}")
                
                # Determine signal dimension based on alternating flag
                alternating_keys = ['alternating', 'is_alternating', 'alternate']
                alternating = False
                for key in alternating_keys:
                    if key in self.metadata:
                        alternating = bool(self.metadata[key])
                        self.log.debug(f"Found alternating flag in metadata key '{key}': {alternating}")
                        break
                        
                signal_dim = 3 if alternating else 2
                self.log.debug(f"Signal dimension: {signal_dim} (alternating: {alternating})")
                
                # Check if dimensions match and adjust if needed
                if alternating:
                    expected_signal_len = len(x_data) * 2
                    self.log.debug(f"Alternating data: expected signal length = {expected_signal_len}, actual = {len(signal)}")
                    
                    if expected_signal_len != len(signal):
                        self.log.warning(f"Mismatch in data dimensions: x_data length ({len(x_data)}) * 2 != signal length ({len(signal)})")
                        
                        # First check if the actual signal length is even (required for alternating data)
                        if len(signal) % 2 != 0:
                            self.log.warning(f"Odd number of data points ({len(signal)}) for alternating data, trimming last point")
                            signal = signal[:-1]
                            error = error[:-1]
                            self.log.debug(f"After trimming: signal length = {len(signal)}")
                        
                        # Now adjust x_data to match signal
                        self.log.info("Adjusting x_data to match signal data")
                        x_data = np.arange(len(signal) // 2)
                        self.log.debug(f"Created new x_data with length {len(x_data)}")
                else:
                    expected_signal_len = len(x_data)
                    self.log.debug(f"Non-alternating data: expected signal length = {expected_signal_len}, actual = {len(signal)}")
                    
                    if expected_signal_len != len(signal):
                        self.log.warning(f"Mismatch in data dimensions: x_data length ({len(x_data)}) != signal length ({len(signal)})")
                        self.log.info("Adjusting x_data to match signal data")
                        x_data = np.arange(len(signal))
                        self.log.debug(f"Created new x_data with length {len(x_data)}")
                
                # Create signal data array
                self.log.debug(f"Creating signal_data array with shape ({signal_dim}, {len(x_data)})")
                self.signal_data = np.zeros((signal_dim, len(x_data)), dtype=float)
                self.signal_data[0] = x_data
                
                # Populate signal data
                if alternating:
                    self.log.info(f"Processing alternating data, signal length: {len(signal)}")
                    
                    # Exclude laser pulses to ignore if present
                    laser_ignore_keys = ['Laser ignore indices', 'laser_ignore_list', 'ignore_indices']
                    laser_ignore_list = []
                    
                    for key in laser_ignore_keys:
                        if key in self.metadata and self.metadata[key]:
                            laser_ignore_list = self.metadata[key]
                            self.log.debug(f"Found laser ignore list in metadata key '{key}': {laser_ignore_list}")
                            break
                    
                    if len(laser_ignore_list) > 0:
                        self.log.info(f"Excluding {len(laser_ignore_list)} laser pulses from analysis")
                        self.log.debug(f"Ignore indices: {laser_ignore_list}")
                        
                        try:
                            signal = np.delete(signal, laser_ignore_list)
                            error = np.delete(error, laser_ignore_list)
                            self.log.debug(f"After ignoring pulses: signal length = {len(signal)}")
                        except Exception as e:
                            self.log.error(f"Error excluding laser pulses: {str(e)}")
                            self.log.debug(f"Will continue with unfiltered data")
                    
                    # Make sure the signal length is even for alternating data
                    if len(signal) % 2 != 0:
                        self.log.warning(f"Odd number of signal points ({len(signal)}) for alternating data, trimming last point")
                        signal = signal[:-1]
                        error = error[:-1]
                        self.log.debug(f"After trimming: signal length = {len(signal)}")
                    
                    # Split alternating data
                    try:
                        self.signal_data[1] = signal[::2]
                        self.signal_data[2] = signal[1::2]
                        self.log.info(f"Alternating data processed, signal_data shape: {self.signal_data.shape}")
                        self.log.debug(f"Signal 1 min: {np.min(self.signal_data[1])}, max: {np.max(self.signal_data[1])}")
                        self.log.debug(f"Signal 2 min: {np.min(self.signal_data[2])}, max: {np.max(self.signal_data[2])}")
                    except Exception as e:
                        self.log.error(f"Error splitting alternating data: {str(e)}")
                        import traceback
                        self.log.error(f"Traceback: {traceback.format_exc()}")
                        # Try to recover by using all data as non-alternating
                        self.log.warning("Attempting to recover by treating data as non-alternating")
                        self.signal_data = np.zeros((2, len(signal)), dtype=float)
                        self.signal_data[0] = np.arange(len(signal))
                        self.signal_data[1] = signal
                else:
                    self.log.info(f"Processing non-alternating data, signal length: {len(signal)}")
                    # Make sure lengths match
                    if len(signal) != len(x_data):
                        self.log.warning(f"Signal length ({len(signal)}) doesn't match x_data length ({len(x_data)}), trimming to shorter")
                        min_len = min(len(signal), len(x_data))
                        self.log.debug(f"Trimming to length {min_len}")
                        self.signal_data[0] = x_data[:min_len]
                        self.signal_data[1] = signal[:min_len]
                    else:
                        self.signal_data[1] = signal
                    
                    self.log.info(f"Non-alternating data processed, signal_data shape: {self.signal_data.shape}")
                    self.log.debug(f"Signal min: {np.min(self.signal_data[1])}, max: {np.max(self.signal_data[1])}")
            else:
                self.log.info(f"Using existing signal_data with shape {self.signal_data.shape}")
                self.log.debug(f"Existing signal_data[0] min: {np.min(self.signal_data[0])}, max: {np.max(self.signal_data[0])}")
                self.log.debug(f"Existing signal_data[1] min: {np.min(self.signal_data[1])}, max: {np.max(self.signal_data[1])}")
                if self.signal_data.shape[0] > 2:
                    self.log.debug(f"Existing signal_data[2] min: {np.min(self.signal_data[2])}, max: {np.max(self.signal_data[2])}")
            
            self.analysis_results = {
                'signal': signal,
                'error': error
            }
            
            # Perform NV state analysis
            try:
                self.log.info("Starting NV state analysis")
                self.analyze_nv_states()
                self.log.info("NV state analysis completed successfully")
            except Exception as e:
                self.log.error(f"Error during NV state analysis: {str(e)}")
                import traceback
                self.log.error(f"Traceback: {traceback.format_exc()}")
            
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
            
            # Try to provide more specific error messages
            error_msg = str(e).lower()
            if "shape" in error_msg or "dimension" in error_msg:
                self.log.error("This appears to be a data shape/dimension mismatch. Check if the laser data format is correct.")
            elif "memory" in error_msg:
                self.log.error("This appears to be a memory error. The data might be too large to process.")
            elif "attribute" in error_msg:
                self.log.error("This appears to be a missing attribute error. Check the pulse analyzer settings.")
            
            return None, None
    
    def analyze_nv_states(self):
        """
        Analyze NV states based on count data using threshold
        
        @return ndarray: Array with state assignments (0: ms=0, 1: ms=-1/+1)
        """
        if self.signal_data is None or self.signal_data.shape[0] < 2:
            self.log.error("No signal data available for NV state analysis")
            self.log.debug(f"Signal data is {'None' if self.signal_data is None else f'available but has shape {self.signal_data.shape}'}")
            return None
        
        try:
            self.log.info(f"Analyzing NV states from signal data with shape {self.signal_data.shape}")
            
            # Validate signal data
            if np.all(np.isnan(self.signal_data[1])):
                self.log.error("All signal data values are NaN, cannot perform state analysis")
                return None
            
            if np.all(self.signal_data[1] == 0):
                self.log.error("All signal data values are zero, cannot perform meaningful state analysis")
                return None
                
            self.log.debug(f"Signal data statistics: min={np.min(self.signal_data[1])}, max={np.max(self.signal_data[1])}, mean={np.mean(self.signal_data[1])}")
            
            # Get reference level
            reference = self._nv_reference_level
            if reference is None:
                # If no reference set, use the mean as reference
                reference = np.mean(self.signal_data[1])
                self._nv_reference_level = reference
                self.log.info(f"No reference level set, using mean value: {reference}")
                self.log.debug(f"Calculated reference from all {len(self.signal_data[1])} data points")
            else:
                self.log.info(f"Using existing reference level: {reference}")
                
            # Validate reference level
            if reference <= 0:
                self.log.warning(f"Reference level is not positive ({reference}), this may cause issues with thresholding")
                # Try to recover by using mean if it's positive
                mean_value = np.mean(self.signal_data[1])
                if mean_value > 0:
                    reference = mean_value
                    self._nv_reference_level = reference
                    self.log.info(f"Adjusted reference level to mean value: {reference}")
                else:
                    self.log.error("Could not determine a valid reference level, NV state analysis may be inaccurate")
            
            # Determine threshold value
            threshold = reference * self._nv_threshold
            self.log.info(f"Using threshold value: {threshold} (reference: {reference} * threshold factor: {self._nv_threshold})")
            
            # Verify threshold is reasonable
            if threshold <= 0:
                self.log.error(f"Calculated threshold is not positive ({threshold}), cannot perform reliable state assignment")
                return None
                
            min_signal = np.min(self.signal_data[1])
            max_signal = np.max(self.signal_data[1])
            
            if threshold <= min_signal:
                self.log.warning(f"Threshold ({threshold}) is less than or equal to minimum signal value ({min_signal})")
                self.log.warning("All data points will be classified as ms=0 state, consider adjusting threshold")
            elif threshold >= max_signal:
                self.log.warning(f"Threshold ({threshold}) is greater than or equal to maximum signal value ({max_signal})")
                self.log.warning("All data points will be classified as ms=-1/+1 state, consider adjusting threshold")
            
            # Assign states based on threshold
            # If counts > threshold: ms=0 state (state=0)
            # If counts <= threshold: ms=-1 or ms=+1 state (state=1)
            self.log.debug("Assigning states based on threshold")
            states = np.zeros_like(self.signal_data[1])
            states[self.signal_data[1] <= threshold] = 1
            
            # Log detailed info about the first few points for debugging
            sample_size = min(10, len(states))
            for i in range(sample_size):
                self.log.debug(f"Point {i}: signal={self.signal_data[1][i]}, threshold={threshold}, assigned state={states[i]}")
            
            # Log how many points are assigned to each state
            ms0_count = np.sum(states == 0)
            ms1_count = np.sum(states == 1)
            self.log.info(f"Assigned states: ms=0: {ms0_count}, ms=-1/+1: {ms1_count}")
            
            # Warn if state assignment is severely imbalanced
            if ms0_count == 0 or ms1_count == 0:
                self.log.warning("All data points were assigned to a single state!")
                self.log.warning("This suggests an issue with the threshold setting or data quality")
            elif ms0_count > 0 and ms1_count > 0:
                ratio = max(ms0_count, ms1_count) / min(ms0_count, ms1_count)
                if ratio > 10:
                    self.log.warning(f"State assignment is highly imbalanced (ratio {ratio:.1f}:1)")
                    self.log.warning("Consider adjusting the threshold or checking data quality")
            
            self.nv_state_data = states
            
            # Calculate histogram and statistics
            self.log.debug("Calculating state histogram and statistics")
            unique, counts = np.unique(states, return_counts=True)
            self.state_histogram = (unique, counts)
            
            total_counts = len(states)
            
            # Check for possible state transitions by calculating state changes
            if total_counts > 1:
                state_changes = np.sum(np.abs(np.diff(states)))
                change_percentage = (state_changes / (total_counts - 1)) * 100
                self.log.info(f"Detected {state_changes} state transitions ({change_percentage:.1f}% of points)")
                
                # Look for potential data issues based on transition patterns
                if change_percentage > 50:
                    self.log.warning("High number of state transitions detected (>50% of points)")
                    self.log.warning("This could indicate noise in the data or threshold set too close to the signal level")
                elif change_percentage < 1 and ms0_count > 0 and ms1_count > 0:
                    self.log.warning("Very few state transitions detected (<1% of points)")
                    self.log.warning("This could indicate distinct populations that should be analyzed separately")
            
            # Compile detailed statistics
            self.state_statistics = {
                'total_counts': total_counts,
                'ms0_count': ms0_count,
                'ms1_count': ms1_count,
                'ms0_percentage': (ms0_count / total_counts) * 100 if total_counts > 0 else 0,
                'ms1_percentage': (ms1_count / total_counts) * 100 if total_counts > 0 else 0,
                'threshold': threshold,
                'reference_level': reference,
                'threshold_factor': self._nv_threshold,
                'signal_min': min_signal,
                'signal_max': max_signal,
                'signal_mean': np.mean(self.signal_data[1]),
                'signal_std': np.std(self.signal_data[1])
            }
            
            # Calculate contrast if both states are present
            if ms0_count > 0 and ms1_count > 0:
                ms0_indices = np.where(states == 0)[0]
                ms1_indices = np.where(states == 1)[0]
                
                ms0_mean = np.mean(self.signal_data[1][ms0_indices])
                ms1_mean = np.mean(self.signal_data[1][ms1_indices])
                
                contrast = ((ms0_mean - ms1_mean) / (ms0_mean + ms1_mean)) * 100 if (ms0_mean + ms1_mean) > 0 else 0
                self.state_statistics['ms0_mean'] = ms0_mean
                self.state_statistics['ms1_mean'] = ms1_mean
                self.state_statistics['contrast'] = contrast
                
                self.log.info(f"State contrast: {contrast:.2f}% (ms0 mean: {ms0_mean:.2f}, ms1 mean: {ms1_mean:.2f})")
            
            self.log.info(f"NV state statistics: ms0={self.state_statistics['ms0_percentage']:.1f}%, ms1={self.state_statistics['ms1_percentage']:.1f}%")
            self.log.debug(f"Detailed statistics: {self.state_statistics}")
            
            # Emit histogram and statistics
            self.sigNvStateHistogramUpdated.emit(self.state_histogram, self.state_statistics)
            
            return states
            
        except Exception as e:
            self.log.error(f"Error analyzing NV states: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")
            
            # Try to provide more helpful error messages
            error_msg = str(e).lower()
            if "shape" in error_msg or "dimension" in error_msg:
                self.log.error("This appears to be a data shape/dimension mismatch in the signal data.")
            elif "nan" in error_msg or "infinity" in error_msg:
                self.log.error("This appears to be an issue with NaN or infinite values in the data.")
            elif "memory" in error_msg:
                self.log.error("This appears to be a memory error. The data might be too large to process.")
            elif "zero" in error_msg or "divide" in error_msg:
                self.log.error("This appears to be a division by zero error, possibly due to signal values being zero.")
            
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