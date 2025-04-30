#!/usr/bin/env python3
"""
Validator for optical amplitude measurement data files.
Checks if a directory contains all required files in the correct format.
"""

import os
import sys
import json
import configparser
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.simpledialog import askinteger


def validate_signal_data(file_path):
    """Validate signal data file (.dat) for Qudi laser pulse data"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:  # At least some data should be present
            return False, "Signal data file has insufficient data"
        
        # Check for Qudi data format - files start with metadata comments
        # Typically starting with # [General] or similar
        qudi_format = False
        metadata_section = False
        data_section = False
        data_start_line = 0
        
        # First scan through the file to identify metadata and data sections
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect Qudi metadata format
            if line.startswith('# [General]'):
                qudi_format = True
                metadata_section = True
                continue
                
            # Detect when metadata section ends and data begins
            if qudi_format and not line.startswith('#') and line:
                data_section = True
                data_start_line = i
                break
        
        if not qudi_format:
            # Not a Qudi format file
            return False, "File does not appear to be in Qudi format (missing metadata headers)"
            
        if not data_section:
            return False, "No data section found after metadata"
            
        # Check data lines - they should be tab or space separated numeric values
        valid_data_lines = 0
        for i, line in enumerate(lines[data_start_line:], start=data_start_line):
            if not line.strip():  # Skip empty lines
                continue
                
            values = line.strip().split()
            
            # Must have at least one value to be valid
            if not values:
                continue
                
            # Try to convert all values to float
            try:
                for val in values:
                    float(val)
                valid_data_lines += 1
            except ValueError:
                # We'll be permissive here - some lines might not be data
                continue
        
        if valid_data_lines < 1:
            return False, "No valid data lines found in file"
        
        # If we got here, the file matches Qudi format and has valid data
        return True, f"Valid Qudi data file with {valid_data_lines} data lines"
    
    except Exception as e:
        return False, f"Error validating signal data: {str(e)}"


def validate_config_file(file_path):
    """Validate configuration file (.cfg)"""
    try:
        config = configparser.ConfigParser()
        config.read(file_path)
        
        # Check if file has at least one section
        if len(config.sections()) == 0:
            return False, "Config file has no sections"
        
        # Check for required sections (customize as needed)
        required_sections = ['hardware', 'measurement']
        missing_sections = [s for s in required_sections if s not in config.sections()]
        
        if missing_sections:
            return False, f"Config file missing sections: {', '.join(missing_sections)}"
            
        return True, "Config file is valid"
    
    except Exception as e:
        return False, f"Error validating config file: {str(e)}"


def validate_metadata(file_path):
    """Validate metadata file (.meta)"""
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        
        # Check for required top-level keys
        required_keys = ['experiment', 'sample', 'conditions']
        missing_keys = [k for k in required_keys if k not in metadata]
        
        if missing_keys:
            return False, f"Metadata missing required sections: {', '.join(missing_keys)}"
            
        # Check experiment section
        if 'experiment' in metadata:
            exp_required = ['name', 'date']
            missing_exp = [k for k in exp_required if k not in metadata['experiment']]
            if missing_exp:
                return False, f"Experiment section missing fields: {', '.join(missing_exp)}"
        
        return True, "Metadata file is valid"
    
    except json.JSONDecodeError:
        return False, "Metadata file is not valid JSON"
    except Exception as e:
        return False, f"Error validating metadata: {str(e)}"


def find_files_by_extension(directory, extension):
    """Find files with the specified extension in the directory"""
    files = []
    for file in os.listdir(directory):
        if file.lower().endswith(extension.lower()):
            files.append(os.path.join(directory, file))
    return files


def validate_directory(directory):
    """Validate if directory contains all required files in correct format"""
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return False
    
    # Find files by extension
    dat_files = find_files_by_extension(directory, '.dat')
    cfg_files = find_files_by_extension(directory, '.cfg')
    meta_files = find_files_by_extension(directory, '.meta')
    
    # Check if all required files exist
    print(f"\nChecking directory: {directory}")
    print("-" * 50)
    
    if not dat_files:
        print("❌ No signal data files (.dat) found")
        success = False
    else:
        print(f"✅ Found {len(dat_files)} signal data file(s)")
        
    if not cfg_files:
        print("❌ No configuration files (.cfg) found")
        success = False
    else:
        print(f"✅ Found {len(cfg_files)} configuration file(s)")
        
    if not meta_files:
        print("❌ No metadata files (.meta) found")
        success = False
    else:
        print(f"✅ Found {len(meta_files)} metadata file(s)")
    
    # If any file type is missing, return
    if not (dat_files and cfg_files and meta_files):
        return False
    
    # Validate file contents
    print("\nValidating file contents:")
    print("-" * 50)
    
    # Validate the first file of each type
    dat_valid, dat_msg = validate_signal_data(dat_files[0])
    print(f"{'✅' if dat_valid else '❌'} Signal data: {dat_msg}")
    
    cfg_valid, cfg_msg = validate_config_file(cfg_files[0])
    print(f"{'✅' if cfg_valid else '❌'} Config file: {cfg_msg}")
    
    meta_valid, meta_msg = validate_metadata(meta_files[0])
    print(f"{'✅' if meta_valid else '❌'} Metadata: {meta_msg}")
    
    # Overall validation result
    overall_valid = dat_valid and cfg_valid and meta_valid
    
    print("\nOverall validation result:")
    print("-" * 50)
    if overall_valid:
        print("✅ Directory contains all required files in correct format")
    else:
        print("❌ Directory validation failed")
    
    return overall_valid


def load_qudi_data(file_path):
    """Load data from a Qudi data file (.dat)"""
    try:
        # Read all lines from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find where the data section begins
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "# ---- END HEADER ----":
                data_start = i + 1
                break
            
            # Alternative detection method if END HEADER line is not present
            if not line.startswith('#') and line.strip() and data_start == 0:
                data_start = i
        
        # Extract metadata from header
        metadata = {}
        for i in range(data_start):
            line = lines[i].strip()
            if line.startswith('# ') and '=' in line:
                parts = line[2:].split('=', 1)
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()
        
        # Parse data section
        data = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
                
            # Parse tab or space-separated values
            values = line.split('\t') if '\t' in line else line.split()
            if len(values) >= 1:
                try:
                    # Convert all values to float
                    float_values = [float(v) for v in values]
                    data.append(float_values)
                except ValueError:
                    continue  # Skip non-numeric lines
        
        if not data:
            return None, metadata
            
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)
        
        return data_array, metadata
        
    except Exception as e:
        print(f"Error loading Qudi data file: {str(e)}")
        return None, {}

def plot_full_pulsed_data(file_path):
    """Create a plot similar to the pulsed measurement module showing tau vs signal"""
    data_array, metadata = load_qudi_data(file_path)
    
    if data_array is None or data_array.shape[0] < 2:
        messagebox.showerror("Error", "No valid data found in file")
        return
    
    # Check if this is a pulsed measurement file (should have 2 columns: tau and signal)
    if data_array.shape[1] < 2:
        messagebox.showerror("Error", "File does not appear to be a pulsed measurement file (expected at least 2 columns)")
        return
    
    # Create a new Tkinter window
    window = tk.Toplevel()
    window.title(f"Pulsed Measurement Plot - {os.path.basename(file_path)}")
    window.geometry("800x600")
    
    # Create figure and plot
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot the data (tau vs signal)
    x_data = data_array[:, 0]  # Tau values
    y_data = data_array[:, 1]  # Signal values
    
    ax.plot(x_data, y_data, 'bo-', markersize=4)
    
    # Add error bars if available (column 3)
    if data_array.shape[1] >= 3:
        error_data = data_array[:, 2]
        ax.errorbar(x_data, y_data, yerr=error_data, fmt='none', ecolor='r', capsize=2)
    
    # Set labels and title
    ax.set_xlabel('Tau')
    ax.set_ylabel('Signal (counts)')
    ax.set_title('Pulsed Measurement Data')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a canvas to display the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add metadata text
    metadata_text = ttk.Label(window, text="Metadata:", font=("TkDefaultFont", 10, "bold"))
    metadata_text.pack(anchor=tk.W, padx=10)
    
    # Add scrollable text widget for metadata
    metadata_frame = ttk.Frame(window)
    metadata_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
    
    metadata_display = tk.Text(metadata_frame, height=5, wrap=tk.WORD)
    scrollbar = ttk.Scrollbar(metadata_frame, command=metadata_display.yview)
    metadata_display.configure(yscrollcommand=scrollbar.set)
    
    metadata_display.pack(side=tk.LEFT, fill=tk.X, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Add metadata to text widget
    metadata_str = ""
    for key, value in metadata.items():
        metadata_str += f"{key}: {value}\n"
    
    metadata_display.insert(tk.END, metadata_str)
    metadata_display.config(state=tk.DISABLED)  # Make read-only
    
    # Close button
    close_button = ttk.Button(window, text="Close", command=window.destroy)
    close_button.pack(pady=10)

def plot_laser_pulse_data(file_path, pulse_index=0):
    """Plot the photon counts for a specific laser pulse"""
    data_array, metadata = load_qudi_data(file_path)
    
    if data_array is None or len(data_array) == 0:
        messagebox.showerror("Error", "No valid data found in file")
        return
    
    # Check if this is a laser pulse file (should have many columns for time bins)
    if "_laser_pulses" not in file_path and "_raw_timetrace" not in file_path:
        messagebox.showerror("Error", "File does not appear to be a laser pulse file")
        return
    
    # Get total number of pulses
    num_pulses = len(data_array)
    
    # Validate pulse index
    if pulse_index < 0 or pulse_index >= num_pulses:
        messagebox.showerror("Error", f"Invalid pulse index. Must be between 0 and {num_pulses-1}")
        return
    
    # Create a new Tkinter window
    window = tk.Toplevel()
    window.title(f"Laser Pulse Data - Pulse #{pulse_index}")
    window.geometry("800x600")
    
    # Create figure and plot
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Extract the data for the specific pulse
    pulse_data = data_array[pulse_index]
    
    # Create x-axis (time bins)
    bin_width = float(metadata.get('bin width (s)', '1e-9'))
    time_axis = np.arange(len(pulse_data)) * bin_width * 1e9  # Convert to ns
    
    # Plot the pulse data
    ax.plot(time_axis, pulse_data, 'g-', linewidth=2)
    ax.fill_between(time_axis, 0, pulse_data, alpha=0.3, color='g')
    
    # Set labels and title
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Laser Pulse #{pulse_index}')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a canvas to display the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Pulse navigation frame
    nav_frame = ttk.Frame(window)
    nav_frame.pack(fill=tk.X, pady=5)
    
    # Previous and Next buttons
    prev_button = ttk.Button(nav_frame, text="Previous Pulse", 
                            command=lambda: change_pulse(-1))
    prev_button.pack(side=tk.LEFT, padx=10)
    
    # Pulse selection
    pulse_var = tk.StringVar(value=f"Pulse: {pulse_index} / {num_pulses-1}")
    pulse_label = ttk.Label(nav_frame, textvariable=pulse_var)
    pulse_label.pack(side=tk.LEFT, expand=True)
    
    # Jump to pulse button
    jump_button = ttk.Button(nav_frame, text="Jump to Pulse", 
                            command=lambda: jump_to_pulse())
    jump_button.pack(side=tk.LEFT, padx=10)
    
    next_button = ttk.Button(nav_frame, text="Next Pulse", 
                            command=lambda: change_pulse(1))
    next_button.pack(side=tk.LEFT, padx=10)
    
    # Close button
    close_button = ttk.Button(window, text="Close", command=window.destroy)
    close_button.pack(pady=10)
    
    # Function to change displayed pulse
    def change_pulse(delta):
        nonlocal pulse_index
        new_index = pulse_index + delta
        if 0 <= new_index < num_pulses:
            pulse_index = new_index
            update_plot()
    
    # Function to jump to a specific pulse
    def jump_to_pulse():
        nonlocal pulse_index
        new_index = askinteger("Jump to Pulse", 
                               f"Enter pulse index (0-{num_pulses-1}):",
                               minvalue=0, maxvalue=num_pulses-1)
        if new_index is not None:
            pulse_index = new_index
            update_plot()
    
    # Function to update the plot with new pulse data
    def update_plot():
        pulse_data = data_array[pulse_index]
        
        # Clear previous plot
        ax.clear()
        
        # Plot new data
        ax.plot(time_axis, pulse_data, 'g-', linewidth=2)
        ax.fill_between(time_axis, 0, pulse_data, alpha=0.3, color='g')
        
        # Update labels and title
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Laser Pulse #{pulse_index}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Update pulse number label
        pulse_var.set(f"Pulse: {pulse_index} / {num_pulses-1}")
        
        # Redraw canvas
        canvas.draw()

def validate_only_signal_files(directory):
    """Validate only the signal data files in the directory"""
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return False
    
    # Find signal files - specifically looking for laser pulses data files
    dat_files = find_files_by_extension(directory, '.dat')
    
    # Check if signal files exist
    print(f"\nChecking directory: {directory}")
    print("-" * 50)
    
    if not dat_files:
        print("❌ No signal data files (.dat) found")
        return False
    else:
        print(f"✅ Found {len(dat_files)} signal data file(s)")
    
    # Validate file contents
    print("\nValidating signal file contents:")
    print("-" * 50)
    
    all_valid = True
    valid_files = []
    invalid_files = []
    
    # Classify files by type
    pulsed_measurement_files = []
    laser_pulse_files = []
    raw_timetrace_files = []
    other_files = []
    
    # Validate each signal file
    for dat_file in dat_files:
        filename = os.path.basename(dat_file)
        dat_valid, dat_msg = validate_signal_data(dat_file)
        
        if dat_valid:
            valid_files.append(dat_file)
            print(f"✅ {filename}: {dat_msg}")
            
            # Classify by filename pattern
            if "_pulsed_measurement.dat" in filename:
                pulsed_measurement_files.append(dat_file)
            elif "_laser_pulses.dat" in filename:
                laser_pulse_files.append(dat_file)
            elif "_raw_timetrace.dat" in filename:
                raw_timetrace_files.append(dat_file)
            else:
                other_files.append(dat_file)
        else:
            invalid_files.append((filename, dat_msg))
            all_valid = False
            print(f"❌ {filename}: {dat_msg}")
    
    # Overall validation result
    print("\nOverall validation result:")
    print("-" * 50)
    
    if all_valid:
        print("✅ All signal files are in correct format for laser pulse data")
    else:
        print("❌ Some files have format issues:")
        for filename, msg in invalid_files:
            print(f"  - {filename}: {msg}")
        
        print("\nFiles with valid format:")
        if valid_files:
            for filename in valid_files:
                print(f"  - {os.path.basename(filename)}")
        else:
            print("  No valid files found")
    
    # Display some helpful information about expected format
    print("\nExpected Qudi data file format:")
    print("-" * 50)
    print("1. File starts with metadata section (# [General])")
    print("2. Metadata contains timestamp, column info, and measurement settings")
    print("3. Data section follows metadata with numeric values")
    print("4. File is produced by Qudi measurement modules")
    print("5. Common files: *_laser_pulses.dat, *_pulsed_measurement.dat, *_raw_timetrace.dat")
    
    # If valid files exist, offer visualization options
    if valid_files:
        print("\nVisualization options:")
        print("-" * 50)
        print("1: View full pulsed measurement plot (tau vs signal)")
        print("2: View individual laser pulse data")
        print("3: Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1" and pulsed_measurement_files:
            # If multiple pulsed measurement files exist, ask which one to visualize
            if len(pulsed_measurement_files) > 1:
                print("\nAvailable pulsed measurement files:")
                for i, file_path in enumerate(pulsed_measurement_files):
                    print(f"{i+1}: {os.path.basename(file_path)}")
                
                file_choice = input(f"\nSelect file (1-{len(pulsed_measurement_files)}): ")
                try:
                    file_index = int(file_choice) - 1
                    if 0 <= file_index < len(pulsed_measurement_files):
                        # Create and show plot
                        root = tk.Tk()
                        root.withdraw()  # Hide the main window
                        plot_full_pulsed_data(pulsed_measurement_files[file_index])
                        root.mainloop()
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                # Only one pulsed measurement file, visualize it directly
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                plot_full_pulsed_data(pulsed_measurement_files[0])
                root.mainloop()
                
        elif choice == "2" and laser_pulse_files:
            # If multiple laser pulse files exist, ask which one to visualize
            if len(laser_pulse_files) > 1:
                print("\nAvailable laser pulse files:")
                for i, file_path in enumerate(laser_pulse_files):
                    print(f"{i+1}: {os.path.basename(file_path)}")
                
                file_choice = input(f"\nSelect file (1-{len(laser_pulse_files)}): ")
                try:
                    file_index = int(file_choice) - 1
                    if 0 <= file_index < len(laser_pulse_files):
                        # Ask for pulse index
                        pulse_choice = input("Enter pulse index (0-based) or press Enter for first pulse: ")
                        try:
                            pulse_index = int(pulse_choice) if pulse_choice.strip() else 0
                            # Create and show plot
                            root = tk.Tk()
                            root.withdraw()  # Hide the main window
                            plot_laser_pulse_data(laser_pulse_files[file_index], pulse_index)
                            root.mainloop()
                        except ValueError:
                            print("Invalid pulse index.")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                # Only one laser pulse file, ask for pulse index
                pulse_choice = input("Enter pulse index (0-based) or press Enter for first pulse: ")
                try:
                    pulse_index = int(pulse_choice) if pulse_choice.strip() else 0
                    # Create and show plot
                    root = tk.Tk()
                    root.withdraw()  # Hide the main window
                    plot_laser_pulse_data(laser_pulse_files[0], pulse_index)
                    root.mainloop()
                except ValueError:
                    print("Invalid pulse index.")
                
        elif choice != "3":
            print("Invalid choice or no suitable files found for the selected option.")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate optical amplitude signal data files")
    parser.add_argument("directory", help="Directory containing the signal data files")
    args = parser.parse_args()
    
    validate_only_signal_files(args.directory)


if __name__ == "__main__":
    main()