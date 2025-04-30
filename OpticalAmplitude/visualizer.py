#!/usr/bin/env python3
"""
Visualizer for Qudi data files.
Provides visualization options for pulsed measurement and laser pulse data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.simpledialog import askinteger

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
    
    # Metadata display
    metadata_text = ttk.Label(window, text=f"Bin width: {bin_width*1e9:.3f} ns | Total time: {len(pulse_data)*bin_width*1e9:.1f} ns")
    metadata_text.pack(side=tk.TOP, pady=5)
    
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

class QudioFileVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("Qudi Data File Visualizer")
        root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Qudi Data File Visualizer", 
                               font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_label = ttk.Label(main_frame, text="Select a Qudi data file to visualize", 
                              wraplength=500)
        desc_label.pack(pady=5)
        
        # File frame
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)
        
        # Option frame
        option_frame = ttk.LabelFrame(main_frame, text="Visualization Options")
        option_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Radio buttons for options
        self.option_var = tk.IntVar(value=1)
        
        option1 = ttk.Radiobutton(option_frame, text="Full Pulsed Measurement Plot (tau vs signal)", 
                                 variable=self.option_var, value=1)
        option1.pack(anchor=tk.W, pady=5, padx=10)
        
        option2 = ttk.Radiobutton(option_frame, text="Individual Laser Pulse Data", 
                                 variable=self.option_var, value=2)
        option2.pack(anchor=tk.W, pady=5, padx=10)
        
        # Pulse index option (only enabled for option 2)
        pulse_frame = ttk.Frame(option_frame)
        pulse_frame.pack(fill=tk.X, pady=5, padx=10)
        
        pulse_label = ttk.Label(pulse_frame, text="Pulse Index (0-based):")
        pulse_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.pulse_index_var = tk.StringVar(value="0")
        pulse_entry = ttk.Entry(pulse_frame, textvariable=self.pulse_index_var, width=5)
        pulse_entry.pack(side=tk.LEFT)
        
        # Visualize button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        visualize_button = ttk.Button(button_frame, text="Visualize", 
                                     command=self.visualize_data)
        visualize_button.pack(side=tk.RIGHT)
        
        # Status message
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                foreground="blue")
        status_label.pack(pady=5)
    
    def browse_file(self):
        """Open a file dialog to select a data file"""
        file_path = filedialog.askopenfilename(
            title="Select Qudi Data File",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"Selected file: {os.path.basename(file_path)}")
    
    def visualize_data(self):
        """Visualize the selected data file with the chosen option"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a data file first.")
            return
        
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        option = self.option_var.get()
        
        if option == 1:  # Full pulsed measurement plot
            # Check if file is a pulsed measurement file
            if "_pulsed_measurement.dat" in file_path:
                plot_full_pulsed_data(file_path)
            else:
                # Ask for confirmation if file doesn't match expected pattern
                response = messagebox.askyesno(
                    "Warning", 
                    "This does not appear to be a pulsed measurement file. Attempt visualization anyway?"
                )
                if response:
                    plot_full_pulsed_data(file_path)
        
        elif option == 2:  # Individual laser pulse data
            # Check if file is a laser pulse file
            if "_laser_pulses.dat" in file_path or "_raw_timetrace.dat" in file_path:
                # Get pulse index
                try:
                    pulse_index = int(self.pulse_index_var.get())
                    plot_laser_pulse_data(file_path, pulse_index)
                except ValueError:
                    messagebox.showerror("Error", "Pulse index must be an integer.")
            else:
                # Ask for confirmation if file doesn't match expected pattern
                response = messagebox.askyesno(
                    "Warning", 
                    "This does not appear to be a laser pulse file. Attempt visualization anyway?"
                )
                if response:
                    try:
                        pulse_index = int(self.pulse_index_var.get())
                        plot_laser_pulse_data(file_path, pulse_index)
                    except ValueError:
                        messagebox.showerror("Error", "Pulse index must be an integer.")

def main():
    root = tk.Tk()
    app = QudioFileVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()