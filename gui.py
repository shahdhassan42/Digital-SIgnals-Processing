import tkinter as tk
from tkinter import ttk
from main import *

# Create the main GUI window
root = tk.Tk()
root.title("Signal Processing Tool")
root.geometry("600x600")

# Function to open the signal generation menu
def open_signal_generation_menu():
    new_window = tk.Toplevel(root)
    new_window.title("Signal Generation")
    new_window.geometry("500x500")

    tk.Label(new_window, text="Select Signal Type:", font=("Arial", 12)).pack(pady=10)

    signal_type_var = tk.IntVar(value=1)
    sine_rb = tk.Radiobutton(new_window, text="Sine Wave", variable=signal_type_var, value=1)
    cosine_rb = tk.Radiobutton(new_window, text="Cosine Wave", variable=signal_type_var, value=2)
    sine_rb.pack()
    cosine_rb.pack()

    tk.Label(new_window, text="Amplitude (A):", font=("Arial", 12)).pack(pady=5)
    amplitude_entry = tk.Entry(new_window)
    amplitude_entry.pack()

    tk.Label(new_window, text="Phase Shift (Theta, in radians):", font=("Arial", 12)).pack(pady=5)
    phase_entry = tk.Entry(new_window)
    phase_entry.pack()
    tk.Label(new_window, text="(Note: Enter phase shift in radians)", font=("Arial", 10)).pack()

    tk.Label(new_window, text="Analog Frequency (Hz):", font=("Arial", 12)).pack(pady=5)
    analog_freq_entry = tk.Entry(new_window)
    analog_freq_entry.pack()

    tk.Label(new_window, text="Sampling Frequency (Hz):", font=("Arial", 12)).pack(pady=5)
    sampling_freq_entry = tk.Entry(new_window)
    sampling_freq_entry.pack()

    display_type_var = tk.IntVar(value=1)
    continuous_rb = tk.Radiobutton(new_window, text="Continuous", variable=display_type_var, value=1)
    discrete_rb = tk.Radiobutton(new_window, text="Discrete", variable=display_type_var, value=2)
    continuous_rb.pack(pady=5)
    discrete_rb.pack(pady=5)

    def call_gen_signal():
        amplitude = amplitude_entry.get()
        phase = phase_entry.get()
        analog_freq = analog_freq_entry.get()
        sampling_freq = sampling_freq_entry.get()
        signal_type = signal_type_var.get()
        display_type = display_type_var.get()
        generate_signal(amplitude, phase, analog_freq, sampling_freq, signal_type, display_type)

    generate_btn = tk.Button(new_window, text="Generate Signal", command=call_gen_signal)
    generate_btn.pack(pady=20)

# Add the GUI elements
ttk.Label(root, text="Signal Processing Tool", font=("Arial", 20)).grid(row=0, column=1, columnspan=2, pady=30)

# Configure the grid to center buttons
total_columns = 4  # Example: 2 columns for buttons + 2 for padding
for col in range(total_columns):
    root.grid_columnconfigure(col, weight=1)  # Equal weight for all columns to center-align

# List of button labels and corresponding commands
buttons = [
    ("Load Signal", load_signal),
    ("Display Signal", display_signal),
    ("Load Second Signal", load_signal2),
    ("Display Both Signals", display_both_signals),
    ("Add Signals", add_signals),
    ("Multiply Signal", multiply_signal),
    ("Subtract Signals", subtract_signals),
    ("Delay/Advance Signal", delay_advance_signal),
    ("Fold/Reverse Signal", fold_signal),
    ("Signal Generation", open_signal_generation_menu),
    ("Quantize Signal", quantize_signal),
    ("Moving Average", moving_average),
    ("Sharpen", sharpen),
    ("Convolution", convolution),
    ("DFT", dft),
    ("IDFT", idft),
    ("Correlation", correlation),
    ("Classify signals", classify_all_signals),
    ("Filtering", filters),
]

# Create buttons and arrange them in a grid
for index, (text, command) in enumerate(buttons):
    row = (index // 2) + 1  # Start rows from 1 to leave space for the title
    col = (index % 2) + 1  # Shift columns by 1 to account for left padding
    ttk.Button(root, text=text, width=20, command=command).grid(row=row, column=col, padx=10, pady=10)

# Exit button, centered below other buttons
ttk.Button(root, text="Exit", width=20, command=root.quit).grid(
    row=(len(buttons) // 2) + 2, column=1, columnspan=2, pady=20
)

# Run the GUI loop
root.mainloop()
