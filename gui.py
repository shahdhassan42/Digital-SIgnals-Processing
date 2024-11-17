import tkinter as tk
from tkinter import ttk
from main import *
# Create the main GUI window
root = tk.Tk()
root.title("Signal Processing Tool")
root.geometry("800x800")
def open_signal_generation_menu():
    new_window = Toplevel(root)
    new_window.title("Signal Generation")
    new_window.geometry("500x500")

    tk.Label(new_window, text="Select Signal Type:", font=("Arial", 12)).pack(pady=10)

    signal_type_var = IntVar(value=1)
    sine_rb = Radiobutton(new_window, text="Sine Wave", variable=signal_type_var, value=1)
    cosine_rb = Radiobutton(new_window, text="Cosine Wave", variable=signal_type_var, value=2)
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

    display_type_var = IntVar(value=1)
    continuous_rb = Radiobutton(new_window, text="Continuous", variable=display_type_var, value=1)
    discrete_rb = Radiobutton(new_window, text="Discrete", variable=display_type_var, value=2)
    continuous_rb.pack(pady=5)
    discrete_rb.pack(pady=5)

    def call_gen_signal():
        # Get values from the widgets
        amplitude = amplitude_entry.get()  # Get string input from amplitude Entry
        phase = phase_entry.get()  # Get string input from phase Entry
        analog_freq = analog_freq_entry.get()  # Get string input from analog frequency Entry
        sampling_freq = sampling_freq_entry.get()  # Get string input from sampling frequency Entry
        signal_type = signal_type_var.get()  # Get value from signal type Radiobutton
        display_type = display_type_var.get()  # Get value from display type Radiobutton

        # Pass the extracted values to the generate_signal function
        generate_signal(amplitude, phase, analog_freq, sampling_freq, signal_type, display_type)

    generate_btn = tk.Button(new_window, text="Generate Signal", command=call_gen_signal)
    generate_btn.pack(pady=20)
# Add the GUI elements
ttk.Label(root, text="Signal Processing Tool", font=("Arial", 20)).pack(pady=20)

# Buttons for various signal processing actions
ttk.Button(root, text="Load Signal", width=20, command=load_signal).pack(pady=10)
ttk.Button(root, text="Display Signal", width=20, command=display_signal).pack(pady=10)
ttk.Button(root, text="Load Second Signal", width=20, command=load_signal2).pack(pady=10)
ttk.Button(root, text="Display Both Signals", width=20, command=display_both_signals).pack(pady=10)
ttk.Button(root, text="Add Signals", width=20, command=add_signals).pack(pady=10)
ttk.Button(root, text="Multiply Signal", width=20, command=multiply_signal).pack(pady=10)
ttk.Button(root, text="Subtract Signals", width=20, command=subtract_signals).pack(pady=10)
ttk.Button(root, text="Delay/Advance Signal", width=20, command=delay_advance_signal).pack(pady=10)
ttk.Button(root, text="Fold/Reverse Signal", width=20, command=fold_signal).pack(pady=10)
ttk.Button(root, text="Signal Generation", width=20, command=open_signal_generation_menu).pack(pady=10)
ttk.Button(root, text="Quantize Signal", width=20, command=quantize_signal).pack(pady=10)
ttk.Button(root, text="Moving Average", width=20, command=moving_average).pack(pady=10)
ttk.Button(root, text="Sharpen", width=20, command=sharpen).pack(pady=10)
ttk.Button(root, text="Convolution", width=20, command=convolution).pack(pady=10)

ttk.Button(root, text="Exit", width=20, command=root.quit).pack(pady=10)

# Function to open the signal generation menu



# Run the GUI loop
root.mainloop()