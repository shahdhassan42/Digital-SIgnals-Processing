import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Radiobutton, IntVar
import numpy as np
from numpy import ndarray
import math
import cmath
import matplotlib.pyplot as plt
import os
import itertools
signal = []
signal2 = []


def generate_signal(amplitude_entry, phase_entry, analog_freq_entry, sampling_freq_entry, signal_type_var,
                    display_type_var):
    try:
        # Convert the string entries to float for numerical operations
        A = float(amplitude_entry)
        theta = float(phase_entry)
        f = float(analog_freq_entry)
        Fs = float(sampling_freq_entry)

        # Nyquist criterion check
        if Fs < 2 * f:
            messagebox.showerror("Error",
                                 "Sampling frequency must be at least twice the analog frequency (Nyquist criterion).")
            return

        t = np.arange(0, 1, 1 / Fs)

        # Generate the signal based on the type
        if signal_type_var == 1:
            signal = A * np.sin(2 * np.pi * f * t + theta)
        else:
            signal = A * np.cos(2 * np.pi * f * t + theta)

        # Plotting the signal
        plt.figure()
        if display_type_var == 1:
            plt.plot(t, signal, label="Signal", color='b')
            plt.title("Continuous Signal")
        else:
            plt.stem(t, signal, linefmt='b-', markerfmt='bo', basefmt='r-')
            plt.title("Discrete Signal")

        # Sample the signal and plot the samples
        sample_times = np.arange(0, 1, 1 / Fs)
        sampled_signal = signal
        plt.scatter(sample_times, sampled_signal, color='red', label='Samples', zorder=5)

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.ylim(-A - 1, A + 1)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for the parameters.")


def load_signal():
    file_path = filedialog.askopenfilename()  # Open file dialog to choose file
    global signal
    signal = []  # Reset signal list

    if file_path:
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Ensure there are at least 3 lines (2 to be ignored, 1 for N)
                if len(lines) < 3:
                    messagebox.showerror("Error", "File format is incorrect. It must contain at least 3 lines.")
                    return

                # Read origin_x, origin_y, and N from the first 3 lines
                origin_x = float(lines[0].strip())  # First line - origin_x
                origin_y = float(lines[1].strip())  # Second line - origin_y
                N = int(lines[2].strip())  # Third line - number of data points

                # Check if we have enough lines for the signal data
                if len(lines) < N + 3:
                    messagebox.showerror("Error", f"File format is incorrect. Expected {N} data points.")
                    return

                # Parse the amplitude and phase shift from the next N lines
                for i in range(3, N + 3):
                    try:
                        # Remove 'f' from each value and split the line into amplitude and phase shift
                        line = lines[i].strip()
                        amplitude, phase_shift = line.split()  # Split by space

                        # Remove the 'f' and convert to float
                        amplitude = float(amplitude.replace('f', '').strip())
                        phase_shift = float(phase_shift.replace('f', '').strip())

                        # Append the tuple (amplitude, phase_shift) directly to signal
                        signal.append((amplitude, phase_shift))  # Appending as a tuple
                    except ValueError:
                        messagebox.showerror("Error",
                                             f"Invalid data on line {i + 1}. Expected 'amplitude phase_shift'.")
                        return

                # At this point, signal should be a list of tuples (amplitude, phase_shift)
                messagebox.showinfo("Success", "Signal loaded successfully!")  # Success message

        except Exception as e:
            messagebox.showerror("Error", f"Error loading signal: {e}")  # Show any other errors


def load_signal2():
    file_path = filedialog.askopenfilename()
    global signal2
    signal2 = []
    if file_path:
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Ensure there are at least 3 lines (2 to be ignored, 1 for N)
                if len(lines) < 3:
                    messagebox.showerror("Error", "File format is incorrect. It must contain at least 3 lines.")
                    return

                # Read origin_x, origin_y, and N from the first 3 lines
                origin_x = float(lines[0].strip())  # First line - origin_x
                origin_y = float(lines[1].strip())  # Second line - origin_y
                N = int(lines[2].strip())  # Third line - number of data points

                # Check if we have enough lines for the signal data
                if len(lines) < N + 3:
                    messagebox.showerror("Error", f"File format is incorrect. Expected {N} data points.")
                    return

                # Parse the amplitude and phase shift from the next N lines
                for i in range(3, N + 3):
                    try:
                        # Remove 'f' from each value and split the line into amplitude and phase shift
                        line = lines[i].strip()
                        amplitude, phase_shift = line.split()  # Split by space

                        # Remove the 'f' and convert to float
                        amplitude = float(amplitude.replace('f', '').strip())
                        phase_shift = float(phase_shift.replace('f', '').strip())

                        # Append the tuple (amplitude, phase_shift) directly to signal
                        signal2.append((amplitude, phase_shift))  # Appending as a tuple
                    except ValueError:
                        messagebox.showerror("Error",
                                             f"Invalid data on line {i + 1}. Expected 'amplitude phase_shift'.")
                        return

                # At this point, signal should be a list of tuples (amplitude, phase_shift)
                messagebox.showinfo("Success", "Signal loaded successfully!")  # Success message

        except Exception as e:
            messagebox.showerror("Error", f"Error loading signal: {e}")  # Show any other errors


def display_signal():
    if signal is None:
        messagebox.showerror("Error", "No signal loaded!")
        return

    selection_window = tk.Toplevel()
    selection_window.title("Choose Display Mode")

    display_mode = tk.StringVar(value="continuous")

    radio1 = tk.Radiobutton(selection_window, text="Continuous", variable=display_mode, value="continuous")
    radio2 = tk.Radiobutton(selection_window, text="Discrete", variable=display_mode, value="discrete")

    radio1.pack(anchor='w')
    radio2.pack(anchor='w')

    def confirm_display():
        choice = display_mode.get()
        indices, values = zip(*signal)
        if choice == "continuous":
            plt.plot(indices, values, marker='o', linestyle='-', color='b')  # Continuous
            plt.title("Continuous Signal")
        else:
            plt.stem(indices, values, linefmt='b-', markerfmt='bo', basefmt='black')  # Discrete
            plt.title("Discrete Signal")

        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.show()
        selection_window.destroy()

    confirm_button = tk.Button(selection_window, text="Confirm", command=confirm_display)
    confirm_button.pack()


def display_both_signals():
    if signal is None or signal2 is None:
        messagebox.showerror("Error", "Both signals must be loaded!")
        return

    selection_window = tk.Toplevel()
    selection_window.title("Choose Display Mode")

    display_mode = tk.StringVar(value="continuous")

    radio1 = tk.Radiobutton(selection_window, text="Continuous", variable=display_mode, value="continuous")
    radio2 = tk.Radiobutton(selection_window, text="Discrete", variable=display_mode, value="discrete")

    radio1.pack(anchor='w')
    radio2.pack(anchor='w')

    def confirm_display_both():
        choice = display_mode.get()
        indices1, values1 = zip(*signal)
        indices2, values2 = zip(*signal2)

        if choice == "continuous":
            plt.plot(indices1, values1, marker='o', linestyle='-', color='b', label='Signal 1')
            plt.plot(indices2, values2, marker='x', linestyle='--', color='r', label='Signal 2')
            plt.title("Continuous Signals")
        else:
            plt.stem(indices1, values1, linefmt='b-', markerfmt='bo', basefmt='black', label='Signal 1')
            plt.stem(indices2, values2, linefmt='r--', markerfmt='rx', basefmt='black', label='Signal 2')
            plt.title("Discrete Signals")

        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.legend()
        plt.show()
        selection_window.destroy()

    confirm_button = tk.Button(selection_window, text="Confirm", command=confirm_display_both)
    confirm_button.pack()


def add_signals():
    signal_dict = {idx: value for idx, value in signal}
    signal2_dict = {idx: value for idx, value in signal2}

    all_indices = set(signal_dict.keys()).union(set(signal2_dict.keys()))

    summed_signal = []
    for idx in sorted(all_indices):
        val1 = signal_dict.get(idx, 0)
        val2 = signal2_dict.get(idx, 0)
        summed_signal.append((idx, val1 + val2))

    summed_signal = summed_signal
    indices = [idx for idx, _ in summed_signal]
    values = [value for _, value in summed_signal]

    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Summed Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Summed Signal Plot')
    plt.grid(True)
    plt.legend()
    plt.show()


def multiply_signal():
    if signal is None:
        messagebox.showerror("Error", "No signal loaded!")
        return

    factor = simpledialog.askfloat("Input", "Enter the multiplication factor:")
    if factor is None:
        messagebox.showwarning("Warning", "Operation canceled. No factor entered.")
        return

    multiplied_signal = [(idx, value * factor) for idx, value in signal]

    indices, values = zip(*multiplied_signal)
    plt.plot(indices, values, marker='o', linestyle='-', color='g', label='Multiplied Signal')
    plt.title("Multiplied Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


def delay_advance_signal():
    if signal is None:
        messagebox.showerror("Error", "No signal loaded!")
        return

    k = simpledialog.askfloat("Input", "Enter the delay/advance steps negative for delay, positive for advance):")
    if k is None:
        messagebox.showwarning("Warning", "Operation canceled. No factor entered.")
        return

    delayed_signal = [(idx - k, value) for idx, value in signal]

    indices, values = zip(*delayed_signal)
    plt.plot(indices, values, marker='o', linestyle='-', color='r', label='Delayed/Advanced Signal')
    plt.title("Delayed/Advanced Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


def subtract_signals():
    signal_dict = {idx: value for idx, value in signal}
    signal2_dict = {idx: value for idx, value in signal2}

    all_indices = set(signal_dict.keys()).union(set(signal2_dict.keys()))

    subtracted_signal = []
    for idx in sorted(all_indices):
        val1 = signal_dict.get(idx, 0)
        val2 = signal2_dict.get(idx, 0)
        subtracted_signal.append((idx, val1 - val2))

    subtracted_signal = subtracted_signal
    indices = [idx for idx, _ in subtracted_signal]
    values = [value for _, value in subtracted_signal]

    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Subtracted Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Subtracted Signal Plot')
    plt.grid(True)
    plt.legend()
    plt.show()


def fold_signal():
    if signal is None:
        messagebox.showerror("Error", "No signal loaded!")
        return

    folded_signal = [(-idx, value) for idx, value in signal]
    folded_signal.sort()

    indices, values = zip(*folded_signal)
    plt.plot(indices, values, marker='o', linestyle='-', color='m', label='Folded Signal')
    plt.title("Folded Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


def quantize_signal():
    if signal is None:
        messagebox.showerror("Error", "No signal loaded!")
        return

    # Ensure signal is a 1D array
    signal_quan = np.asarray(signal).flatten()  # Convert to a NumPy array and flatten

    # Step 1: Get the number of levels or bits
    bits_or_levels = simpledialog.askstring("Quantization", "Enter number of levels or bits (e.g., 4 or 'bits=3'):")
    if bits_or_levels is None:
        return

    # Parse input for levels or bits
    try:
        if "bits=" in bits_or_levels:
            bits = int(bits_or_levels.split('=')[1])
            levels = 2 ** bits
        else:
            levels = int(bits_or_levels)
            bits = int(np.log2(levels))
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number of levels or bits.")
        return

    # Step 2: Find min and max of the signal
    min_val, max_val = np.min(signal_quan), np.max(signal_quan)

    # Step 3: Calculate delta
    delta = (max_val - min_val) / levels

    # Step 4: Create ranges and midpoints for quantization
    ranges = [min_val + i * delta for i in range(levels + 1)]
    midpoints = [(ranges[i] + ranges[i + 1]) / 2 for i in range(levels)]
    midpoints = np.array(midpoints)  # Ensure midpoints is a NumPy array

    # Step 6: Quantize the signal by assigning each sample to the closest midpoint
    quantized_signal = np.zeros_like(signal_quan)
    for i in range(len(signal_quan)):
        sample = signal_quan[i]  # Ensure sample is a scalar value
        closest_index = np.argmin(np.abs(midpoints - sample))  # Find closest midpoint
        quantized_signal[i] = midpoints[closest_index]

    # Step 7: Calculate the average power of the quantization error
    quantization_error = signal_quan - quantized_signal
    avg_power_error = np.mean(quantization_error ** 2)

    # Step 8: Encode quantized values in binary
    binary_encoded = [format(i, f'0{bits}b') for i in
                      [np.argmin(np.abs(midpoints - val)) for val in quantized_signal]]

    # Display the quantized signal and error
    indices = np.arange(len(signal_quan))  # Create an array of indices

    plt.figure(figsize=(12, 8))

    # Original and Quantized Signal
    plt.subplot(3, 1, 1)
    plt.plot(indices, signal_quan, label="Original Signal", color='blue')
    plt.step(indices, quantized_signal, label="Quantized Signal", color='orange', where='mid')
    plt.title("Original and Quantized Signal")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Quantization Error
    plt.subplot(3, 1, 2)
    plt.plot(indices, quantization_error, color='red')
    plt.title("Quantization Error")
    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.grid(True)

    # Binary Encoding for first 10 samples
    plt.subplot(3, 1, 3)
    plt.plot(indices[:10], quantized_signal[:10], 'bo', label="Quantized Levels")
    for i, bin_code in enumerate(binary_encoded[:10]):
        plt.text(indices[i], quantized_signal[i], bin_code, fontsize=10, ha='center', color='purple')
    plt.title("Binary Encoding of Quantized Signal (First 10 Samples)")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Display average power error
    messagebox.showinfo("Quantization Results", f"Average Power Error: {avg_power_error:.4f}")


def moving_average():
    window_size = int(simpledialog.askstring("Average", "Enter the window size"))
    signal_dict = {idx: value for idx, value in signal}

    average_signal = []
    for idx in (sorted(signal_dict.keys())):
        val = 0
        if idx == len(signal_dict) - window_size + 1:
            break
        for i in range(0, window_size):
            val += signal_dict.get(idx + i, 0)
        average_signal.append((idx, val / window_size))

    average_signal = average_signal
    indices = [idx for idx, _ in average_signal]
    values = [value for _, value in average_signal]

    plt.stem(indices, values, linefmt='b-', markerfmt='bo', basefmt='black')  # Discrete
    plt.title("Moving Average")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.show()


def sharpen():
    signal_dict = {idx: value for idx, value in signal}

    first_derivative = []
    for idx in range(1, len(signal_dict)):
        val = 0
        val = signal_dict.get(idx, 0) - signal_dict.get(idx - 1, 0)
        first_derivative.append((idx - 1, val))

    second_derivative = []
    for idx2 in range(1, len(signal_dict) - 1):
        val = 0
        val = signal_dict.get(idx2 + 1, 0) - (2 * signal_dict.get(idx2, 0)) + signal_dict.get(idx2 - 1, 0)
        second_derivative.append((idx2 - 1, val))

    # Plotting
    x_first = [item[0] for item in first_derivative]
    y_first = [item[1] for item in first_derivative]

    x_second = [item[0] for item in second_derivative]
    y_second = [item[1] for item in second_derivative]

    plt.figure(figsize=(10, 6))

    # First Derivative
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
    plt.plot(x_first, y_first, label="First Derivative", color="blue")
    plt.title("First Derivative")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Second Derivative
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
    plt.plot(x_second, y_second, label="Second Derivative", color="red")
    plt.title("Second Derivative")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def convolution(signal, signal2):
    if signal is None or signal2 is None:
        messagebox.showerror("Error", "Both signals must be loaded!")
        return

    # Extract indices and values from the input lists
    signal_indices, signal_values = zip(*signal)
    signal2_indices, signal2_values = zip(*signal2)

    # Determine the first and last indices of both signals
    signal1_first, signal1_last = signal_indices[0], signal_indices[-1]
    signal2_first, signal2_last = signal2_indices[0], signal2_indices[-1]

    # Calculate the convolution range
    start = signal1_first + signal2_first
    end = signal1_last + signal2_last

    # Convert start and end to integers for the range function
    start = int(start)
    end = int(end)

    # Perform convolution using lists
    y = []
    for n in range(start, end + 1):
        counter = 0
        for k, value1 in zip(signal_indices, signal_values):
            if n - k < signal2_first or n - k > signal2_last:
                continue
            index2 = n - k - signal2_first  # Align with signal2's index
            counter += value1 * signal2_values[int(index2)]
        y.append((n, counter))

    # Extract indices and values for plotting
    indices = [idx for idx, _ in y]
    values = [value for _, value in y]

    # Plot the convolution result
    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Summed Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Convolution Output')
    plt.grid(True)
    plt.legend()
    plt.show()

    return y


def dft(signal):
    signal2 = np.array(signal)
    values = signal2[:, 1]
    N = len(values)
    # # DFT computation
    X = np.zeros(N, dtype=complex)
    #
    for k in range(N):
         for n in range(N):
              X[k] += values[n]* np.exp(-2j * np.pi * k * n / N)

    X = np.round(X, decimals=10)

    return X


def idft(X):
    N = len(X)  # Length of the signal
    x_reconstructed = np.zeros(N, dtype=complex)

    # Perform IDFT
    for k in range(N):
        for n in range(N):
            x_reconstructed[n] += X[k] * np.exp(2j * np.pi * k * n / N)

    # Scale by 1/N
    x_reconstructed /= N
    return x_reconstructed.real


def calc_correlation(signal, signal2):
    # Check if signal and signal2 are lists of tuples
    if isinstance(signal, list) and all(isinstance(x, tuple) for x in signal) and \
            isinstance(signal2, list) and all(isinstance(x, tuple) for x in signal2):
        # Extract the second element of each tuple
        signal_values = [x[1] for x in signal]
        signal2_values = [x[1] for x in signal2]
    else:
        # In case signals are not lists of tuples, use them directly
        signal_values = signal
        signal2_values = signal2
    N1 = len(signal)
    N2 = len(signal2)

    if N1 != N2:
        N = N1 + N2 - 1
        signal_values = np.append(signal_values, [0] * (N - N1))
        signal2_values = np.append(signal2_values, [0] * (N - N2))
    else:
        N = N1

    correlation = []
    lags = []

    for k in range(N):
        sum_corr = 0
        for n in range(N):
            valid_index = (n + k) % N
            if 0 <= valid_index < N:
                sum_corr += signal_values[n] * signal2_values[valid_index]
        correlation.append((1 / N) * sum_corr)
        lags.append(k)

    # Normalization
    correlation /= 1 / N * (
        np.sqrt(np.dot(np.dot(signal_values, signal_values), np.dot(signal2_values, signal2_values))))
    return correlation, lags


def correlation():
    correlation, lags = calc_correlation(signal, signal2)

    # if correlation button pressed
    def correlation_output():
        corr_Compare_Signals("CorrOutput.txt", lags, correlation)

        plt.stem(lags, correlation, linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.title("Cross Correlation")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.show()

    # if time delay button pressed
    def time_delay():
        FS = simpledialog.askfloat("Sampling frequency", "Enter sampling frequency: ")
        TS = 1 / FS
        # lag at max correlation
        max_corr_index = np.argmax(correlation)
        lag_max_corr = lags[max_corr_index]

        TD = lag_max_corr * TS
        messagebox.showinfo("Time Delay", f"Time delay= {TD:.4f}")

    # ask Correlation or Time Delay
    selection_window = tk.Toplevel()
    selection_window.title("Correlation or Time Delay?")
    selection_window.geometry("300x150")

    # Correlation button
    correlation_button = tk.Button(selection_window, text="Correlation", command=correlation_output, bg="lightblue",
                                   font=("Arial", 12))
    correlation_button.pack(pady=20)

    # Time Delay button
    time_delay_button = tk.Button(selection_window, text="Time Delay", command=time_delay, bg="lightgreen",
                                  font=("Arial", 12))
    time_delay_button.pack(pady=20)


def read_signals(folder_path):
    signals = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r') as file:
                # Read all lines in the file as a single list
                signal = [int(line.strip()) for line in file]
                signals.append(signal)
    return signals


# Classify a test signal against classes
def classify_signal(test_signal, class1, class2):
    # Split the class signals into individual variables dynamically
    corr_c1 = []
    corr_c2 = []
    for i, signal in enumerate(class1, start=1):
        globals()[f'd{i}'] = signal
        corr_class1, _ = calc_correlation(test_signal, class1)
        corr_c1.append(corr_class1)

    for i, signal in enumerate(class2, start=1):
        globals()[f'u{i}'] = signal
        corr_class2, _ = calc_correlation(test_signal, class2)
        corr_c2.append(corr_class2)

    max_corr_class1 = np.max(corr_c1)
    max_corr_class2 = np.max(corr_c2)

    if max_corr_class1 > max_corr_class2:
        return 1, corr_c1
    else:
        return 2, corr_c2

# Full classification workflow
def classify_all_signals():
    # Read signals
    signals_class1 = read_signals("Class 1")
    signals_class2 = read_signals("Class 2")
    test_signals = read_signals("Test Signals")

    # Classify each test signal
    for i, test_signal in enumerate(test_signals):
        class_label, correlation = classify_signal(test_signal, signals_class1, signals_class2)
        print(f"Test Signal {i + 1} is classified as Class {class_label}.")


def filters():
    def design_filter():
        try:
            fs = float(entry_fs.get())
            stop_attenuation = float(entry_stop_att.get())
            transition_band = float(entry_trans_band.get())
            conv_type=entry_conv_type.get()

            if combo_filter_type.get() in ["Low", "High"]:
                cutoff_frequency = float(entry_fc.get())

            else:

                lower_cutoff = float(entry_f1.get())
                upper_cutoff = float(entry_f2.get())
        except ValueError as e:
            tk.messagebox.showerror("Error", f"Invalid input: {e}")
        delta_f = transition_band / fs
        if stop_attenuation <= 21:
            wf = "rect"
            N = math.ceil(0.9 / delta_f)
        elif stop_attenuation > 21 and stop_attenuation <= 44:
            wf = "hanning"
            N = math.ceil(3.1 / delta_f)
        elif stop_attenuation > 44 and stop_attenuation <= 53:
            wf = "hamming"
            N = math.ceil(3.3 / delta_f)
        elif stop_attenuation > 53 and stop_attenuation <= 74:
            wf = "blackman"
            N = math.ceil(5.5 / delta_f)

        if N % 2 == 0:
            N += 1
        start = -int(N / 2)
        end = int(N / 2)
        h = []

        if (combo_filter_type.get() == "Low"):
            fc = (cutoff_frequency + (transition_band / 2)) / fs
            for n in range(start, end + 1):
                if wf == "rect":
                    w = 1
                    if n == 0:
                        hd = 2 * fc
                    else:
                        hd = 2 * fc * (math.sin(n * 2 * math.pi * fc) / n * 2 * math.pi * fc)
                elif wf == "hanning":
                    w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 2 * fc
                    else:
                        hd = 2 * fc * (math.sin(n * 2 * math.pi * fc) / n * 2 * math.pi * fc)
                elif wf == "hamming":
                    w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 2 * fc
                    else:
                        hd = (2 * fc) / (n * 2 * math.pi * fc) * (math.sin(n * 2 * math.pi * fc))
                elif wf == "blackman":
                    w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / N - 1) + 0.08 * math.cos((4 * math.pi * n) / N - 1)
                    if n == 0:
                        hd = 2 * fc
                    else:
                        hd = 2 * fc * (math.sin(n * 2 * math.pi * fc) / n * 2 * math.pi * fc)
                h.append((n, hd * w))
            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
            Compare_Signals("LPFCoefficients.txt", indices, samples)
            with open('Low pass coefficients.txt', 'w') as f:
                f.write(str(h))

            if conv_type == "Direct":
              conv_res = convolution(signal, h)
              indices = [x[0] for x in conv_res]
              samples = [x[1] for x in conv_res]
              Compare_Signals("ecg_low_pass_filtered.txt", indices, samples)
            else:
              conv_res = fast_convolution(signal, h)
              indices = [x[0] for x in conv_res]
              samples = [x[1] for x in conv_res]
              Compare_Signals("ecg_low_pass_filtered.txt", indices, samples)

        elif (combo_filter_type.get() == "High"):
            fc = (cutoff_frequency - (transition_band / 2)) / fs  # Normalized cutoff frequency for high-pass
            for n in range(start, end + 1):
                if wf == "rect":
                    w = 1
                    if n == 0:
                        hd = 1 - 2 * fc
                    else:
                        hd = -2 * fc * (math.sin(n * 2 * math.pi * fc) / (n * 2 * math.pi * fc))
                elif wf == "hanning":
                    w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 1 - 2 * fc
                    else:
                        hd = -2 * fc * (math.sin(n * 2 * math.pi * fc) / (n * 2 * math.pi * fc))
                elif wf == "hamming":
                    w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 1 - 2 * fc
                    else:
                        hd = (-2 * fc) / (n * 2 * math.pi * fc) * (math.sin(n * 2 * math.pi * fc))
                elif wf == "blackman":
                    w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N - 1)) + 0.08 * math.cos(
                        (4 * math.pi * n) / (N - 1))
                    if n == 0:
                        hd = 1 - 2 * fc
                    else:
                        hd = -2 * fc * (math.sin(n * 2 * math.pi * fc) / (n * 2 * math.pi * fc))
                h.append((n, hd * w))
                # print(f"{n}   {hd*w}")

            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
            Compare_Signals("HPFCoefficients.txt", indices, samples)
            with open('high pass coefficients.txt', 'w') as f:
                f.write(str(h))

            if conv_type == "Direct":
                conv_res = convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_high_pass_filtered.txt", indices, samples)
            else:
                conv_res = fast_convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_high_pass_filtered.txt", indices, samples)

        elif (combo_filter_type.get() == "Band pass"):
            f1 = (lower_cutoff - (transition_band / 2)) / fs
            f2 = (upper_cutoff + (transition_band / 2)) / fs
            for n in range(start, end + 1):
                if wf == "rect":
                    w = 1
                    if n == 0:
                        hd = 2 * (f2 - f1)
                    else:
                        hd = (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2))) - (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1)))
                elif wf == "hanning":
                    w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 2 * (f2 - f1)
                    else:
                        hd = (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2))) - (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1)))
                elif wf == "hamming":
                    w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 2 * (f2 - f1)
                    else:
                        hd = (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2))) - (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1)))
                elif wf == "blackman":
                    w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N - 1)) + 0.08 * math.cos(
                        (4 * math.pi * n) / (N - 1))
                    if n == 0:
                        hd = 2 * (f2 - f1)
                    else:
                        hd = (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2))) - (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1)))
                h.append((n, hd * w))
                # print(f"{n}   {hd*w}")
            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
            Compare_Signals("BPFCoefficients.txt", indices, samples)
            with open('band pass coefficients.txt', 'w') as f:
                f.write(str(h))

            if conv_type == "Direct":
                conv_res = convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_band_pass_filtered.txt", indices, samples)
            else:
                conv_res = fast_convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_band_pass_filtered.txt", indices, samples)

        elif (combo_filter_type.get() == "Band stop"):
            f1 = (lower_cutoff + (transition_band / 2)) / fs
            f2 = (upper_cutoff - (transition_band / 2)) / fs
            for n in range(start, end + 1):
                if wf == "rect":
                    w = 1
                    if n == 0:
                        hd = 1 - 2 * (f2 - f1)
                    else:
                        hd = (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1))) - (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2)))
                elif wf == "hanning":
                    w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 1 - 2 * (f2 - f1)
                    else:
                        hd = (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1))) - (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2)))
                elif wf == "hamming":
                    w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N)
                    if n == 0:
                        hd = 1 - 2 * (f2 - f1)
                    else:
                        hd = (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1))) - (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2)))
                elif wf == "blackman":
                    w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N - 1)) + 0.08 * math.cos(
                        (4 * math.pi * n) / (N - 1))
                    if n == 0:
                        hd = 1 - 2 * (f2 - f1)
                    else:
                        hd = (2 * f1 * (math.sin(n * 2 * math.pi * f1) / (n * 2 * math.pi * f1))) - (2 * f2 * (math.sin(n * 2 * math.pi * f2) / (n * 2 * math.pi * f2)))
                h.append((n, hd * w))
                # print(f"{n}   {hd*w}")
            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
            Compare_Signals("BSFCoefficients.txt", indices, samples)
            with open('band stop coefficients.txt', 'w') as f:
                f.write(str(h))

            if conv_type == "Direct":
                conv_res = convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_band_stop_filtered.txt", indices, samples)
            else:
                conv_res = fast_convolution(signal, h)
                indices = [x[0] for x in conv_res]
                samples = [x[1] for x in conv_res]
                Compare_Signals("ecg_band_stop_filtered.txt", indices, samples)

    def specifications():
        specifications_window = tk.Toplevel()
        specifications_window.title("Specifications")
        specifications_window.geometry("350x300")

        tk.Label(specifications_window, text="Sampling Frequency (Hz):").grid(row=0, column=0, padx=10, pady=5,
                                                                              sticky="w")
        global entry_fs
        entry_fs = tk.Entry(specifications_window)
        entry_fs.grid(row=0, column=1, padx=10, pady=5)

        if combo_filter_type.get() in ["Low", "High"]:
            tk.Label(specifications_window, text="Cut-off Frequency (Hz):").grid(row=2, column=0, padx=10, pady=5,
                                                                                 sticky="w")
            global entry_fc
            entry_fc = tk.Entry(specifications_window)
            entry_fc.grid(row=2, column=1, padx=10, pady=5)
        else:
            tk.Label(specifications_window, text="Lower Cut-off Frequency (Hz):").grid(row=3, column=0, padx=10, pady=5,
                                                                                       sticky="w")
            global entry_f1
            entry_f1 = tk.Entry(specifications_window)
            entry_f1.grid(row=3, column=1, padx=10, pady=5)

            tk.Label(specifications_window, text="Upper Cut-off Frequency (Hz):").grid(row=4, column=0, padx=10, pady=5,
                                                                                       sticky="w")
            global entry_f2
            entry_f2 = tk.Entry(specifications_window)
            entry_f2.grid(row=4, column=1, padx=10, pady=5)

        tk.Label(specifications_window, text="Stop Attenuation (dB):").grid(row=5, column=0, padx=10, pady=5,
                                                                            sticky="w")
        global entry_stop_att
        entry_stop_att = tk.Entry(specifications_window)
        entry_stop_att.grid(row=5, column=1, padx=10, pady=5)

        tk.Label(specifications_window, text="Transition Band Width (Hz):").grid(row=6, column=0, padx=10, pady=5,
                                                                                 sticky="w")
        global entry_trans_band
        entry_trans_band = tk.Entry(specifications_window)
        entry_trans_band.grid(row=6, column=1, padx=10, pady=5)

        tk.Label(specifications_window, text="Convolution Type:").grid(row=7, column=0, padx=10, pady=5,
                                                                                 sticky="w")
        global entry_conv_type
        entry_conv_type = ttk.Combobox(specifications_window, values=["Direct", "Fast"],
                                         state="readonly")
        entry_conv_type.grid(row=7, column=1, padx=10, pady=5)

        # Button to design filter
        btn_design = tk.Button(specifications_window, text="Design Filter", command=design_filter)
        btn_design.grid(row=8, column=0, columnspan=2, pady=20)

    # Filter type selection
    type_window = tk.Toplevel()
    type_window.title("Filter Type")
    type_window.geometry("250x150")

    tk.Label(type_window, text="Filter Type:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    global combo_filter_type
    combo_filter_type = ttk.Combobox(type_window, values=["Low", "High", "Band pass", "Band stop"], state="readonly")
    combo_filter_type.grid(row=1, column=1, padx=10, pady=5)

    # Button to input specifications
    btn_specifications = tk.Button(type_window, text="Input Specifications", command=specifications)
    btn_specifications.grid(row=2, column=0, columnspan=2, pady=20)

def fast_convolution(signal1, signal2):
    # Extract indices and values from the input lists
    signal_indices, signal_values = zip(*signal)
    signal2_indices, signal2_values = zip(*signal2)

    # Determine the first and last indices of both signals
    signal1_first, signal1_last = signal_indices[0], signal_indices[-1]
    signal2_first, signal2_last = signal2_indices[0], signal2_indices[-1]

    # Calculate the convolution range
    start = signal1_first + signal2_first
    end = signal1_last + signal2_last

    # Convert start and end to integers for the range function
    start = int(start)
    end = int(end)

    N1 = len(signal1)
    N2 = len(signal2)
    padded_signal1 = np.pad(signal1, (0, N1 + N2 - 1 - N1), 'constant')
    padded_signal2 = np.pad(signal2, (0, N1 + N2 - 1 - N2), 'constant')

    fft_signal1 = dft(padded_signal1)
    fft_signal2 = dft(padded_signal2)

    result_fft = fft_signal1 * fft_signal2

    convolution_result = idft(result_fft).real
    final = []
    indices = []
    for i,item in itertools.zip_longest(range(start,end+1),convolution_result,fillvalue=None):
        indices.append(i)
        final.append((i,item))

    # Plot the convolution result
    plt.plot(indices, convolution_result, marker='o', linestyle='-', color='b', label='Summed Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Convolution Output')
    plt.grid(True)
    plt.legend()
    plt.show()

    return final

# correlation
def corr_Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one")
            return
    print("Correlation Test case passed successfully")


# Fliters
def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")