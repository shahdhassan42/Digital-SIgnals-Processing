import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Radiobutton, IntVar
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

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
    file_path = filedialog.askopenfilename()
    global signal
    signal=[]
    if file_path:
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
                origin_x = float(lines[0].strip())
                origin_y = float(lines[1].strip())
                N = int(lines[2].strip())

                for i in range(3, N + 3):
                    idx, value = map(float, lines[i].split())
                    signal.append((idx, value))
            messagebox.showinfo("Success", "Signal loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading signal: {e}")


def load_signal2():
    file_path = filedialog.askopenfilename()
    global signal2
    if file_path:
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
                origin_x2 = float(lines[0].strip())
                origin_y2 = float(lines[1].strip())
                N = int(lines[2].strip())
                signal2 = []
                for i in range(3, N + 3):
                    idx, value = map(float, lines[i].split())
                    signal2.append((idx, value))
            messagebox.showinfo("Success", "Second signal loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading second signal: {e}")


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

    if window_size==3:
        CompareSignals("MovingAvg_out1.txt", indices, values)
    elif window_size==5:
        CompareSignals("MovingAvg_out2.txt", indices, values)

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
    CompareSignals("1st_derivative_out.txt", x_first, y_first)
    CompareSignals("2nd_derivative_out.txt", x_second, y_second)


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



def convolution():
    if signal is None or signal2 is None:
        messagebox.showerror("Error", "Both signals must be loaded!")
        return

    signal_dict = {idx: value for idx, value in signal}
    signal2_dict = {idx: value for idx, value in signal2}
    signal1_first = int(next(iter(signal_dict)))
    signal1_last = int(next(reversed(signal_dict)))
    signal2_first = int(next(iter(signal2_dict)))
    signal2_last = int(next(reversed(signal2_dict)))

    start = signal1_first + signal2_first
    end = signal1_last + signal2_last

    y = []
    for n in range(start, end + 1):
        counter = 0
        for k in range(signal1_first, signal1_last + 1):

            if n - k < signal2_first:
                break
            else:
                counter += signal_dict.get(k, 0) * signal2_dict.get(n - k, 0)

        y.append((n, counter))
    y = y
    indices = [idx for idx, _ in y]
    values = [value for _, value in y]
    CompareSignals("Conv_output.txt", indices, values)
    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Summed Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Convolution Output')
    plt.grid(True)
    plt.legend()
    plt.show()



def CompareSignals(file_name, Your_EncodedValues, Your_Values):
    expectedIndices=[]
    expectedValues=[]
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
                V2=int(L[0])
                V3=float(L[1])
                expectedIndices.append(V2)
                expectedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedIndices)) or (len(Your_Values) != len(expectedValues))):

        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedIndices[i]):
            print("Test case failed, your indices have different values from the expected one")
            return
    for i in range(len(expectedValues)):
        if abs(Your_Values[i] - expectedValues[i]) < 0.01:
            continue
        else:
            print("Test case failed, your Values have different values from the expected one")
            return
    print("Test case passed successfully")