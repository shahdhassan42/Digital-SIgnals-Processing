import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Radiobutton, IntVar
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import math
import cmath
import matplotlib.pyplot as plt


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
    signal2=[]
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
    plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Summed Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Convolution Output')
    plt.grid(True)
    plt.legend()
    plt.show()



def dft():
    signal_dict = {idx: value for idx, value in signal}
    N = len(signal_dict)
    # DFT computation
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
             X[k] += signal_dict.get(n, 0) * np.exp(-2j * np.pi * k * n / N)

    X = np.round(X, decimals=10)
    dft_amp = []
    dft_phase = []
    for k, value in enumerate(X):
        val = abs(value)
        dft_amp.append(val)
        x = value.real
        y = value.imag
        phase_shift = math.atan2(y, x)
        dft_phase.append(phase_shift)

    sampling_freq = float(simpledialog.askstring("Input", "Enter the sampling frequency (Hz):"))
    step = (sampling_freq*2*np.pi)/N
    omega = []
    for i in range (0,N):
        omega.append(i*step)
    out_amp, out_phase = zip(*signal2)
    flag = SignalComapreAmplitude(out_amp, dft_amp)
    flag1 = SignalComaprePhaseShift(out_phase,dft_phase)
    if flag and flag1:
        print("DFT test case passed successfully")
    else:
        print("DFT test failed")
    plt.stem(omega, dft_amp, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title("DFT")
    plt.xlabel("Ω")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    plt.stem(omega, dft_phase, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title("DFT")
    plt.xlabel("Ω")
    plt.ylabel("Phase Shift")
    plt.grid(True)
    plt.show()




def idft():
    N = len(signal)  # Length of the signal
    X = np.zeros(N, dtype=complex)
    x_reconstructed = np.zeros(N, dtype=complex)

    # Construct X(k) from amplitude and phase shift
    for k, (amplitude, phase_shift) in enumerate(signal):
        X[k] = amplitude * np.exp(1j * phase_shift)  # Ensure phase_shift is in radians

    # Perform IDFT
    for k in range(N):
        for n in range(N):
            x_reconstructed[n] += X[k] * np.exp(2j * np.pi * k * n / N)

    # Scale by 1/N
    x_reconstructed /= N
    indices =[]
    for i in range(0,N):
        indices.append(i)
    out_amp = [amp for _, amp in signal2]

    if SignalComapreAmplitude(np.round(x_reconstructed.real, decimals=4),out_amp):
        print("IDFT Test passed sucessfully")
    else:
        print("IDFT Test failed")

    plt.stem(indices, np.round(x_reconstructed.real, decimals=4), linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title("Reconstructed signal")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def SignalComapreAmplitude(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i]-SignalOutput[i])>0.001:
                return False
        return True

def RoundPhaseShift(P):
    while P<0:
        P += 2*math.pi
    return float(P % (2*math.pi))

#Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A=round(SignalInput[i])
            B=round(SignalOutput[i])
            if abs(A-B)>0.0001:
                return False
        return True
