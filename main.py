import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json
from scipy.signal import butter, filtfilt
from pandas import read_csv
from numpy.fft import fft, ifft

frequencies = np.array([200e3, 400e3, 500e3, 600e3, 700e3, 800e3, 900e3, 1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6])
low_frequencies = np.array([0, 10, 100, 1e3, 10e3, 30e3, 100e3])
square_factor = np.pi/np.log(2)

frequencies = low_frequencies

def rectangle_factor(a, b):
    return np.pi/(8 * sum([1/((2*i-1)*np.sinh((2*i-1)*np.pi*b/a)) for i in range(1, 20)]))

rectangle1 = rectangle_factor(2.4, 1.7) #first two traps, t1 and annealed
rectangle2 = rectangle_factor(1.16, 1)
simulations_factor = 2.3 #2.4
simulations_factor_new = 3.5

def trigo(t, A, f, phi, C):
    return A*np.cos(2*np.pi*f*t+phi)+C

def get_amplitudes(file, freq, plot=False): #return amplitudes and phases of input channels
    data = read_csv(file).to_numpy().transpose()
    times = data[0]
    channels = data[1:]
    waves_param = [] #[(amplitude, phase), ...]
    for i, ch in enumerate(channels):
        if freq != 0:
            print(freq)
            # Normalize cutoff frequency
            fs = 100 / (times[100] - times[0])  # Sampling frequency
            try:
                cutoff = 5e6  # Cutoff frequency for low-pass filter
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                # Create Butterworth filter coefficients
                b, a = butter(4, normal_cutoff, btype='low', analog=False)
                
                # Apply the filter with zero-phase distortion
                filtered_signal = filtfilt(b, a, ch)
            except Exception as e:
                print(f"Error applying filter: {e}")
                print(f"Signal unfiltered for frequency {freq} channel {i+1}")
                filtered_signal = ch

            fit = curve_fit(trigo, times, filtered_signal, p0=(0.2, freq, 0, 0), bounds=([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf]))[0]
            waves_param.append((abs(fit[0]), fit[2]))  # (amplitude, phase)
        else:
            waves_param.append((np.mean(ch), 0))
        if plot:
            plt.plot(times, ch)
            plt.title(f"Check - ch{i}")
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            plt.show()
            X = fft(filtered_signal)
            Y = fft(ch)
            N = len(X)
            n = np.arange(N)
            T = N/fs
            plt.xlim(0, 10e6)
            plt.stem(n/T, np.abs(X), label="Filtered", markerfmt='red')
            plt.stem(n/T, np.abs(Y), label = "Original", markerfmt='blue')
            plt.legend()
            plt.show()


    return waves_param



def trap_measurement(folder, file_name, factor = 1):
    Z_measurements = []
    phases = []
    folder_path = Path(folder)
    csv_files = sorted([f.name for f in folder_path.glob("*.csv")], key=lambda x: int(x.split('.')[0][len(file_name):]))[:len(frequencies)]  # Sort by file name
    #load gains
    gain1_folder = "low_frequency1" #"supernew1"
    gain180_folder = "low_frequency180" #"supernew180"
    with open("gains.json", 'r') as f:
        data = json.load(f)
        gains180 = data[gain180_folder]["gain"]
        gains1 = data[gain1_folder]["gain"]

        phase_offset180 = data[gain180_folder]["phase"]
        phase_offset1 = data[gain1_folder]["phase"]
    
    for i, file in enumerate(csv_files):
        waves_param = get_amplitudes(folder_path / file, freq=frequencies[i], plot=False)
        ref_amp, ref_phase = waves_param[0]
        meas_amp, meas_phase = waves_param[1]
        current = (1/gains1[i])*ref_amp / 33.0  # Current in Amperes
        voltage_drop = meas_amp/gains180[i]
        Z =  factor * voltage_drop / current  # Impedance in Ohms
        phase = (meas_phase - phase_offset180[i]) - (ref_phase - phase_offset1[i]) 
        #phase_offset = output - input and i want to get input given i know output = measured => input = output - phase_offset
        Z_measurements.append(Z)
        phases.append(phase)
    return Z_measurements, phases


def gain_measurements(folder, file_name, save = False, plot = True):
    folder_path = Path("Gains/" + folder)
    phase_lag = []  # output - input
    gain = []       # output/input

    csv_files = sorted(
        [f.name for f in folder_path.glob("*.csv")],
        key=lambda x: int(x.split('.')[0][len(file_name):])
    )
    for i, file in enumerate(csv_files):
        waves_param = get_amplitudes(folder_path / file, freq=frequencies[i], plot=False)
        gain.append(waves_param[0][0] / waves_param[1][0])       # output/input
        phase_lag.append(waves_param[0][1] - waves_param[1][1])  # output - input phase

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title(f"Gain vs Frequency")
        ax1.plot(frequencies, gain, marker = "o")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Gain")
        ax1.grid(True)
        ax1.legend()
        ax2.plot(frequencies, phase_lag, marker='o')
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Phase Lag [rad]")
        ax2.legend()
        ax2.set_title(f"Phase Lag vs Frequency")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    if save:
        if Path("gains.json").exists():
            with open("gains.json", 'r') as f:
                results = json.load(f)
        else:
            results = {}

        # Step 2: Add or update dataset
        results[folder] = {
            "gain": gain,
            "phase": phase_lag
        }

        # Step 3: Save merged results
        with open("gains.json", 'w') as f:
            json.dump(results, f, indent=4)

    return gain, phase_lag

def compute_noise(resistivities):
    kb = 1.380649e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in K
    d = 75e-6  # Thickness of the sheet in meters (assumed)
    S = kb*T*resistivities*d/(2*np.pi*d**3) # Johnson-Nyquist noise formula
    e = 1.602176634e-19  # Elementary charge in C
    m_Ca = 6.655e-26  # Mass of Calcium ion in kg
    h_bar = 1.0545718e-34  # Reduced Planck's constant in J*s
    heating_rate = (S * e**2) / (4 * m_Ca * h_bar * frequencies * 2 * np.pi)  # Heating rate in K/s
    return S, heating_rate


g1_graphs = [("supernew1", "Take 4")]#[("gain1", "Take 1"), ("lowg", "Take 2"), ("new1", "Take 3"), ("supernew1", "Take 4"), ("supernew1again", "Take 5")]
g180_graphs = [("supernew180", "Take 3")] #[("gain180", "Take 1"), ("newg180", "Take 2"), ("supernew180", "Take 3")]

#gain_measurements("supernew1again", "g", save=True, plot=True)
def gain_plot(g1_graphs, g180_graphs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for folder, label in g1_graphs:
        g1, phase1 = gain_measurements(folder, "g", save=False, plot=False)
        axes[0, 0].plot(frequencies, g1, marker='o', color = "orange", label=label)
        axes[1, 0].plot(frequencies, phase1, marker='o', color = "orange", label=label)

    for folder, label in g180_graphs:
        g180, phase180 = gain_measurements(folder, "g", save=False, plot=False)
        axes[0, 1].plot(frequencies, g180, marker='o', label=label)
        axes[1, 1].plot(frequencies, phase180, marker='o', label=label)

    axes[0, 0].set_title("G=1")
    axes[0, 0].set_ylabel("Gain")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].set_title("G=180")
    axes[0, 1].set_ylabel("Gain")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].set_title("G=1")
    axes[1, 0].set_ylabel("Phase Offset")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].set_title("G=180")
    axes[1, 1].set_ylabel("Phase Offset")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


#gain_plot(g1_graphs, g180_graphs)

def plot_traps(graphs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for folder, file, tag, factor in graphs:
        Z_measurements, phases = trap_measurement(folder, file, factor=factor)
        # Real part
        axes[0].plot(frequencies[:len(Z_measurements)],
                    Z_measurements * np.cos(phases),
                    marker='o', label=tag)

        # Imaginary part
        axes[1].plot(frequencies[:len(Z_measurements)],
                    Z_measurements * np.sin(phases),
                    marker='o', label=tag)

        # Phase
        axes[2].plot(frequencies[:len(phases)],
                    phases,
                    marker='o', label=tag)
    

    # Formatting
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Re[Z] [Ohm]")
    axes[0].set_title("Impedance vs Frequency (Real part)")
    axes[0].grid(True)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Im[Z] [Ohm]")
    axes[1].set_title("Impedance vs Frequency (Imaginary part)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Phase [rad]")
    axes[2].set_title("Phase vs Frequency")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def open_circuit(folder, file_name):
    Z_measurements, phases = [], []
    folder_path = Path(folder)
    csv_files = sorted([f.name for f in folder_path.glob("*.csv")], key=lambda x: int(x.split('.')[0][len(file_name):]))[:len(frequencies)]  # Sort by file name
    for i, file in enumerate(csv_files):
        data = read_csv(folder_path /file).to_numpy().transpose()
        times, ch1, ch3 = data[:3]
        ref_voltage = curve_fit(trigo, times, ch1, p0= (0.1, frequencies[i], 0, 0))[0] #reference
        total_voltage = curve_fit(trigo, times, ch3, p0=(0.1, frequencies[i], 0, 0))[0] #input
        R_ref = 33.1
        im_Z = R_ref*abs(total_voltage[0]/ref_voltage[0])*np.sin(total_voltage[2] - ref_voltage[2])  # imaginary part of impedance
        re_Z = R_ref*abs(total_voltage[0]/ref_voltage[0])*np.cos(total_voltage[2] - ref_voltage[2])-R_ref  # real part of impedance
        Z_measurements.append(np.sqrt(re_Z**2 + im_Z**2))
        phases.append(np.arctan2(im_Z, re_Z))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Real part
    axes[0].plot(frequencies, Z_measurements*np.cos(phases), marker='o')
    axes[0].set_xscale('log')
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Re(Z) [Ω]")
    axes[0].set_title("Real Part of Impedance")
    axes[0].grid(True)

    # Imaginary part
    def Z_capacitive(f, C):
        return -1 / ( f * C * 2 * np.pi)  # Capacitive impedance formula
    
    popt, pcov = curve_fit(Z_capacitive, frequencies, Z_measurements*np.sin(phases), p0=[1e-9])  # initial guess 1nF
    C_fit = popt[0]

    axes[1].plot(frequencies, Z_measurements*np.sin(phases), marker='o')
    axes[1].plot(frequencies, Z_capacitive(frequencies, C_fit), marker='o', label=f'Capacitance Fit: {C_fit:.2e} F')

    axes[1].set_xscale('log')
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Im(Z) [Ω]")
    axes[1].set_title("Imaginary Part of Impedance")
    axes[1].grid(True)
    axes[1].legend()

    # Phase
    axes[2].plot(frequencies, phases, marker='o')
    axes[2].set_xscale('log')
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Phase [rad]")
    axes[2].set_title("Phase of Impedance")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

graphs = [("Traps/constant", "t", "Trap mid-Annealed", 3.5),
          ("Traps/annealed", "t", "Trap Unannealed", simulations_factor),
          ("Traps/unannealed", "c", "Trap Annealed", simulations_factor),
          #("Calibration/wirebond150m", "w", "Wirebond 150m", 1),
          ("Calibration/20m", "r", "Resistor 20m", 1)]

graphs = [("Calibration/low20m", "r", "Resistor 20m", 1)]

plot_traps(graphs)
