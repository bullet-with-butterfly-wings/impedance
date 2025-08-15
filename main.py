import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.optimize import curve_fit
from pathlib import Path
import csv
import json
from pandas import read_csv

def trigo(t, A, f, phi, C):
    return A*np.cos(2*np.pi*f*t+phi)+C

def get_amplitudes(file, freq, plot=False): #return amplitudes and phases of input channels
    data = read_csv(file).to_numpy().transpose()
    times = data[0]
    channels = data[1:]
    waves_param = [] #[(amplitude, phase), ...]
    for i, ch in enumerate(channels):
        try:
            fit = curve_fit(trigo, times, ch, p0=(0.2, freq, 0, 0), bounds=([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf]))[0]
            waves_param.append((abs(fit[0]), fit[2]))  # (amplitude, phase)
        except:
            plt.plot(times, ch)
            plt.title(f"Check - ch{i}")
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            plt.show()
            
        if plot:
            plt.plot(times, ch)
            plt.plot(times, trigo(times, *fit))
            plt.title(f"Check - ch{i}")
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            plt.show()

    return waves_param



def analyze_data_series(folder, file_name, frequencies, factor = 1):
    Z_measurements = []
    phases = []
    folder_path = Path(folder)
    csv_files = sorted([f.name for f in folder_path.glob("*.csv")], key=lambda x: int(x.split('.')[0][len(file_name):]))[:len(frequencies)]  # Sort by file name
    # Compute impedance
    gains180 = [181.6536993666521, 181.10908318947835, 180.96822135880151, 181.32461127420774, 180.54012545048073, 179.36427365243014, 178.56808980776844, 177.76632097606324, 176.72510785593596, 175.35433526281867, 173.97188127573085, 172.3127814993842, 173.2785831431388, 171.9525155635197, 168.88238440784636, 165.81941763054135, 149.37926224065077]
    #gains180 = [182.51955147574677, 182.51955147574677, 181.8320770919205, 181.34154982327615, 181.38252951812666, 181.68922680554016, 181.9497024756988, 182.46765692068846, 181.25946298434604, 181.0280981307654, 180.80067635424209, 180.43748186173985, 179.89016706227116, 179.8337629063515, 179.00179381599403, 178.72165388506647, 177.61780319319155, 176.3353708475552, 175.34278572228703, 174.22525141391188, 140.10504931032926]
    gains1 = [0.9513819959418973, 0.9463947416868762, 0.9520177501876951, 0.953709070967034, 0.9537698470654291, 0.9530382833871122, 0.9522255201956997, 0.954260321959585, 0.9546127135093464, 0.9559477882195507, 0.9589102019192043, 0.9589328190616203, 0.9622162366585242, 0.9642881683729931, 0.9673066979017761, 0.969709262712179, 0.9666652523357678]
    phase_offset180 = np.array([-0.0003870482392201863, -0.00720423664239167, -0.020625678802741687, -0.039448373306188556, -0.06738126939749889, -0.1467471756302834, -0.1495107260523263, -0.17384981970832603, -0.2140852382648648, -0.23872976102776855, -0.27165775409860626, -0.30371609170047553, -0.33942869303788803, -0.39095311100437935, -0.41592424760335156, -0.45375032637816337, -0.4900745007808015, -0.5343277454404249, -0.5716723352235936, -0.6111882293028771, -0.7940941219966424])
    phase_offset1 = np.array([0.24690965130714648, 0.9872997104459099, 0.9818871046337448, 0.9781583636852881, 0.5971666596069256, 0.9744872179022416, 0.9718756602648132, 0.9699455827450802, 0.9703643762865761, 0.9643701255783912, 0.9636871608886713, 0.9606462808534766, 0.9583426620171196, 0.9565969253842098, 0.9329721736958403, 0.9512475723964606, 0.9501777210348147, 0.9444345743958644, 0.9377785945850461, 0.9313174348425736, 0.8938060482432211])
    phase_offset1 = np.array([2.5686166531672094e-14, -0.006594028573514876, -0.019533181140972244, -0.020048269822829257, -0.0493457550382157, -0.10021537319108287, -0.11314947360568861, -0.13725856389063829, -0.16356325583326803, -0.18042384731705008, -0.21744652953525279, -0.22181631774375177, -0.2451561834154079, -0.2667650364216836, -0.29582125146413496, -0.30484543824006116, -0.3262611601842542, -0.35160588853579555, -0.38799121275596393, -0.43843537273253164, -0.6671343813829523])
    phase_offset = [-0.03470476, -0.08452138, -0.10320951, -0.13357745, -0.16950331,-0.18035703, -0.23243813, -0.26523342, -0.31183017, -0.3296347 ,-0.36086979, -0.39439893, -0.36836078, -0.3756529 , -0.44683867,-0.50340868, -0.6228643 ]
    #phase_offset = phase_offset1 - phase_offset180
    gains1 = gains1
    gains180 = gains180
    #phase_offset = phase_offset[4:]
    print(len(gains1))
    print(len(gains180))
    print(len(phase_offset))
    for i, file in enumerate(csv_files):
        print(frequencies[i])
        waves_param = get_amplitudes(folder_path / file, freq=frequencies[i], plot=False)
        ref_amp, ref_phase = waves_param[0]
        meas_amp, meas_phase = waves_param[1]
        #gain = 1 + 6000 / 33.4 
        current = (1/gains1[i])*ref_amp / 33.0  # Current in Amperes
        voltage_drop = meas_amp/gains180[i]
        Z =  factor * voltage_drop / current  # Impedance in Ohms
        phase = meas_phase - ref_phase - phase_offset[i]  #maybe
        Z_measurements.append(Z)
        phases.append(phase)
    return Z_measurements, phases


def gain_measurements(folder, file_name, freq, save = False, plot = True):
    folder_path = Path(folder)
    phase_lag = []  # output - input
    gain = []       # output/input

    csv_files = sorted(
        [f.name for f in folder_path.glob("*.csv")],
        key=lambda x: int(x.split('.')[0][len(file_name):])
    )
    print(len(freq))
    print(len(csv_files))
    for i, file in enumerate(csv_files):
        waves_param = get_amplitudes(folder_path / file, freq=freq[i], plot=False)
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

def compute_noise(resistivities, frequencies):
    kb = 1.380649e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in K
    d = 75e-6  # Thickness of the sheet in meters (assumed)
    S = kb*T*resistivities*d/(2*np.pi*d**3) # Johnson-Nyquist noise formula
    e = 1.602176634e-19  # Elementary charge in C
    m_Ca = 6.655e-26  # Mass of Calcium ion in kg
    h_bar = 1.0545718e-34  # Reduced Planck's constant in J*s
    heating_rate = (S * e**2) / (4 * m_Ca * h_bar * frequencies * 2 * np.pi)  # Heating rate in K/s
    return S, heating_rate

# Rigid values
frequencies = [200e3, 400e3, 500e3, 600e3, 700e3, 800e3, 900e3, 1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6]
square_factor = np.pi/np.log(2)

def rectangle_factor(a, b):
    return np.pi/(8 * sum([1/((2*i-1)*np.sinh((2*i-1)*np.pi*b/a)) for i in range(1, 20)]))

rectangle1 = rectangle_factor(2.4, 1.7) #first two traps, t1 and annealed
rectangle2 = rectangle_factor(1.16, 1)
simulations_factor = 2.24 #2.4

"""
g1, p1 = gain_measurements("gain1", "g", freq=frequencies, save = True, plot=False)
g2, p3 = gain_measurements("lowg", "lowg", freq=frequencies, save = True, plot=False)

plt.plot(frequencies, g1, marker='o', label='Gain 180')
plt.plot(frequencies, g2, marker='o', label='Gain New 180')
plt.legend()
plt.show()

plt.plot(frequencies, p1, marker='o', label='Phase 180')
plt.plot(frequencies, p3, marker='o', label='Phase New 180')
plt.legend()
plt.show()
"""

def plot_traps(graphs, master_folder):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for folder, file, tag, factor in graphs:
        Z_measurements, phases = analyze_data_series(master_folder + "/" +folder, file, frequencies, factor=factor)
        Z_measurements = Z_measurements[:-1]
        phases = phases[:-1]
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
    #axes[0].set_ylim(0, 50e-3)
    axes[0].set_title("Impedance vs Frequency (Real part)")
    axes[0].grid(True)
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


def open(folder, file_name):
    Z_measurements, phases = [], []
    frequencies = np.array([200e3, 400e3, 500e3, 600e3, 700e3, 800e3, 900e3, 1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6])  # 3e6, 5e6, 7.5e6, 10e6, 12.5e6, 15e6, 17.5e6, 20e6, 22.5e6, 25e6
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

graphs = [("constant", "t", "Trap Constant", rectangle_factor(1.16, 1)),
          ("annealed", "t", "Trap Unannealed", simulations_factor),
          ("unannealed", "c", "Trap Annealed", simulations_factor)]

plot_traps(graphs, "Traps")