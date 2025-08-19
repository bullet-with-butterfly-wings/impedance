from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd


def rename_g_files(folder):
    folder_path = Path(folder)
    csv_files = sorted(
        [f for f in folder_path.glob("lowg*.csv")],
        key=lambda x: int(x.stem[4:])  # Extract number after 'lowg'
    )

    for i, file in enumerate(csv_files):
        new_name = f"g{i}.csv"
        file.rename(folder_path / new_name)
        print(f"Renamed {file.name}  {new_name}")



def remove_third_column(folder):
    folder_path = Path(folder)
    csv_files = folder_path.glob("*.csv")
    
    for file in csv_files:
        # Read CSV
        df = pd.read_csv(file)
        
        # Drop the 3rd column (index 2)
        if df.shape[1] >= 3:  
            df.drop(df.columns[2], axis=1, inplace=True)
            # Overwrite file
            df.to_csv(file, index=False)
            print(f"Updated: {file.name}")
        else:
            print(f"Skipped (less than 3 columns): {file.name}")

low_frequencies = np.array([0, 10, 100, 1e3, 10e3, 30e3, 100e3])
frequencies = np.array([200e3, 400e3, 500e3, 600e3, 700e3, 800e3, 900e3, 1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6, 1.6e6, 1.8e6, 2e6])
with open("gains.json", 'r') as f:    
    data = json.load(f)
    phase180 = data["supernew180"]["phase"]
    phase1 = data["supernew1"]["phase"]
#fit linear fit
fit1 = curve_fit(lambda x, a, b: a * x + b, frequencies, phase1, p0= (-1.0e-7, 0))[0]
fit180 = curve_fit(lambda x, a, b: a * x + b, frequencies, phase180, p0= (-1.0e-7, 0))[0]
plt.plot(frequencies, fit1[0] * frequencies + fit1[1], label="1 phase fit")
plt.plot(low_frequencies, fit1[0] * low_frequencies + fit1[1], label="1 phase fit low frequencies")
plt.plot(frequencies, phase1, label="1 phase data")
plt.legend()
plt.show()
plt.plot(frequencies, fit180[0] * frequencies + fit180[1], label="180 phase fit")
plt.plot(low_frequencies, fit180[0] * low_frequencies + fit180[1], label="180 phase fit low frequencies")
plt.plot(frequencies, phase180, label="180 phase data")
plt.legend()
plt.show()


if Path("gains.json").exists():
    with open("gains.json", 'r') as f:
        results = json.load(f)
else:
    results = {}


results["low_frequency1"] = {
    "gain": [0.9766547539586383]*len(low_frequencies),
    "phase": list(fit1[0] * low_frequencies + fit1[1])
}

results["low_frequency180"] = {
    "gain": [181.2099232485082]*len(low_frequencies),
    "phase": list(fit180[0] * low_frequencies + fit180[1])
}


# Step 3: Save merged results
with open("gains.json", 'w') as f:
    json.dump(results, f, indent=4)
