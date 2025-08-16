from pathlib import Path

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


# Example usage
import pandas as pd
from pathlib import Path

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

# Example usage:
