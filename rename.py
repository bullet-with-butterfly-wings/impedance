from pathlib import Path

def rename_g_files(folder):
    folder_path = Path(folder)
    csv_files = sorted(
        [f for f in folder_path.glob("lowg*.csv")],
        key=lambda x: int(x.stem[4:])  # Extract number after 'lowg'
    )

    for i, file in enumerate(csv_files):
        new_name = f"lowg{i}.csv"
        file.rename(folder_path / new_name)
        print(f"Renamed {file.name}  {new_name}")

# Example usage
rename_g_files("lowg")
