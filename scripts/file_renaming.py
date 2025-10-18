import os

def rename_files(folder):
    # Get the folder name
    folder_name = os.path.basename(os.path.normpath(folder))
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder):
        old_path = os.path.join(folder, file_name)

        # Skip if it's a subfolder
        if os.path.isdir(old_path):
            continue

        # Create the new name
        new_name = f"{folder_name}_{file_name}"
        new_path = os.path.join(folder, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

# Replace with your target folder path
target_folder = r"C:\Users\bayli\Documents\CS Demos\BAST_Rivals_2025_Season_1"
rename_files(target_folder)