# Code to download the "IDS 2018 Intrusion CSVs" dataset, which is the same used by the team
# The code download the multiple csv files (roughly 7 GB) found at https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv

# RUN THIS IN THE FOLDER YOU WANT TO WORK IN

import kagglehub
import pandas as pd
import os
import shutil
import glob

# Download the folder from the Kagglehub cache
cache_path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
print("Folder downloaded in cache ", cache_path)

# Folder we want the CSVs in, here it is the local folder
local_folder = os.getcwd()  # local folder
print("Work folder :", local_folder)

# Copies the CSV files from the cache to our local folder.
csv_files = glob.glob(os.path.join(cache_path, "*.csv"))

if not csv_files:
    print("No CSV file was found in cache :(")
else:
    for f in csv_files:
        dest = os.path.join(local_folder, os.path.basename(f))
        shutil.copy2(f, dest)
        print(f"Copied {f} into {dest}")

# Just to check that everything went well, we check that the first file of the dataset was properly downloaded and copied in our local folder
csv_path = os.path.join(local_folder, "02-14-2018.csv")

try:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    print("Chargement réussi avec ISO-8859-1")
except UnicodeDecodeError:
    print("Tentative avec cp1252...")
    df = pd.read_csv(csv_path, encoding="cp1252")

print(df.head())