import csv
import pandas as pd
from datetime import datetime

print("---------Conversion mode menu---------")
print("-Specification of the file name-")
print("-The relevant data are located in the folder data/toConvert-")
dataSet = input("Please enter the name of the file:")

df_convertedStops = pd.DataFrame(columns=["stopIdentifier", "stopNr", "Longitude", "Latitude", "DemandWeight", "DemandVolume", "BoxAmount"])

print("Trying to open the file...")
with open('data/toConvert/' + dataSet + '.csv', 'r') as file:
    print("Successfully opened the file...")
    csv_reader = csv.reader(file)
    # row_count = sum(1 for row in csv_reader)
    next(csv_reader, None)
    stopNr = 0  # 0 equals Microhub
    print("Starting to convert...")
    for row in csv_reader:
        identifierUnique = hash(float(row[2]) + float(row[1]))
        df_convertedStops.loc[stopNr] = [identifierUnique, stopNr, row[0], row[1], row[2], row[4]]
        stopNr += 1
        print("Converted ", stopNr, " stop...")

now = datetime.now().strftime('%Y-%m-%d-%H-%M')
df_convertedStops.to_csv(r'data/stops/' + dataSet + '.csv', index=False)
print("Succesfully converted the file")
print("Saved to: data/stops/stops_" + dataSet + ".csv")
