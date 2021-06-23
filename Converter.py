import csv
import pandas as pd

print("---------Conversion mode menu---------")
print("-Specification of the file name-")
print("-The relevant data are located in the folder data/toConvert-")
dataSet = input("Please enter the name of the file:")

df_convertedStops = pd.DataFrame(columns=["stopIdentifier", "stopNr", "Longitude", "Latitude", "DemandWeight", "DemandVolume", "BoxAmount", "TourStopId"])

print("Trying to open the file...")
with open('data/toConvert/' + dataSet + '.csv', 'r') as file:
    print("Successfully opened the file...")
    csv_reader = csv.reader(file)
    # row_count = sum(1 for row in csv_reader)
    next(csv_reader, None)
    stopNr = 0  # 0 equals Microhub
    hash_ids = []
    hash_id_counter = dict()
    print("Starting to convert...")
    for row in csv_reader:
        import hashlib
        addedhash = str(row[0]) + str(row[1]) + str(row[5])
        hashobject = hashlib.md5(addedhash.encode())
        hasher = hashobject.hexdigest()
        outputid = hasher
        if (hasher in hash_ids):
            if hasher in hash_id_counter:
                hash_id_counter[hasher] += 1
                outputid = '{}/{}'.format(str(hasher), str(hash_id_counter.get(hasher)))
            else:
                hash_id_counter[hasher] = 1
                outputid = '{}/{}'.format(str(hasher), str(hash_id_counter.get(hasher)))
        df_convertedStops.loc[stopNr] = [outputid, stopNr, row[0], row[1], row[2], row[3], row[4], row[6]]
        stopNr += 1
        hash_ids.append(outputid)
        print("Converted ", stopNr, " stop...")

df_convertedStops.to_csv(r'data/stops/' + dataSet + '.csv', index=False)
print("Succesfully converted the file")
print("Saved to: data/stops/" + dataSet + ".csv")
