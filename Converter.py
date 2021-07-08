import csv
import pandas as pd

print("---------Conversion mode menu---------")
print("-Specification of the file name-")
print("-The relevant data are located in the folder data/toConvert-")
input_data = input("Please enter the name of the file:")

df_converted_stops = pd.DataFrame(columns=["stopIdentifier", "stopNr", "Longitude", "Latitude", "DemandWeight", "DemandVolume", "BoxAmount", "TourStopId"])

print("Trying to open the file...")
with open('data/toConvert/' + input_data + '.csv', 'r') as file:
    print("Successfully opened the file...")
    csv_reader = csv.reader(file)
    next(csv_reader, None)
    stop_number = 0  # 0 equals Microhub
    hash_ids = []
    hash_id_counter = dict()
    print("Starting to convert...")
    for row in csv_reader:
        import hashlib
        hash_base = str(row[0]) + str(row[1]) + str(row[5])
        hash_object = hashlib.md5(hash_base.encode())
        hash_id = hash_object.hexdigest()
        final_id = hash_id

        if hash_id in hash_ids:
            if hash_id in hash_id_counter:
                hash_id_counter[hash_id] += 1
                final_id = '{}/{}'.format(str(hash_id), str(hash_id_counter.get(hash_id)))
            else:
                hash_id_counter[hash_id] = 1
                final_id = '{}/{}'.format(str(hash_id), str(hash_id_counter.get(hash_id)))

        df_converted_stops.loc[stop_number] = [final_id, stop_number, row[0], row[1], row[2], row[3], row[4], row[6]]
        stop_number += 1
        hash_ids.append(final_id)
        print("Converted ", stop_number, " stop...")

df_converted_stops.to_csv(r'data/stops/' + input_data + '.csv', index=False)
print("Successfully converted the file")
print("Saved to: data/stops/" + input_data + ".csv")
