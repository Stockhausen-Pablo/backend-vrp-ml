import pandas as pd
import json


def load_memory_df_from_local(pickle_name, state_hashes, microhub_hash):
    try:
        df_pickle = pd.read_pickle(pickle_name)
        print("Found Pickle with name: ", pickle_name)
        if df_pickle.empty:
            print("Creating new Pickle file")
            df_pickle = pd.DataFrame(index=state_hashes[1:], columns=state_hashes[1:])
            new_row = pd.Series(name='{}/{}'.format(microhub_hash, 0))
            df_pickle = df_pickle.append(new_row, ignore_index=False)
            df_pickle['{}/{}'.format(microhub_hash, 0)] = 0.0
            df_pickle.fillna(value=0.0, inplace=True)
            df_pickle = df_pickle + (1 / len(state_hashes))
            return df_pickle
        else:
            for state in state_hashes:
                if state == microhub_hash:
                    state = '{}/{}'.format(microhub_hash, 0)
                if state not in df_pickle.index:
                    print("Filling up the pickle with missing hashIdentifiers")
                    new_row = pd.Series(name=state)
                    df_pickle = df_pickle.append(new_row, ignore_index=False)
                    df_pickle.fillna(value=1 / len(state_hashes), inplace=True)
                    df_pickle[state] = 1 / len(state_hashes)
            df_pickle.fillna(value=float(0.0), inplace=True)
            return df_pickle
    except Exception as ex:
        print('Exception occurred '+str(ex))
        print("No Pickle file to load from")
        print("Creating new Pickle file")
        df_new_pickle = pd.DataFrame(index=state_hashes[1:], columns=state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(microhub_hash, 0))
        df_new_pickle = df_new_pickle.append(new_row, ignore_index=False)
        df_new_pickle['{}/{}'.format(microhub_hash, 0)] = 0.0
        df_new_pickle.fillna(value=0.0, inplace=True)
        df_new_pickle = df_new_pickle + (1 / len(state_hashes))
        return df_new_pickle


def save_memory_df_to_local(pickle_name, df_source):
    df_source.to_pickle(pickle_name)


def create_model_name(microhub_name, capacity_weight, capacity_volume, shipper_name, carrier_name, delivery_date, ml_agent):
    return microhub_name + '_w_' + str(capacity_weight) + '_v_' + str(capacity_volume) + '_s_'+ shipper_name + '_c_' + carrier_name + '_d_' + delivery_date + '_a_' + str(ml_agent)


def export_constructed_tours_to_json(file_name, final_tours):
    with open(r'data/constructed_tours_by_za/'+file_name+'.json', 'w+') as file:
        json_data = json.dumps(final_tours)
        file.write(json_data)
