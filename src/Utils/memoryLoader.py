import pandas as pd


def load_memory_df_from_local(pickleName, state_hashes, microhub_hash):
    try:
        df_pickle = pd.read_pickle(pickleName)
        if df_pickle.empty:
            df_pickle = pd.DataFrame(index=state_hashes[1:], columns=state_hashes[1:])
            new_row = pd.Series(name='{}/{}'.format(microhub_hash, 0))
            df_pickle = df_pickle.append(new_row, ignore_index=False)
            df_pickle['{}/{}'.format(microhub_hash, 0)] = 0.0
            df_pickle.fillna(value=0.0, inplace=True)
            df_pickle = df_pickle + (1 / len(state_hashes))
            return df_pickle
        else:
            for state in state_hashes:
                print("Filling up the pickle with missing hashIdentifiers")
                if state == microhub_hash:
                    state = '{}/{}'.format(microhub_hash, 0)
                if state not in df_pickle.index:
                    df_pickle[state] = df_pickle.index
            df_pickle.fillna(value=float(0.0), inplace=True)
            return df_pickle
    except Exception as ex:
        print('Exception occured '+str(ex))
        print("-No Pickle file to load from-")
        print("-Creating new Pickle file-")
        df_new_pickle = pd.DataFrame(index=state_hashes[1:], columns=state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(microhub_hash, 0))
        df_new_pickle = df_new_pickle.append(new_row, ignore_index=False)
        df_new_pickle['{}/{}'.format(microhub_hash, 0)] = 0.0
        df_new_pickle.fillna(value=0.0, inplace=True)
        df_new_pickle = df_new_pickle + (1 / len(state_hashes))
        return df_new_pickle


def save_memory_df_to_local(pickleName, df_source):
    df_source.to_pickle(pickleName)


def create_model_name(microhub_name, capacity_weight, capacity_volume, ml_agent):
    return microhub_name + '_w_' + str(capacity_weight) + '_v_' + str(capacity_volume) + '_a_' + str(ml_agent)
