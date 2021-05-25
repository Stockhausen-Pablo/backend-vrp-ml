import pandas as pd


def load_memory_df_from_local(pickleName, nodes):
    try:
        df_pickle = pd.read_pickle(pickleName)
        if df_pickle.empty:
            df_new_pickle = pd.DataFrame(index=[node.hashIdentifier for node in nodes],
                                         columns=[node.hashIdentifier for node in nodes])
            df_new_pickle.fillna(value=float(0.0), inplace=True)
            df_new_pickle.to_pickle(pickleName)
            return df_new_pickle
        else:
            for node in nodes:
                print("Filling up the pickle with missing hashIdentifiers")
                if node.hashIdentifier not in df_pickle.index:
                    df_pickle[node.hashIdentifier] = df_pickle.index
            df_pickle.fillna(value=float(0.0), inplace=True)
            df_pickle.to_pickle(pickleName)
            return df_pickle
    except:
        print("-No Pickle file to load from-")
        print("-Creating new Pickle file-")
        df_new_pickle = pd.DataFrame(index=[node.hashIdentifier for node in nodes],
                                     columns=[node.hashIdentifier for node in nodes])
        df_new_pickle.fillna(value=float(0.0), inplace=True)
        df_new_pickle.to_pickle(pickleName)
        return df_new_pickle


def save_memory_df_to_local(pickleName, df_source):
    df_source.to_pickle(pickleName)
